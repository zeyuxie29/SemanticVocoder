
# SemanticVocoder: Audio Autoencoder combining Dasheng encoder and Flow2GAN generator.

# This implementation is inspired by and builds upon two excellent open-source projects:
# - Dasheng: https://github.com/XiaoMi/dasheng (Audio representation learning)
# - Flow2GAN: https://github.com/k2-fsa/Flow2GAN (Audio generation)

# We gratefully acknowledge the original authors and their open-source contributions
# that made this work possible.


from typing import Tuple

import torch
import torch.nn as nn
import torchaudio

from models.autoencoder.autoencoder_base import AutoEncoderBase
from models.common import LoadPretrainedBase
from utils.torch_utilities import create_mask_from_length

class DashengEncoderWrapper(nn.Module):
    """
    Wrapper for Dasheng encoder to be compatible with Flow2GAN framework.
    
    Dasheng encoder outputs:
    - frame: [B, T, D] - frame-level embeddings
    - global: [B, D] - global pooled embedding
    
    We use the frame-level embeddings as the latent representation for audio reconstruction.
    """
    
    def __init__(self, 
        sampling_rate: int = 24000,
        model_name: str = "dasheng_base", 
        latent_dim: int = None, 
        dasheng_sampling_rate: int = 16000, 
        **kwargs):
        """
        Args:
            model_name: Name of the Dasheng model ('dasheng_base', 'dasheng_06B', 'dasheng_12B', 'dasheng_audioset')
            latent_dim: Optional projection dimension for the latent space
            **kwargs: Additional arguments for Dasheng model
        """
        super().__init__()        
        try:
            import dasheng
        except ImportError:
            raise ImportError("dasheng is not installed. Please install it first.")
        
        self.model_name = model_name
        self.latent_dim = latent_dim
        self.sampling_rate = sampling_rate
        self.dasheng_sampling_rate = dasheng_sampling_rate

        # Initialize Dasheng model
        if model_name == 'dasheng_audioset':
            self.dasheng_model = dasheng.dasheng_base()
            self.embed_dim = self.dasheng_model.embed_dim
            # Load pretrained weights for audioset
            checkpoint = torch.hub.load_state_dict_from_url(
                'https://zenodo.org/records/13315686/files/dasheng_audioset_mAP497.pt?download=1',
                map_location='cpu'
            )
            self.dasheng_model.load_state_dict(checkpoint, strict=False) 
        elif model_name == 'dasheng_12B':
            self.dasheng_model = dasheng.dasheng_12B()
            self.embed_dim = self.dasheng_model.embed_dim
        elif model_name == 'dasheng_06B':
            self.dasheng_model = dasheng.dasheng_06B()
            self.embed_dim = self.dasheng_model.embed_dim
        elif model_name == 'dasheng_base':
            self.dasheng_model = dasheng.dasheng_base()
            self.embed_dim = self.dasheng_model.embed_dim
        elif model_name == 'none':
            # During inference, encoder is not used.
            self.dasheng_model = None
            self.embed_dim = None
        else:
            raise ValueError(f"Unknown Dasheng model name: {model_name}")
        
        # Optional projection to latent_dim
        if latent_dim is not None and latent_dim != self.embed_dim:
            self.proj_out = nn.Conv1d(self.embed_dim, latent_dim, kernel_size=1)
            self._latent_dim = latent_dim
        else:
            self.proj_out = nn.Identity()
            self._latent_dim = self.embed_dim
    
    @property
    def out_channels(self):
        """Return output channels (latent dimension)"""
        return self._latent_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] - input audio waveform
            
        Returns:
            latent: [B, D, T'] - frame-level latent representations
        """

        # resampler if needed
        if self.sampling_rate != self.dasheng_sampling_rate:
            x = torchaudio.functional.resample(
                x, orig_freq=self.sampling_rate, new_freq=self.dasheng_sampling_rate
            )

        # x: [B, C, T] -> [B, T] - squeeze channel dimension
        if x.dim() == 3:
            x = x.mean(dim=1)
        outputs = self.dasheng_model(x)

        # outputs: [B, T, D] - frame embeddings
        # For 10 seconds, audioset, base [N, 250, 768]
        # Transpose to [B, D, T] for consistency with other encoders
        latent = outputs.transpose(1, 2)
        
        # Apply optional projection
        latent = self.proj_out(latent)
        
        return latent


class SemanticVocoder(LoadPretrainedBase, AutoEncoderBase):
    """
    SemanticVocoder as an AutoEncoder for audio reconstruction.
    
    Uses a condition module (DashengEncoder) as encoder
    and generator (Flow2GAN vocoder) as decoder.
    """
    
    def __init__(
        self,
        vocoder,
        encoder_name: str = "dasheng_base",
        checkpoint: str = None,
        n_timesteps: int = 200,
        sample_rate: int = 24000,
        downsampling_ratio: int = 960,
        encoder_sampling_rate: int = 16000,
        clamp_pred: bool = True,
    ):
        """
        Args:
            vocoder: Flow2GAN vocoder model instance
            encoder_name: Encoder name, e.g., 'dasheng_base', 'dasheng_audioset'
            checkpoint: Path to vocoder checkpoint
            n_timesteps: Number of inference steps (1, 2, 4, etc.)
            sample_rate: Audio sample rate
            downsampling_ratio: Downsampling ratio
            encoder_sampling_rate: Mae encoder model sampling rate
            clamp_pred: Whether to clamp predictions during inference
        """
        # Validate downsampling ratio
        assert downsampling_ratio == sample_rate // 25, \
            f"Downsampling ratio {downsampling_ratio} does not match sample rate {sample_rate}"
        
        self.checkpoint = checkpoint
        self.n_timesteps = n_timesteps
        self.clamp_pred = clamp_pred
        self.sample_rate = sample_rate
        self.downsampling_ratio = downsampling_ratio
        
        LoadPretrainedBase.__init__(self)
        AutoEncoderBase.__init__(
            self,
            downsampling_ratio=downsampling_ratio,
            sample_rate=sample_rate,
            latent_shape=(None, None)
        )
        
        self.io_channels = 1
        self.in_channels = 1
        self.out_channels = 1
        self.min_length = downsampling_ratio
        
        # For consistency with V1
        self.cond_module =  DashengEncoderWrapper(
            sampling_rate=self.sample_rate,
            model_name=encoder_name,
            dasheng_samping_rate=encoder_sampling_rate,
        )
        self.generator = vocoder

        # Load vocoder checkpoint if provided
        if checkpoint is not None:
            self._load_checkpoint(checkpoint)
    
    def _load_checkpoint(self, checkpoint_path: str, strict: bool = True):
        """Load checkpoint for the vocoder/generator."""
        #print(f"Loading vocoder checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.generator.load_state_dict(checkpoint["model"], strict=strict)
    
    @property
    def device(self):
        """Get the device of the model."""
        return next(self.parameters()).device
    
    def to(self, device):
        """Move model to device."""
        super().to(device)
        self.cond_module = self.cond_module.to(device)
        self.generator = self.generator.to(device)
        return self
    
    def eval(self):
        """Set model to evaluation mode."""
        super().eval()
        self.cond_module.eval()
        self.generator.eval()
        return self
    
    def train(self, mode: bool = True):
        """Set model to training mode."""
        super().train(mode)
        self.cond_module.train(mode)
        self.generator.train(mode)
        return self
    
    def encode(
        self, waveform: torch.Tensor, waveform_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode waveform to latent representation using condition module.
        
        Args:
            waveform: [B, C, T] or [B, T] - input audio waveform
            waveform_lengths: [B] - lengths of each waveform
            
        Returns:
            z: [B, D, T'] - latent representation
            z_mask: [B, T'] - mask for valid frames
        """
        # Ensure waveform is [B, C, T]
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        # Get condition (latent)
        z = self.cond_module(waveform)  # [B, D, T']

        # Compute mask based on downsampled lengths
        z_length = waveform_lengths // self.downsampling_ratio
        z_mask = create_mask_from_length(z_length)
        
        return z, z_mask
    
    def decode(self, latents: torch.Tensor, normalize: bool = True, boost_quiet: bool = False, vocoder_steps: int = None, **kwargs) -> torch.Tensor:
        """
        Decode latent representation to waveform using vocoder.
        
        Args:
            latents: [B, D, T'] - latent representation
            
        Returns:
            waveform: [B, 1, T] - reconstructed audio
        """
        if vocoder_steps is None:
            print(f"Using default vocoder_steps in SemanticVocoder: {self.n_timesteps}")
            vocoder_steps = self.n_timesteps
        with torch.inference_mode():
            waveform = self.generator.infer(
                cond=latents,
                n_timesteps=vocoder_steps,
                clamp_pred=self.clamp_pred
            )
        
        # Normalize to prevent clipping
        if normalize:
            waveform = waveform / waveform.abs().max(dim=-1, keepdim=True)[0].clamp(min=1.0)
        if boost_quiet:
            max_val = waveform.abs().max(dim=-1, keepdim=True)[0]
            target_level = 0.9  # Target peak level
            scale = target_level / max_val.clamp(min=1e-8)
            scale = scale.clamp(max=10.0)  # Limit max amplification to avoid noise boost
            waveform = waveform * scale
        waveform = waveform.unsqueeze(1)
        return waveform
