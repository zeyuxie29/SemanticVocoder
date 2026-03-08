from pathlib import Path
import copy

import torch
import hydra
from omegaconf import OmegaConf
from transformers import PreTrainedModel, PretrainedConfig

class SemanticVocoderConfig(PretrainedConfig):
    """Configuration class for SemanticVocoder model."""
    model_type = "semantic_vocoder"
    
    def __init__(self, 
        model_config=None,
        **kwargs):
        super().__init__(**kwargs)
        self.model_config = model_config

class SemanticVocoder(PreTrainedModel):
    """HuggingFace compatible SemanticVocoder model."""
    config_class = SemanticVocoderConfig
    
    def __init__(self, config):
        super().__init__(config)

        self.model = hydra.utils.instantiate(config.model_config)
        
    def forward(self, 
        content,
        num_steps=100,
        guidance_scale=3.5,
        guidance_rescale=0.5,
        vocoder_steps=200,
        latent_shape=[768, 250],
        **kwargs):
        """Forward pass through the model."""

        waveform = self.model.inference(
            content=[content],
            condition=None,
            task=["text_to_audio"],
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            vocoder_steps=vocoder_steps,
            latent_shape=latent_shape,
            **kwargs,
        )
        return waveform[0][0].cpu().numpy()
