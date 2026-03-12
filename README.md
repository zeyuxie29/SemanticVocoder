# :seedling: SemanticVocoder：Bridging Audio Generation and Audio Understanding via Semantic Latents
[![arXiv](https://img.shields.io/badge/arXiv-2602.23333-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2602.23333)
[![githubio](https://img.shields.io/badge/GitHub.io-Audio_Samples-blue?logo=Github&style=flat-square)](https://zeyuxie29.github.io/SemanticVocoder/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/ZeyuXie/SemanticVocoder)


### Table of Contents

 - [Introduction](#introduction)
 - [Text-To-Audio Generation](#tta_inference)
 - [SemanticVocoder Inference Encode&Decode](#vocoder_inference)
 

<a id="introduction"></a>
## :collision: SemanticVocoder
We propose SemanticVocoder, which innovatively **generates waveforms** directly from semantic latents. 
The core advantages are:
* Enables the audio generation framework to operate in the semantic latent space, while **eliminating any reliance on VAE modules** and mitigating their adverse effects;
* Empowers our text-to-audio system to achieve strong performance on AudioCaps, with a Fréchet Distance of **12.823** and a Fréchet Audio Distance of **1.709**;
* Allows the two-stage pipeline (text-to-latent & latent-to-waveform) to be **independently trained** with semantic latents as the anchor, supporting **plug-and-play** deployment;
* Bridges semantic latents and generative tasks, enabling semantic latents to support **unified modeling** for both audio generation and audio understanding.


***

<a id="tta_inference"></a>

## Text-to-Audio Generation Inference
#### Clone the repository:
> [!IMPORTANT]
> Use *--single-branch --branch main* or *--depth=1* to avoid downloading oversized files.
```shell
git clone --single-branch --branch main https://github.com/zeyuxie29/SemanticVocoder
or
git clone --depth=1 https://github.com/zeyuxie29/SemanticVocoder
```
#### Install dependencies:
> [!NOTE]
> you may need to adjust the PyTorch version in *build_env.sh* to match your hardware
```shell
cd SemanticVocoder/src/inference
sh bash_scripts/build_env.sh
```
#### Run inference:
```shell
sh bash_scripts/infer.sh
```
<a id="vocoder_inference"></a>

## SemanticVocoder Inference 
Set up the environment as noted above, then execute **encoding** and **decoding**. 
```python

import os
import json

import hydra
import torch
import torchaudio
import soundfile as sf
from tqdm import tqdm 
from huggingface_hub import snapshot_download

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATH = snapshot_download(
    repo_id="ZeyuXie/SemanticVocoderSnapshot",
    allow_patterns=[
        "semanticVocoder_epoch-270.pt",
        "config.json",
    ],  
    cache_dir=None,
)

model_config = json.load(open(f"{PATH}/config.json", "r"))
autoencoder = hydra.utils.instantiate(model_config["model"]["autoencoder"])
sample_rate = autoencoder.sample_rate

model_path = f"{PATH}/semanticVocoder_epoch-270.pt"
autoencoder._load_checkpoint(model_path)
autoencoder.eval()
autoencoder.to(device)

test_audio = "data/audiocaps/test/Y_C2HinL8VlM.wav"
test_output = "test_output/Y_C2HinL8VlM.wav"

waveform, sr = torchaudio.load(test_audio)
# Resample if needed
if sr != sample_rate :
    print(f"Resampling from {sr} Hz to {autoencoder.sample_rate} Hz")
    waveform = torchaudio.functional.resample(
        waveform, orig_freq=sr, new_freq=sample_rate 
    )

waveform = waveform.to(device)

# Encode
waveform_lengths = torch.tensor([waveform.shape[-1]], device=device)
z, z_mask = autoencoder.encode(waveform, waveform_lengths)
print(f"Latent shape: {z.shape}")

# Decode
recon = autoencoder.decode(z, vocoder_steps=200)
print(f"Reconstructed waveform shape: {recon.shape}")

output_path = f"{test_output}"
sf.write(output_path, recon.squeeze().squeeze().cpu().numpy(), sample_rate)
print(f"Saved reconstructed audio to {test_output}")
```

***

## TODO List
- [x] Add demo page
- [x] Release text-to-audio generation inference code and usage instructions
- [x] Release vocoder inference module (responsible for encoding latent representations and decoding)
- [ ] Release vocoder training code
- [ ] Release text-to-audio generation training code
