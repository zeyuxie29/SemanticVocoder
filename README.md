# :seedling: SemanticVocoder：Bridging Audio Generation and Audio Understanding via Semantic Latents
[![arXiv](https://img.shields.io/badge/arXiv-2602.23333-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2602.23333)
[![githubio](https://img.shields.io/badge/GitHub.io-Audio_Samples-blue?logo=Github&style=flat-square)](https://zeyuxie29.github.io/SemanticVocoder/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/ZeyuXie/SemanticVocoder)




<a id="introduction"></a>
## :collision: SemanticVocoder
We propose SemanticVocoder, which innovatively **generates waveforms** directly from semantic latents. 
The core advantages are:
* Enables the audio generation framework to operate in the semantic latent space, while **eliminating any reliance on VAE modules** and mitigating their adverse effects;
* Empowers our text-to-audio system to achieve strong performance on AudioCaps, with a Fréchet Distance of **12.823** and a Fréchet Audio Distance of **1.709**;
* Allows the two-stage pipeline (text-to-latent & latent-to-waveform) to be **independently trained** with semantic latents as the anchor, supporting **plug-and-play** deployment;
* Bridges semantic latents and generative tasks, enabling semantic latents to support **unified modeling** for both audio generation and audio understanding.


***

<a id="inference"></a>

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

***

## TODO List
- [x] Add demo page
- [x] Release text-to-audio generation inference code and usage instructions
- [ ] Release vocoder inference module (responsible for encoding latent representations and decoding)
- [ ] Release vocoder training code
- [ ] Release text-to-audio generation training code
