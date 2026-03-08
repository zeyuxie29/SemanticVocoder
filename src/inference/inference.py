import os
import argparse
from pathlib import Path

import re
import json
import soundfile as sf
import torch
from tqdm import tqdm

from transformers import AutoModel

def device_setting(mode="medium"):
    torch.set_float32_matmul_precision(mode) 
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def sanitize_filename(name: str, max_len: int = 100) -> str:
    """
    Clean and truncate a string to make it a valid and safe filename.
    """
    name = re.sub(r'[\\/*?:"<>|]', '_', name)
    name = name.replace('/', '_')
    max_len = min(len(name), max_len)
    return name[:max_len]

def seed_setting(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Set random seed: \033[1;92m{seed}\033[0m")

def parse_args():
    parser = argparse.ArgumentParser(description="Audio generation inference")

    parser.add_argument("--model_name", type=str, default="ZeyuXie/SemanticVocoder", help="Huggingface model name / Local checkpoint directory")
    parser.add_argument("--infer_file_path", type=str, default="data/audiocaps/test/caption.jsonl", help="Path to inference data file")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--sample_rate", type=int, default=24000, help="Sample rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_steps", type=int, default=100, help="Number of steps for audioDiT")
    parser.add_argument("--vocoder_steps", type=int, default=200, help="Number of steps for semanticVocoder")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Guidance scale")
    parser.add_argument("--guidance_rescale", type=float, default=0.5, help="Guidance rescale")
    parser.add_argument("--latent_shape", type=int, nargs=2, default=[768, 250], help="Latent shape [dim, length]")
    
    
    return parser.parse_args()

def main():

    ### Parse arguments
    args = parse_args()
    
    ### Init device
    device_setting()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    ### Init model
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True,  weights_only=False)
    model = model.to(device)
    model.eval()
    
    # Init output dir
    audio_output_dir = Path(args.output_dir) / f"guidance_scale-{args.guidance_scale}_dit_steps-{args.num_steps}_vocoder_steps-{args.vocoder_steps}_latent_shape-{args.latent_shape[0]}-{args.latent_shape[1]}"
    audio_output_dir.mkdir(parents=True, exist_ok=True)
    
    seed_setting(args.seed)
    
    # Init data
    infer_file_path = args.infer_file_path
    with open(infer_file_path, "r") as f:
        infer_list = [json.loads(line) for line in f.readlines()]


    # Start inference
    with torch.no_grad():
        for idx, item in enumerate(tqdm(infer_list, ncols=100, colour="cyan")):
            audio_id = item["audio_id"]
            caption = item["caption"]

            waveform = model(
                content=caption,
                num_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                guidance_rescale=args.guidance_rescale,
                vocoder_steps=args.vocoder_steps,
                latent_shape=args.latent_shape
            )
            
            name = f"{audio_id[:-4]}_{caption.replace(' ', '_')}"
            safe_name = sanitize_filename(name)
            sf.write(
                audio_output_dir / f"{safe_name}.wav",
                waveform,
                samplerate=args.sample_rate,
            )


if __name__ == "__main__":
    main()
