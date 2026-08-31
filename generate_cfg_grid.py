"""
生成 CFG Scale 渐变对比网格图（4列×2行）。
同一 seed + 同一 prompt，CFG 从 1.0 到 7.0 的渐变过程。

用法:
  python generate_cfg_grid.py \
    --ckpt checkpoint/model_600.pth \
    --vae ./VAE \
    --defect scratch \
    --product bottle \
    --seed 42 \
    --out ./img/cfg_grid.png
"""
import torch
import numpy as np
import argparse
import os
from torchvision.utils import save_image
from PIL import Image, ImageDraw, ImageFont

from models_add_cross_concate import DiT
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
import clip.clip as clip


def load_models(ckpt_path, vae_path, device, image_size=256):
    latent_size = image_size // 8
    model = DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16,
                input_size=latent_size, num_classes=1000).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    if isinstance(state_dict, dict):
        # 兼容旧式 {'model_state_dict': {...}, ...} 格式和纯 state_dict 格式
        if 'model_state_dict' in state_dict or 'pos_embed' not in state_dict:
            state_dict = state_dict.get('model_state_dict', state_dict)
    model.load_state_dict(state_dict)
    model.eval()
    model = model.float()

    diffusion = create_diffusion(timestep_respacing="50")
    vae = AutoencoderKL.from_pretrained(vae_path).to(device)
    model_clip, _ = clip.load('RN50', device)
    return model, diffusion, vae, model_clip


def encode_text_prompts(model_clip, defect_type, product, num_img, device):
    defect_text = f"a photo of {defect_type}"
    class_text = f"a photo of {product}"
    all_text = f"a photo of {defect_type} {product}"
    good_text = "a photo of good"
    good_class_text = f"a photo of good {product}"

    with torch.no_grad():
        y_defect = model_clip.encode_text(clip.tokenize([defect_text] * num_img).to(device)).float()
        y_class = model_clip.encode_text(clip.tokenize([class_text] * num_img).to(device)).float()
        y_all = model_clip.encode_text(clip.tokenize([all_text] * num_img).to(device)).float()
        y_good = model_clip.encode_text(clip.tokenize([good_text] * num_img).to(device)).float()
        y_good_cls = model_clip.encode_text(clip.tokenize([good_class_text] * num_img).to(device)).float()

    for emb in [y_defect, y_class, y_all, y_good, y_good_cls]:
        emb /= emb.norm(dim=-1, keepdim=True)

    y_defect_class = [y_defect, y_class, y_all]
    y_good_class = [y_good, y_class, y_good_cls]
    return y_defect_class, y_good_class


@torch.no_grad()
def generate_one(model, diffusion, vae, y_defect_class, y_good_class,
                 latent_size, cfg_scale, device):
    z = torch.randn(1, 4, latent_size, latent_size, device=device).float()
    z = torch.cat([z, z], 0)
    y = [y_defect_class, y_good_class]
    model_kwargs = dict(y=y, cfg_scale=cfg_scale)

    samples, cross = diffusion.p_sample_loop(
        model.forward_with_cfg_2, z.shape, z,
        clip_denoised=False, model_kwargs=model_kwargs,
        progress=False, device=device
    )
    img_gen, _ = samples.chunk(2, dim=0)
    mask_gen, _ = cross.chunk(2, dim=0)

    img = vae.decode(img_gen / 0.18215).sample
    mask = vae.decode(mask_gen / 0.18215).sample
    return img, mask


def tensor_to_pil(tensor):
    """Convert a (1, 3, H, W) tensor in [-1, 1] to a PIL Image."""
    img = tensor.squeeze(0).cpu()
    img = (img + 1) / 2  # [-1,1] -> [0,1]
    img = img.clamp(0, 1)
    img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(img)


def make_grid_image(images, labels, cols=4, image_size=256, label_height=40, spacing=8):
    """
    Arrange PIL images into a grid with labels.

    images: list of PIL Image
    labels: list of str (same length)
    cols: number of columns
    """
    rows = (len(images) + cols - 1) // cols
    cell_w = image_size + spacing * 2
    cell_h = image_size + label_height + spacing * 2

    grid_w = cols * cell_w + spacing
    grid_h = rows * cell_h + spacing

    canvas = Image.new("RGB", (grid_w, grid_h), color=(15, 17, 33))  # dark bg

    try:
        font = ImageFont.truetype("arial.ttf", 16)
        font_small = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    draw = ImageDraw.Draw(canvas)

    for idx, (img, label) in enumerate(zip(images, labels)):
        row = idx // cols
        col = idx % cols

        x = spacing + col * cell_w
        y = spacing + row * cell_h

        # Paste image
        img_resized = img.resize((image_size, image_size), Image.LANCZOS)
        canvas.paste(img_resized, (x + spacing, y + spacing))

        # Draw label
        draw.text((x + spacing, y + spacing + image_size + 6), label,
                  fill=(148, 163, 184), font=font)

    # Title at top
    draw.text((spacing, 2), "CFG Scale 渐变对比 — " + labels[0].split("cfg")[0].strip(),
              fill=(241, 245, 249), font=font)

    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--vae", type=str, required=True, help="Path to VAE directory")
    parser.add_argument("--defect", type=str, required=True, help="Defect type, e.g. scratch")
    parser.add_argument("--product", type=str, required=True, help="Product class, e.g. bottle")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="./img/cfg_grid.png", help="Output grid image path")
    parser.add_argument("--out_dir", type=str, default="./img/cfg_sweep",
                        help="Directory for individual images")
    parser.add_argument("--imagesize", type=int, default=256)
    parser.add_argument("--cfg_list", type=str, default="1.0,2.0,3.0,4.0,5.0,6.0,7.0",
                        help="Comma-separated CFG values")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    latent_size = args.imagesize // 8
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cfg_values = [float(x.strip()) for x in args.cfg_list.split(",")]

    print(f"Loading models...")
    model, diffusion, vae, model_clip = load_models(args.ckpt, args.vae, device, args.imagesize)
    y_defect_class, y_good_class = encode_text_prompts(
        model_clip, args.defect, args.product, 1, device)

    images = []
    labels = []
    print(f"Generating CFG sweep for {args.defect} {args.product} (seed={args.seed})")

    for cfg in cfg_values:
        # Reset seed for identical noise each time
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        print(f"  CFG = {cfg:.1f} ...", end=" ", flush=True)
        img, mask = generate_one(model, diffusion, vae,
                                 y_defect_class, y_good_class,
                                 latent_size, float(cfg), device)
        print("done")

        # Save individual image
        img_pil = tensor_to_pil(img)
        mask_pil = tensor_to_pil(mask)
        images.append(img_pil)
        labels.append(f"CFG = {cfg:.1f}")

        fname = f"{args.product}_{args.defect}_cfg{cfg:.1f}"
        save_image(img, os.path.join(args.out_dir, f"{fname}.png"), normalize=True)
        save_image(mask, os.path.join(args.out_dir, f"{fname}_mask.png"), normalize=True)

    # Create and save grid image
    print(f"Creating grid image...")
    grid = make_grid_image(images, labels, cols=4, image_size=args.imagesize)
    grid.save(args.out)
    print(f"Grid saved to: {args.out}")

    # Also create a horizontal strip version (1 row × 7 cols) for widescreen slides
    strip_w = args.imagesize * len(cfg_values) + 16 * (len(cfg_values) + 1)
    strip_h = args.imagesize + 60
    strip = Image.new("RGB", (strip_w, strip_h), color=(15, 17, 33))
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(strip)

    for i, (img, label) in enumerate(zip(images, labels)):
        x = 16 + i * (args.imagesize + 16)
        img_r = img.resize((args.imagesize, args.imagesize), Image.LANCZOS)
        strip.paste(img_r, (x, 8))
        draw.text((x, args.imagesize + 12), label, fill=(148, 163, 184), font=font)

    strip_path = args.out.replace(".png", "_strip.png")
    strip.save(strip_path)
    print(f"Strip saved to: {strip_path}")

    print(f"\nDone! Generated {len(cfg_values)} images.")
    print(f"  Grid:  {args.out}")
    print(f"  Strip: {strip_path}")
    print(f"  Individual: {args.out_dir}/")


if __name__ == "__main__":
    main()
