"""
带可控变化的多样化缺陷图像生成。
支持三种多样性轴：
  1. 噪声级别退火 — 缩放初始潜变量噪声
  2. CFG 强度扫描 — 变化无分类器引导强度
  3. 种子变化 — 显式随机种子控制
"""
import torch
import numpy as np
import argparse
import os
from torchvision.utils import save_image
from PIL import Image

from models_add_cross_concate import DiT
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
import clip.clip as clip


def load_models(ckpt_path, vae_path, device, image_size=256):
    latent_size = image_size // 8
    model = DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16,
                input_size=latent_size, num_classes=1000).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()
    model = model.float()

    diffusion = create_diffusion(timestep_respacing="50")
    vae = AutoencoderKL.from_pretrained(vae_path).to(device)
    model_clip, _ = clip.load('RN50', device)
    return model, diffusion, vae, model_clip


def encode_text_prompts(model_clip, defect_type, product, num_img, device):
    """为 (缺陷, 产品) 对构建并编码文本提示集。"""
    defect_text = f"a photo of {defect_type}"
    class_text = f"a photo of {product}"
    all_text = f"a photo of {defect_type} {product}"
    good_text = "a photo of good"
    good_class_text = f"a photo of good {product}"
    good_industry_text = "a photo of good industry"

    with torch.no_grad():
        y_defect = model_clip.encode_text(clip.tokenize([defect_text] * num_img).to(device)).float()
        y_class   = model_clip.encode_text(clip.tokenize([class_text] * num_img).to(device)).float()
        y_all     = model_clip.encode_text(clip.tokenize([all_text] * num_img).to(device)).float()
        y_good    = model_clip.encode_text(clip.tokenize([good_text] * num_img).to(device)).float()
        y_good_cls = model_clip.encode_text(clip.tokenize([good_class_text] * num_img).to(device)).float()
        y_good_ind = model_clip.encode_text(clip.tokenize([good_industry_text] * num_img).to(device)).float()

    for emb in [y_defect, y_class, y_all, y_good, y_good_cls, y_good_ind]:
        emb /= emb.norm(dim=-1, keepdim=True)

    y_defect_class = [y_defect, y_class, y_all]
    y_good_class   = [y_good, y_class, y_good_cls]
    return y_defect_class, y_good_class


@torch.no_grad()
def generate_one(model, diffusion, vae, y_defect_class, y_good_class,
                 latent_size, cfg_scale, noise_scale, device):
    """使用给定的 noise_scale 和 cfg_scale 生成单张图像和掩码。"""
    z = torch.randn(1, 4, latent_size, latent_size, device=device).float()
    z = z * noise_scale
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--vae", type=str, required=True)
    parser.add_argument("--defect", type=str, required=True, help="e.g. scratch")
    parser.add_argument("--product", type=str, required=True, help="e.g. bottle")
    parser.add_argument("--out_dir", type=str, default="./img_diverse")
    parser.add_argument("--num_per_axis", type=int, default=5,
                        help="Number of samples per diversity axis")
    parser.add_argument("--imagesize", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    latent_size = args.imagesize // 8
    os.makedirs(args.out_dir, exist_ok=True)

    # 设置全局种子以获得基础可重复性
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model, diffusion, vae, model_clip = load_models(args.ckpt, args.vae, device, args.imagesize)
    y_defect_class, y_good_class = encode_text_prompts(
        model_clip, args.defect, args.product, 1, device)

    # ============================================================
    # 轴1: CFG 强度扫描（noise 固定为 1.0）
    # ============================================================
    print(f"--- CFG scale sweep for {args.defect} {args.product} ---")
    for i, cfg in enumerate(np.linspace(0.5, 4.0, args.num_per_axis)):
        torch.manual_seed(args.seed)  # 相同的噪声基础
        img, mask = generate_one(model, diffusion, vae,
                                 y_defect_class, y_good_class,
                                 latent_size, float(cfg), 1.0, device)
        save_image(img, os.path.join(args.out_dir,
                     f"{args.product}_{args.defect}_cfg{cfg:.1f}.png"), normalize=True)
        save_image(mask, os.path.join(args.out_dir,
                     f"{args.product}_{args.defect}_cfg{cfg:.1f}_mask.png"), normalize=True)
        print(f"  cfg={cfg:.1f} saved")

    # ============================================================
    # 轴2: 噪声强度扫描（cfg 固定为 2.0）
    # ============================================================
    print(f"--- Noise scale sweep for {args.defect} {args.product} ---")
    for i, ns in enumerate(np.linspace(0.7, 1.8, args.num_per_axis)):
        torch.manual_seed(args.seed + 1000)  # 不同的基础种子以与 CFG 扫描解耦
        img, mask = generate_one(model, diffusion, vae,
                                 y_defect_class, y_good_class,
                                 latent_size, 2.0, float(ns), device)
        save_image(img, os.path.join(args.out_dir,
                     f"{args.product}_{args.defect}_ns{ns:.2f}.png"), normalize=True)
        save_image(mask, os.path.join(args.out_dir,
                     f"{args.product}_{args.defect}_ns{ns:.2f}_mask.png"), normalize=True)
        print(f"  noise_scale={ns:.2f} saved")

    # ============================================================
    # 轴3: 随机种子变化（cfg 固定为 2.0，noise 固定为 1.0）
    # ============================================================
    print(f"--- Seed sweep for {args.defect} {args.product} ---")
    for i in range(args.num_per_axis):
        seed_i = args.seed + 2000 + i * 7
        torch.manual_seed(seed_i)
        img, mask = generate_one(model, diffusion, vae,
                                 y_defect_class, y_good_class,
                                 latent_size, 2.0, 1.0, device)
        save_image(img, os.path.join(args.out_dir,
                     f"{args.product}_{args.defect}_seed{seed_i}.png"), normalize=True)
        save_image(mask, os.path.join(args.out_dir,
                     f"{args.product}_{args.defect}_seed{seed_i}_mask.png"), normalize=True)
        print(f"  seed={seed_i} saved")

    # ============================================================
    # 轴4: 交叉乘积网格（cfg × noise_scale）实现全覆盖
    # ============================================================
    print(f"--- Grid sweep for {args.defect} {args.product} ---")
    n_grid = min(args.num_per_axis, 4)  # 保持网格较小
    for i, cfg in enumerate(np.linspace(0.5, 3.5, n_grid)):
        for j, ns in enumerate(np.linspace(0.8, 1.6, n_grid)):
            seed_ij = args.seed + 3000 + i * 100 + j
            torch.manual_seed(seed_ij)
            img, mask = generate_one(model, diffusion, vae,
                                     y_defect_class, y_good_class,
                                     latent_size, float(cfg), float(ns), device)
            save_image(img, os.path.join(args.out_dir,
                         f"{args.product}_{args.defect}_grid_c{cfg:.1f}_n{ns:.2f}.png"),
                       normalize=True)
            save_image(mask, os.path.join(args.out_dir,
                         f"{args.product}_{args.defect}_grid_c{cfg:.1f}_n{ns:.2f}_mask.png"),
                       normalize=True)
    print(f"Grid sweep done.")

    print(f"\nAll diverse samples saved to {args.out_dir}/")
    print(f"Total images: {3 * args.num_per_axis + n_grid * n_grid}")


if __name__ == "__main__":
    main()
