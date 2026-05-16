import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
import numpy as np
from PIL import Image
from torchvision.utils import save_image

# 导入项目中的模块（确保路径正确）
from models_add_cross_concate import DiT
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
import clip.clip as clip

def generate_from_text(
    ckpt_path="checkpoint/model_600.pth",   # 你的微调检查点
    vae_path="VAE",   # 或者本地路径
    text="scratch bottle",                  # 文本提示，格式 "缺陷 产品"
    cfg_scale=2.0,                          # 引导强度，越大缺陷越明显
    image_size=256,
    batch_size=1,
    device="cuda"
):
    # 1. 解析文本
    parts = text.split()
    if len(parts) != 2:
        raise ValueError("文本格式应为 '缺陷 产品'，例如 'scratch bottle'")
    defect_type, product = parts[0], parts[1]

    # 2. 初始化模型
    latent_size = image_size // 8
    model = DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16,
                input_size=latent_size, num_classes=1000).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    model =model.float()
    model.eval()

    diffusion = create_diffusion(timestep_respacing="50")
    vae = AutoencoderKL.from_pretrained(vae_path).to(device)
    model_clip, _ = clip.load('RN50', device)

    # 3. 构造文本嵌入
    num_img = batch_size
    # 缺陷侧条件
    defect_text = f"a photo of {defect_type}"
    class_text = f"a photo of {product}"
    all_text = f"a photo of {defect_type} {product}"
    # 正常侧条件
    good_text = "a photo of good"
    good_class_text = f"a photo of good {product}"
    good_industry_text = "a photo of good industry"

    with torch.no_grad():
        y_defect = model_clip.encode_text(clip.tokenize([defect_text] * num_img).to(device)).float()
        y_class = model_clip.encode_text(clip.tokenize([class_text] * num_img).to(device)).float()
        y_all = model_clip.encode_text(clip.tokenize([all_text] * num_img).to(device)).float()
        y_good = model_clip.encode_text(clip.tokenize([good_text] * num_img).to(device)).float()
        y_good_class = model_clip.encode_text(clip.tokenize([good_class_text] * num_img).to(device)).float()
        y_good_industry = model_clip.encode_text(clip.tokenize([good_industry_text] * num_img).to(device)).float()

    # 归一化
    for emb in [y_defect, y_class, y_all, y_good, y_good_class, y_good_industry]:
        emb /= emb.norm(dim=-1, keepdim=True)

    y_defect_class = [y_defect, y_class, y_all]
    y_good_class = [y_good, y_class, y_good_class]

    # 4. 随机噪声
    z = torch.randn(num_img, 4, latent_size, latent_size, device=device).float()
    z = torch.cat([z, z], 0)     # 一份用于条件，一份用于无条件，用于CFG
    y = [y_defect_class, y_good_class]
    model_kwargs = dict(y=y, cfg_scale=cfg_scale)

    # 5. 采样生成
    with torch.no_grad():
        samples, _ = diffusion.p_sample_loop(
            model.forward_with_cfg_2,
            z.shape, z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
            device=device
        )
    img_gen, _ = samples.chunk(2, dim=0)
    img_gen = vae.decode(img_gen / 0.18215).sample

    # 6. 保存图像
    save_image(img_gen, f"{product}_{defect_type}_cfg{cfg_scale}.png", normalize=True)
    print(f"生成图像已保存为 {product}_{defect_type}_cfg{cfg_scale}.png")

if __name__ == "__main__":
    # 示例：生成瓶子上的划痕，引导强度2.0
    generate_from_text(
        ckpt_path="checkpoint/model_600.pth",      # 改成你的检查点路径
        vae_path="VAE",      # 或本地VAE目录
        text="scratch bottle",
        cfg_scale=2.0
    )