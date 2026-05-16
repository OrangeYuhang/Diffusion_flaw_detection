import os
import shutil
import torch
torch.cuda.is_available()
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image

# 导入 DefectDiffu 的模型和生成函数（复用 test_gen.py 中的逻辑）
from models_add_cross_concate import DiT
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
import clip.clip as clip

# ======================== 配置 ========================
ORIGINAL_MVTEC_PATH = "./data/mvtec"          # 原始 MVTec AD 数据集路径
TARGET_ROOT = "D:/desktop/anodetection/data/mvtec"         # 目标根目录（用于 PatchCore 训练）
CKPT_PATH = "model_para/model_400_plus.pth"        # 你训练好的 DiT 检查点
VAE_PATH = "VAE"                              # VAE 本地目录或 HuggingFace ID
IMAGE_SIZE = 256
BATCH_SIZE = 1                                # 生成时 batch size（显存小建议 =1）
# 每种缺陷的 CFG 范围（基于视觉强度分级）
DEFECT_CFG_RANGE = {
    # faint —— 轻微缺陷，低 CFG 避免过饱和
    "scratch":              (1.0, 1.8),
    "contamination":        (1.0, 1.5),
    "color":                (1.0, 1.5),
    "thread":               (1.0, 1.5),
    "glue":                 (1.0, 1.5),
    "rough":                (1.2, 2.0),
    "oil":                  (1.0, 1.5),
    "faulty_imprint":       (1.0, 1.5),
    "poke":                 (1.2, 2.0),
    # medium —— 中等缺陷
    "crack":                (1.5, 2.5),
    "cut":                  (1.5, 2.5),
    "fold":                 (1.5, 2.5),
    "print":                (1.5, 2.5),
    "manipulated_front":    (1.5, 2.5),
    "scratch_head":         (1.5, 2.5),
    "scratch_neck":         (1.5, 2.5),
    "thread_side":          (1.5, 2.5),
    "thread_top":           (1.5, 2.5),
    "fabric_border":        (1.5, 2.5),
    "fabric_interior":      (1.5, 2.5),
    "squeeze":              (1.5, 2.5),
    "combined":             (1.5, 2.5),
    "gray_stroke":          (1.5, 2.5),
    "metal_contamination":  (1.5, 2.5),
    "misplaced":            (1.5, 2.5),
    "flip":                 (1.5, 2.5),
    "squeezed_teeth":       (1.5, 2.5),
    "damaged_case":         (1.5, 2.5),
    "cable_swap":           (1.5, 2.5),
    "missing_cable":        (1.5, 2.5),
    "missing_wire":         (1.5, 2.5),
    "bent_wire":            (1.5, 2.5),
    "cut_inner_insulation": (1.5, 2.5),
    "cut_outer_insulation": (1.5, 2.5),
    "poke_insulation":      (1.5, 2.5),
    "bent":                 (1.5, 2.5),
    # heavy —— 严重结构性损伤
    "hole":                 (2.5, 3.5),
    "broken":               (2.5, 3.5),
    "broken_large":         (2.5, 3.5),
    "broken_small":         (2.0, 3.0),
    "missing":              (2.5, 3.5),
    "split":                (2.5, 3.5),
    "split_teeth":          (2.5, 3.5),
    "broken_teeth":         (2.5, 3.5),
    "bent_lead":            (2.0, 3.0),
    "cut_lead":             (2.0, 3.0),
    "defective":            (2.0, 3.0),
    "liquid":               (2.0, 3.0),
    "pill_type":            (2.0, 3.0),
}
DEFAULT_CFG_RANGE = (1.5, 2.5)               # 未列出缺陷类型的默认 CFG 范围
NUM_CFG_LEVELS = 4                            # 每种缺陷在 CFG 范围内均匀采样几档
NUM_PER_LEVEL = 3                             # 每档生成几张（不同随机种子）
# 总计每种缺陷 ≈ NUM_CFG_LEVELS × NUM_PER_LEVEL = 12 张

DEVICE = "cuda"
# =====================================================

def generate_defect_image(defect_type, product, ckpt_path, vae_path, cfg_scale, device):
    """生成单张缺陷图像和掩码，返回图像 Tensor (1,3,256,256)"""
    # 延迟导入，避免循环中重复加载模型（实际会在循环外加载，这里仅为示例）
    # 实际我们会把模型加载一次，然后在循环内调用
    pass

def load_models(ckpt_path, vae_path, device):
    """加载 DiT、VAE、CLIP 模型"""
    image_size = IMAGE_SIZE
    latent_size = image_size // 8

    model = DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16,
                input_size=latent_size, num_classes=1000).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model = model.float()   # 确保全精度

    diffusion = create_diffusion(timestep_respacing="50")
    vae = AutoencoderKL.from_pretrained(vae_path).to(device)
    model_clip, _ = clip.load('RN50', device)
    return model, diffusion, vae, model_clip

def generate_single(model, diffusion, vae, model_clip, defect_type, product, cfg_scale, device):
    """生成一张缺陷图像，返回 PIL 兼容的 Tensor (1,3,256,256)"""
    text = f"{defect_type} {product}"
    num_img = 1
    latent_size = IMAGE_SIZE // 8

    # 构造文本嵌入
    defect_text = f"a photo of {defect_type}"
    class_text = f"a photo of {product}"
    all_text = f"a photo of {defect_type} {product}"
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

    z = torch.randn(num_img, 4, latent_size, latent_size, device=device).float()
    z = torch.cat([z, z], 0)   # 用于 CFG
    y = [y_defect_class, y_good_class]
    model_kwargs = dict(y=y, cfg_scale=cfg_scale)

    with torch.no_grad():
        samples, _ = diffusion.p_sample_loop(
            model.forward_with_cfg_2,
            z.shape, z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=device
        )
    img_gen, _ = samples.chunk(2, dim=0)
    img_gen = vae.decode(img_gen / 0.18215).sample
    return img_gen   # 形状 (1,3,256,256)

def main():
    print("1. 复制原始正常样本到 train/good/ ...")
    src_good_root = os.path.join(ORIGINAL_MVTEC_PATH, "*", "train", "good")
    target_good_dir = os.path.join(TARGET_ROOT, "train", "good")
    os.makedirs(target_good_dir, exist_ok=True)
    # 遍历所有类别
    classes = [d for d in os.listdir(ORIGINAL_MVTEC_PATH) 
               if os.path.isdir(os.path.join(ORIGINAL_MVTEC_PATH, d))]
    for cls in classes:
        src_good = os.path.join(ORIGINAL_MVTEC_PATH, cls, "train", "good")
        if not os.path.isdir(src_good):
            continue
        for fname in os.listdir(src_good):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                src = os.path.join(src_good, fname)
                dst = os.path.join(target_good_dir, f"{cls}_{fname}")
                shutil.copy2(src, dst)
    print(f"已复制正常样本到 {target_good_dir}")

    print("2. 复制原始测试集到 test/ ...")
    target_test_dir = os.path.join(TARGET_ROOT, "test")
    if os.path.exists(target_test_dir):
        shutil.rmtree(target_test_dir)
    shutil.copytree(os.path.join(ORIGINAL_MVTEC_PATH), target_test_dir,
                    ignore=lambda src, names: [n for n in names if n not in ['test']])  # 只复制 test 目录
    # 实际上 copytree 不能直接按子目录筛选，简单做法：遍历每个 class 的 test
    # 更可靠：重新实现
    # 简单点：清空后手动复制
    shutil.rmtree(target_test_dir, ignore_errors=True)
    os.makedirs(target_test_dir)
    for cls in classes:
        src_test = os.path.join(ORIGINAL_MVTEC_PATH, cls, "test")
        if os.path.isdir(src_test):
            dst_cls = os.path.join(target_test_dir, cls)
            shutil.copytree(src_test, dst_cls)
    print("测试集复制完成。")

    print("3. 复制原始 ground_truth 到 ground_truth/ ...")
    target_gt_dir = os.path.join(TARGET_ROOT, "ground_truth")
    shutil.rmtree(target_gt_dir, ignore_errors=True)
    os.makedirs(target_gt_dir)
    for cls in classes:
        src_gt = os.path.join(ORIGINAL_MVTEC_PATH, cls, "ground_truth")
        if os.path.isdir(src_gt):
            dst_gt_cls = os.path.join(target_gt_dir, cls)
            shutil.copytree(src_gt, dst_gt_cls)
    print("ground_truth 复制完成。")

    print("4. 加载 DefectDiffu 模型...")
    model, diffusion, vae, model_clip = load_models(CKPT_PATH, VAE_PATH, DEVICE)
    print("模型加载完成。")

    print("5. 开始生成缺陷图像...")
    target_bad_dir = os.path.join(TARGET_ROOT, "train", "bad")
    os.makedirs(target_bad_dir, exist_ok=True)

    # 收集所有需要生成的 (defect, product) 对
    tasks = []
    for cls in classes:
        test_dir = os.path.join(ORIGINAL_MVTEC_PATH, cls, "test")
        if not os.path.isdir(test_dir):
            continue
        for defect in os.listdir(test_dir):
            if defect == "good":
                continue
            tasks.append((defect, cls))
    print(f"共发现 {len(tasks)} 种缺陷类型。")

    total_generated = 0
    for defect, product in tqdm(tasks, desc="生成缺陷图像"):
        cfg_lo, cfg_hi = DEFECT_CFG_RANGE.get(defect, DEFAULT_CFG_RANGE)
        cfg_values = np.linspace(cfg_lo, cfg_hi, NUM_CFG_LEVELS)
        for i, cfg in enumerate(cfg_values):
            for seed in range(NUM_PER_LEVEL):
                total_generated += 1
                # 固定种子保证可复现  seed = i * 100 + seed
                torch.manual_seed(i * 100 + seed)
                try:
                    img_tensor = generate_single(model, diffusion, vae, model_clip,
                                                 defect, product, float(cfg), DEVICE)
                    save_name = (f"{product}_{defect}_"
                                 f"cfg{cfg:.1f}_{seed:02d}.png")
                    save_path = os.path.join(target_bad_dir, save_name)
                    save_image(img_tensor, save_path, normalize=True)
                except Exception as e:
                    print(f"  ✗ {product}/{defect} cfg={cfg:.1f}/{seed}: {e}")
    print(f"共生成 {total_generated} 张缺陷图像")

    print(f"所有缺陷图像已保存到 {target_bad_dir}")
    print("数据集准备完成！")

if __name__ == "__main__":
    main()