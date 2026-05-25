"""
mask_cond 可控性验证脚本。

测试 DiTMaskConditioned 是否真正响应 mask_cond：
对同一缺陷/产品、同一随机种子，使用不同位置的 mask_cond 生成图像，
比较缺陷位置是否跟随 mask_cond 移动。

用法:
    python test_mask_cond_control.py \
        --ckpt ./model_para/model_mask_cond_s1_100.pth \
        --vae ./VAE --data ./data/mvtec

输出:
    img/test_mask_cond/ 下每个测试用例的对比图
    控制台输出各位置的定量指标
"""
import os
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision.utils import save_image

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from models_mask_condition import DiTMaskConditioned
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
import clip.clip as clip

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LATENT_SIZE = 32


def make_position_mask(position, latent_size=LATENT_SIZE, coverage=0.25):
    """
    在潜空间中创建指定位置的二值掩码。

    position: "top_left", "top_right", "bottom_left", "bottom_right", "center"
    coverage: 掩码面积占比 (边长 ≈ sqrt(coverage))
    """
    side = int(np.sqrt(coverage) * latent_size)
    h_min, w_min = 0, 0

    if position == "top_left":
        h_min, w_min = 1, 1
    elif position == "top_right":
        h_min, w_min = 1, latent_size - side - 1
    elif position == "bottom_left":
        h_min, w_min = latent_size - side - 1, 1
    elif position == "bottom_right":
        h_min, w_min = latent_size - side - 1, latent_size - side - 1
    elif position == "center":
        h_min = w_min = (latent_size - side) // 2
    elif position == "none":
        return torch.zeros(1, 1, latent_size, latent_size)

    mask = torch.zeros(latent_size, latent_size)
    mask[h_min:h_min + side, w_min:w_min + side] = 1.0
    return mask.unsqueeze(0).unsqueeze(0)


def binarize_mask(mask_tensor):
    """自适应阈值二值化掩码张量 (C, H, W) -> (H, W)"""
    mask = mask_tensor.cpu()
    gray = 0.299 * mask[0] + 0.587 * mask[1] + 0.114 * mask[2]
    gray_np = gray.numpy()
    threshold = gray_np.mean() + 0.5 * gray_np.std()
    return (gray_np > threshold).astype(np.uint8)


def compute_mask_iou(pred_bin, target_bin):
    """计算两个二值掩码之间的 IoU"""
    intersection = (pred_bin & target_bin).sum()
    union = (pred_bin | target_bin).sum()
    if union == 0:
        return 0.0
    return intersection / union


def get_label_list(data_path):
    """收集所有缺陷类型+产品组合"""
    labels = []
    for name_class in os.listdir(data_path):
        test_path = os.path.join(data_path, name_class, 'test')
        if not os.path.isdir(test_path):
            continue
        for defect_type in os.listdir(test_path):
            if defect_type != 'good':
                labels.append((defect_type, name_class))
    return labels


@torch.no_grad()
def run_test(args):
    print(f"[Device] {DEVICE}")
    os.makedirs("img/test_mask_cond", exist_ok=True)

    # ---- 加载模型 ----
    model_clip, _ = clip.load('RN50', DEVICE)

    model = DiTMaskConditioned(
        depth=28, hidden_size=1152, patch_size=2, num_heads=16,
        input_size=LATENT_SIZE, num_classes=1000).to(DEVICE)

    state_dict = torch.load(args.ckpt, map_location=DEVICE)
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[Model] Loaded {len(state_dict)} keys")
    print(f"[Model] Missing (will use init): {len(missing)}")
    model.eval()

    diffusion = create_diffusion(timestep_respacing="50")
    vae = AutoencoderKL.from_pretrained(args.vae).to(DEVICE)

    # ---- 测试位置 ----
    positions = ["none", "top_left", "bottom_right"]
    pos_masks = {p: make_position_mask(p).to(DEVICE) for p in positions}

    # ---- 选择测试用例 ----
    all_labels = get_label_list(args.data)
    # 每类产品选 1 个代表（优先 scratch, hole, cut）
    test_cases = []
    seen_products = set()
    priority = ['scratch', 'hole', 'cut', 'dent', 'broken', 'crack']
    for defect, product in all_labels:
        if product not in seen_products:
            test_cases.append((defect, product))
            seen_products.add(product)
    # 按产品字母排序保证可复现
    test_cases.sort(key=lambda x: x[1])
    print(f"[Test] {len(test_cases)} cases: {[p for _, p in test_cases]}")

    results_summary = []

    for defect_text, product_text in test_cases:
        print(f"\n{'='*60}")
        print(f"  {defect_text} {product_text}")
        print(f"{'='*60}")

        # ---- 文本编码 ----
        prompt_defect = f"a photo of {defect_text}"
        prompt_class = f"a photo of {product_text}"
        prompt_all = f"a photo of {defect_text} {product_text}"

        tok_defect = clip.tokenize([prompt_defect]).to(DEVICE)
        tok_class = clip.tokenize([prompt_class]).to(DEVICE)
        tok_all = clip.tokenize([prompt_all]).to(DEVICE)
        tok_good = clip.tokenize(["a photo of good"]).to(DEVICE)
        tok_good_class = clip.tokenize([f"a photo of good {product_text}"]).to(DEVICE)

        emb_defect = model_clip.encode_text(tok_defect).float()
        emb_class = model_clip.encode_text(tok_class).float()
        emb_all = model_clip.encode_text(tok_all).float()
        emb_good = model_clip.encode_text(tok_good).float()
        emb_good_class = model_clip.encode_text(tok_good_class).float()

        for emb in [emb_defect, emb_class, emb_all, emb_good, emb_good_class]:
            emb /= emb.norm(dim=-1, keepdim=True)

        y = [
            [emb_defect, emb_class, emb_all],
            [emb_good, emb_class, emb_good_class],
        ]

        generated = {}

        for pos_name in positions:
            mask_cond = pos_masks[pos_name]

            z = torch.randn(1, 4, LATENT_SIZE, LATENT_SIZE, device=DEVICE)
            z_cfg = torch.cat([z, z], dim=0)

            samples, cross, _ = model.forward_with_cfg_mask(
                z_cfg,
                torch.zeros(z_cfg.shape[0], device=DEVICE, dtype=torch.long),
                y, args.cfg_scale, mask_cond)

            img_gen, _ = samples.chunk(2, dim=0)
            mask_gen, _ = cross.chunk(2, dim=0)

            # 模型输出 8 通道 (4 mean + 4 var)，VAE 需要 4 通道
            img_gen_4ch = img_gen[:, :4, :, :]

            # adapt_mask 输出是 6D: (B, 4, 16, 2, 16, 2) → (B, 4, 32, 32)
            if mask_gen.ndim == 6:
                B = mask_gen.shape[0]
                mask_gen_4ch = mask_gen.permute(0, 1, 2, 4, 3, 5).reshape(B, 4, 32, 32)
            elif mask_gen.ndim == 4:
                mask_gen_4ch = mask_gen[:, :4, :, :] if mask_gen.shape[1] > 4 else mask_gen
            else:
                mask_gen_4ch = mask_gen

            img = vae.decode(img_gen_4ch / 0.18215).sample
            mask = vae.decode(mask_gen_4ch / 0.18215).sample

            generated[pos_name] = {
                'img': img[0],
                'mask': mask[0],
            }

        # ---- 定量分析 ----
        # 1) 各位置下生成掩码与对应 mask_cond 的 IoU
        print(f"  {'Position':<16} {'Mask Mean':>10} {'IoU vs mask_cond':>15}")
        row_metrics = {'defect': defect_text, 'product': product_text}
        for pos_name in positions:
            mask_out = generated[pos_name]['mask']
            mask_bin = binarize_mask(mask_out)
            mask_cond_np = pos_masks[pos_name][0, 0].cpu().numpy()
            # 将 32x32 的 mask_cond 上采样到 256x256 作比较
            mask_cond_up = np.array(
                Image.fromarray((mask_cond_np * 255).astype(np.uint8)).resize(
                    (256, 256), Image.NEAREST)) > 128
            iou = compute_mask_iou(mask_bin, mask_cond_up)
            mask_mean = mask_out.mean().item()
            print(f"  {pos_name:<16} {mask_mean:>10.6f} {iou:>15.4f}")
            row_metrics[f'iou_{pos_name}'] = iou
            row_metrics[f'mask_mean_{pos_name}'] = mask_mean

        # 2) 位置间差异：top_left vs bottom_right 掩码是否不同
        mask_bin_tl = binarize_mask(generated['top_left']['mask'])
        mask_bin_br = binarize_mask(generated['bottom_right']['mask'])
        diff_ratio = (mask_bin_tl != mask_bin_br).mean()
        print(f"  TL vs BR mask difference ratio: {diff_ratio:.4f} "
              f"{'(different=GOOD)' if diff_ratio > 0.05 else '(similar=BAD)'}")
        row_metrics['tl_br_diff'] = diff_ratio

        # 3) 图像差异：top_left vs bottom_right 像素级差别
        img_tl = generated['top_left']['img']
        img_br = generated['bottom_right']['img']
        img_diff = (img_tl - img_br).abs().mean().item()
        print(f"  TL vs BR image pixel diff:   {img_diff:.6f}")
        row_metrics['img_diff_tl_br'] = img_diff

        results_summary.append(row_metrics)

        # ---- 保存对比图 ----
        # 布局: 行=位置, 列=[图像, 掩码, mask_cond]
        rows_to_save = []
        for pos_name in positions:
            img = generated[pos_name]['img']
            mask = generated[pos_name]['mask']
            mask_c = pos_masks[pos_name][0].cpu()  # (1, 32, 32)
            # 上采样 mask_cond 到 256 以便并排
            mask_c_up = torch.from_numpy(
                np.array(Image.fromarray(
                    (mask_c[0].numpy() * 255).astype(np.uint8)).resize(
                        (256, 256), Image.NEAREST))).float() / 255.0
            mask_c_up = mask_c_up.unsqueeze(0).repeat(3, 1, 1).to(DEVICE)  # 3ch for save
            rows_to_save.extend([img, mask, mask_c_up])

        grid = torch.stack(rows_to_save, dim=0)  # (9, 3, 256, 256)
        save_path = f"img/test_mask_cond/{product_text}_{defect_text}.png"
        save_image(grid, save_path, nrow=3, normalize=True, value_range=(-1, 1))
        print(f"  -> saved {save_path}")

    # ---- 汇总 ----
    print(f"\n{'='*60}")
    print("  汇总")
    print(f"{'='*60}")
    print(f"  {'Product':<14} {'Defect':<12} {'IoU TL':>8} {'IoU BR':>8} "
          f"{'TL-BR diff':>11} {'Img diff':>10}")
    for r in results_summary:
        print(f"  {r['product']:<14} {r['defect']:<12} "
              f"{r['iou_top_left']:>8.4f} {r['iou_bottom_right']:>8.4f} "
              f"{r['tl_br_diff']:>11.4f} {r['img_diff_tl_br']:>10.6f}")

    # 汇总判断
    avg_tl_br_diff = np.mean([r['tl_br_diff'] for r in results_summary])
    avg_img_diff = np.mean([r['img_diff_tl_br'] for r in results_summary])
    print(f"\n  Avg TL vs BR mask diff: {avg_tl_br_diff:.4f}")
    print(f"  Avg TL vs BR image diff: {avg_img_diff:.6f}")

    if avg_tl_br_diff < 0.02 and avg_img_diff < 0.005:
        print("\n  *** 结论: mask_cond 基本无效 ***")
        print("  不同位置的 mask_cond 产生的图像和掩码几乎相同，")
        print("  说明模型没有学会根据 mask_cond 控制缺陷位置。")
        print("  建议: 添加直接的 mask_cond 空间一致性损失。")
    elif avg_tl_br_diff < 0.10:
        print("\n  *** 结论: mask_cond 效果微弱 ***")
        print("  不同位置的 mask_cond 有微小差异，但不足以可靠控制缺陷位置。")
        print("  建议: 增强空间注入权重或添加位置一致性损失。")
    else:
        print("\n  *** 结论: mask_cond 有效 ***")
        print("  不同位置的 mask_cond 产生了明显不同的缺陷位置。")

    print("\n[完成] 对比图保存在 img/test_mask_cond/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="mask_cond 可控性验证测试")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="DiTMaskConditioned 检查点路径")
    parser.add_argument("--vae", type=str, required=True,
                        help="VAE 检查点路径")
    parser.add_argument("--data", type=str, required=True,
                        help="MVTec AD 数据集根路径")
    parser.add_argument("--cfg_scale", type=float, default=2.0,
                        help="CFG 引导强度 (default: 2.0)")
    args = parser.parse_args()
    run_test(args)
