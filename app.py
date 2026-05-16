"""
DefectDiffu Gradio Web UI — 支持掩码位置控制
  模式1: 标准 DiT（文本→图像+掩码）
  模式2: 掩码条件 DiT（文本+位置→图像+掩码）
"""
import re
import gradio as gr
import torch
torch.cuda.is_available()
import numpy as np
from PIL import Image
from torchvision.utils import save_image

from models_add_cross_concate import DiT
from models_mask_condition import DiTMaskConditioned, generate_random_mask
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
import clip.clip as clip

DEVICE = "cuda"
LATENT_SIZE = 32  # image_size // 8


# ============================================================
#  模型加载
# ============================================================

def load_standard_model(ckpt_path, vae_path):
    """加载标准 DiT 模型（无掩码条件）"""
    model = DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16,
                input_size=LATENT_SIZE, num_classes=1000).to(DEVICE)
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()
    diffusion = create_diffusion(timestep_respacing="50")
    vae = AutoencoderKL.from_pretrained(vae_path).to(DEVICE)
    model_clip, _ = clip.load('RN50', DEVICE)
    return model, diffusion, vae, model_clip


def load_mask_model(ckpt_path, vae_path):
    """加载掩码条件 DiTMaskConditioned 模型"""
    # 先创建标准 DiT 取参数
    model = DiTMaskConditioned(
        depth=28, hidden_size=1152, patch_size=2, num_heads=16,
        input_size=LATENT_SIZE, num_classes=1000).to(DEVICE)
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    # DiTMaskConditioned 比 DiT 多了 mask_encoder / mask_fusion / mask_to_tokens
    # 如果加载的是标准 DiT 的 ckpt，缺失的 key 用默认初始化
    model.load_state_dict(state_dict['model_state_dict'], strict=False)
    model.eval()
    diffusion = create_diffusion(timestep_respacing="50")
    vae = AutoencoderKL.from_pretrained(vae_path).to(DEVICE)
    model_clip, _ = clip.load('RN50', DEVICE)
    return model, diffusion, vae, model_clip


# ============================================================
#  文本编码
# ============================================================

@torch.no_grad()
def encode_prompts(model_clip, defect_text, product_text, num_img=1):
    """编码文本提示，返回 CFG 所需的 embedding 列表"""
    prompt_defect = f"a photo of {defect_text}"
    prompt_class = f"a photo of {product_text}"
    prompt_all = f"a photo of {defect_text} {product_text}"
    prompt_good = "a photo of good"
    prompt_good_class = f"a photo of good {product_text}"
    prompt_good_industry = "a photo of good industry"

    tokens = {
        'defect': prompt_defect,
        'class': prompt_class,
        'all': prompt_all,
        'good': prompt_good,
        'good_class': prompt_good_class,
        'good_industry': prompt_good_industry,
    }
    embeds = {}
    for key, text in tokens.items():
        tok = clip.tokenize([text] * num_img).to(DEVICE)
        e = model_clip.encode_text(tok)
        embeds[key] = e / e.norm(dim=-1, keepdim=True)
    return embeds


# ============================================================
#  自然语言 → CFG 强度语义映射
# ============================================================

# 强度关键词 → CFG 值映射（中英双语）
SEVERITY_KEYWORD_MAP = {
    # 轻微
    "轻微": 1.0, "轻度": 1.0, "细小": 0.8, "微弱": 0.7, "浅浅": 0.9,
    "不明显": 1.0, "隐约": 0.8,
    # 中等
    "中等": 2.0, "一般": 2.0, "普通": 2.0, "中度": 2.2, "常规": 2.0,
    # 严重
    "严重": 3.5, "重度": 3.5, "明显": 3.0, "强烈": 3.8, "重大": 3.8,
    "极度": 4.0, "极重": 4.0, "深度": 3.5,
    # English — slight
    "slight": 1.0, "mild": 1.0, "minor": 1.0, "light": 1.0,
    "tiny": 0.8, "subtle": 0.9, "faint": 0.7,
    # English — moderate
    "moderate": 2.0, "medium": 2.0, "normal": 2.0, "average": 2.0,
    # English — severe
    "severe": 3.5, "heavy": 3.5, "strong": 3.5, "serious": 3.8,
    "major": 3.8, "deep": 3.0, "large": 3.0, "big": 3.0, "huge": 3.8,
    "extreme": 4.0, "critical": 4.0,
}

# 用于 CLIP 语义相似度回退的强度锚点
SEVERITY_ANCHORS_CFG = [
    ("a photo of a slight defect", 1.0),
    ("a photo of a moderate defect", 2.0),
    ("a photo of a severe defect", 3.5),
]


def extract_severity_cfg(defect_text, model_clip=None):
    """
    从缺陷文本中提取强度 → CFG 映射值。
    1. 优先匹配中英关键词
    2. 回退：CLIP 语义相似度插值
    3. 最终回退：默认 2.0
    返回 (clean_defect_text, cfg_value, severity_label)
    """
    clean = defect_text.strip()
    cfg_value = 2.0
    severity_label = "中等 (CFG≈2.0)"
    matched_keyword = None

    # Step 1: 关键词遍历（词边界匹配，避免 "large" 误匹配 "broken_large"）
    text_lower = clean.lower()
    best_len = 0
    for keyword, cfg in SEVERITY_KEYWORD_MAP.items():
        # 用词边界确保只匹配独立单词，不匹配缺陷名内部的子串
        kw = keyword.replace(' ', r'\s+')  # 多词关键词中间允许多个空格
        pattern = rf'(?<![a-z]){kw}(?![a-z])'
        if re.search(pattern, text_lower):
            if len(keyword) > best_len:
                best_len = len(keyword)
                cfg_value = cfg
                matched_keyword = keyword

    if matched_keyword:
        clean = re.sub(rf'(?<![a-z]){matched_keyword}(?![a-z])', '', clean,
                       flags=re.IGNORECASE).strip()
        # 清理多余空格
        clean = re.sub(r'\s+', ' ', clean).strip()
        # 按 CFG 区间归类标签
        if cfg_value <= 1.2:
            severity_label = f"轻微 (CFG≈{cfg_value:.1f})"
        elif cfg_value <= 2.5:
            severity_label = f"中等 (CFG≈{cfg_value:.1f})"
        else:
            severity_label = f"严重 (CFG≈{cfg_value:.1f})"
        return clean, cfg_value, severity_label

    # Step 2: CLIP 回退（如果 model_clip 可用）
    if model_clip is not None:
        try:
            full_prompt = f"a photo of {clean}"
            tok = clip.tokenize([full_prompt] + [a[0] for a in SEVERITY_ANCHORS_CFG]).to(DEVICE)
            embeds = model_clip.encode_text(tok)
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            user_emb = embeds[0:1]
            anchor_embs = embeds[1:]
            sims = (user_emb @ anchor_embs.T).squeeze(0)
            weights = torch.softmax(sims * 4, dim=0)
            cfg_value = sum(w.item() * SEVERITY_ANCHORS_CFG[i][1]
                            for i, w in enumerate(weights))
            cfg_value = round(cfg_value, 1)
            severity_label = f"语义映射 (CFG≈{cfg_value:.1f})"
        except Exception:
            pass

    return clean, cfg_value, severity_label


# ============================================================
#  掩码预处理
# ============================================================

def preprocess_mask(mask_input, mask_mode):
    """
    将用户输入的掩码转为 (1, 1, 32, 32) 张量。
    mask_input: 来自 gr.Image 或 gr.ImageEditor 的 numpy 数组
    mask_mode: "none" | "blob" | "line" | "noise" | "draw" | "upload"
    """
    if mask_mode == "none":
        return None

    if mask_mode in ("blob", "line", "noise"):
        # 使用 models_mask_condition 中的随机掩码生成器
        return generate_random_mask(LATENT_SIZE, DEVICE, mode=mask_mode)

    # draw / upload: 用户提供了图像
    if mask_input is None:
        return None

    if isinstance(mask_input, np.ndarray):
        # gr.Image 返回 H×W 或 H×W×C 的 numpy
        if mask_input.ndim == 3:
            # 取平均变灰度，或取 R 通道
            mask_gray = mask_input[:, :, 0] if mask_input.shape[2] >= 1 else mask_input.mean(axis=2)
        else:
            mask_gray = mask_input

        # 归一化到 [0, 1]
        mask_gray = mask_gray.astype(np.float32) / 255.0
        # 二值化
        mask_bin = (mask_gray > 0.3).astype(np.float32)

        # Resize 到 32×32（潜空间尺寸）
        mask_pil = Image.fromarray((mask_bin * 255).astype(np.uint8))
        mask_pil = mask_pil.resize((LATENT_SIZE, LATENT_SIZE), Image.BILINEAR)
        mask_tensor = torch.from_numpy(np.array(mask_pil)).float() / 255.0
        # 转为 (1, 1, 32, 32)
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).to(DEVICE)
        return mask_tensor

    return None


# ============================================================
#  生成逻辑
# ============================================================

@torch.no_grad()
def generate(defect_text, product_text, cfg_scale,
             mask_mode, mask_input,
             model, diffusion, vae, model_clip, use_mask_model):
    """统一生成函数，根据 use_mask_model 选择生成路径"""
    num_img = 1
    emb = encode_prompts(model_clip, defect_text, product_text, num_img)

    y_defect_class = [emb['defect'], emb['class'], emb['all']]
    y_good_class   = [emb['good'], emb['class'], emb['good_class']]

    z = torch.randn(num_img, 4, LATENT_SIZE, LATENT_SIZE, device=DEVICE)
    z = torch.cat([z, z], 0)
    y = [y_defect_class, y_good_class]

    if use_mask_model:
        # 掩码条件路径
        mask_cond = preprocess_mask(mask_input, mask_mode)
        model_kwargs = dict(y=y, cfg_scale=cfg_scale, mask_cond=mask_cond)
        samples, cross, _ = model.forward_with_cfg_mask(
            z, torch.zeros(z.shape[0], device=DEVICE, dtype=torch.long),
            y, cfg_scale, mask_cond)
    else:
        # 标准 CFG 路径
        model_kwargs = dict(y=y, cfg_scale=cfg_scale)
        samples, cross = diffusion.p_sample_loop(
            model.forward_with_cfg_2, z.shape, z,
            clip_denoised=False, model_kwargs=model_kwargs,
            progress=False, device=DEVICE)

    img_gen, _ = samples.chunk(2, dim=0)
    mask_gen, _ = cross.chunk(2, dim=0)

    img_gen = vae.decode(img_gen / 0.18215).sample
    mask_gen = vae.decode(mask_gen / 0.18215).sample

    # 图像后处理
    img = (img_gen[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
    img = (img.clip(0, 1) * 255).astype(np.uint8)

    # 掩码后处理（自适应阈值二值化）
    mask = mask_gen[0].cpu()
    mask_gray = 0.299 * mask[0] + 0.587 * mask[1] + 0.114 * mask[2]
    mask_np = mask_gray.numpy()
    # 简单二值化
    threshold = mask_np.mean() + 0.5 * mask_np.std()
    mask_bin = (mask_np > threshold).astype(np.uint8) * 255

    return img, mask_bin


# ============================================================
#  生成静态掩码预览图（供用户参考/绘制）
# ============================================================

def generate_blank_canvas():
    """生成一个 256×256 的白色画布供用户绘制掩码"""
    return np.ones((256, 256), dtype=np.uint8) * 255


def generate_random_mask_preview(mode):
    """生成随机掩码的 256×256 预览"""
    mask = generate_random_mask(LATENT_SIZE, DEVICE, mode=mode)
    if mask is None:
        return np.ones((256, 256), dtype=np.uint8) * 255
    mask_np = mask[0, 0].cpu().numpy()
    mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
    mask_pil = mask_pil.resize((256, 256), Image.NEAREST)
    return np.array(mask_pil)


# ============================================================
#  Gradio UI
# ============================================================

def create_ui():
    # 全局状态：加载的模型
    model_state = {"model": None, "diffusion": None, "vae": None,
                   "model_clip": None, "use_mask": False}

    def on_load_model(ckpt_path, vae_path, model_type):
        if model_type == "标准 DiT（无掩码控制）":
            m, d, v, c = load_standard_model(ckpt_path, vae_path)
            model_state.update(model=m, diffusion=d, vae=v, model_clip=c, use_mask=False)
            return "✅ 标准 DiT 加载成功", gr.update(visible=False)
        else:
            m, d, v, c = load_mask_model(ckpt_path, vae_path)
            model_state.update(model=m, diffusion=d, vae=v, model_clip=c, use_mask=True)
            return "✅ 掩码条件 DiT 加载成功", gr.update(visible=True)

    def on_defect_change(defect_text):
        """缺陷文本变化时自动检测强度并更新 CFG 推荐值"""
        if model_state["model_clip"] is not None:
            _, auto_cfg, label = extract_severity_cfg(
                defect_text, model_state["model_clip"])
        else:
            _, auto_cfg, label = extract_severity_cfg(defect_text, None)
        return gr.update(value=auto_cfg), label

    def on_generate(defect, product, cfg, mask_mode, mask_img, auto_enabled):
        # 如果启用了自动映射，重新从文本提取强度
        if auto_enabled and model_state["model_clip"] is not None:
            clean_defect, auto_cfg, _ = extract_severity_cfg(
                defect, model_state["model_clip"])
            if clean_defect:
                defect = clean_defect
            cfg = auto_cfg
        if model_state["model"] is None:
            return (
                np.zeros((256, 256, 3), dtype=np.uint8),
                np.zeros((256, 256), dtype=np.uint8),
                "⚠️ 请先加载模型"
            )
        try:
            img, mask = generate(
                defect, product, cfg,
                mask_mode, mask_img,
                model_state["model"], model_state["diffusion"],
                model_state["vae"], model_state["model_clip"],
                model_state["use_mask"])
            return img, mask, f"✅ 生成完成 — 缺陷: {defect} | 产品: {product} | CFG: {cfg}"
        except Exception as e:
            return (
                np.zeros((256, 256, 3), dtype=np.uint8),
                np.zeros((256, 256), dtype=np.uint8),
                f"❌ 生成失败: {e}"
            )

    def on_mask_mode_change(mode):
        """根据掩码模式显示/隐藏掩码输入组件"""
        if mode in ("draw", "upload"):
            return gr.update(visible=True)
        return gr.update(visible=False)

    with gr.Blocks(title="DefectDiffu — 缺陷生成") as demo:
        gr.Markdown("""
        # DefectDiffu 文本引导缺陷生成
        ### 支持掩码位置控制 — 指定缺陷在产品的哪个位置生成
        """)

        # ---- 模型加载区 ----
        with gr.Accordion("⚙️ 模型配置", open=True):
            with gr.Row():
                ckpt_input = gr.Textbox(
                    label="检查点路径", value="./model_para/model_1500_prime.pth")
                vae_input = gr.Textbox(
                    label="VAE 路径", value="./VAE")
            with gr.Row():
                model_type = gr.Radio(
                    label="模型类型",
                    choices=["标准 DiT（无掩码控制）", "掩码条件 DiT（支持位置控制）"],
                    value="标准 DiT（无掩码控制）")
                load_btn = gr.Button("加载模型", variant="primary")
            load_status = gr.Textbox(label="状态", interactive=False)

        # ---- 掩码控制区（仅在掩码模式下可见） ----
        mask_panel = gr.Column(visible=False)
        with mask_panel:
            gr.Markdown("### 🎯 缺陷位置控制")
            mask_mode = gr.Radio(
                label="掩码模式",
                choices=[
                    ("无掩码（随机位置）", "none"),
                    ("随机斑点", "blob"),
                    ("随机线条（划痕）", "line"),
                    ("随机噪声点", "noise"),
                    ("手绘 / 上传掩码", "draw"),
                ],
                value="none")

            mask_image = gr.Image(
                label="绘制或上传掩码（白色=缺陷位置，黑色=背景）",
                image_mode="L",
                sources=["upload"],
                type="numpy",
                value=generate_blank_canvas(),
                visible=False)

        # ---- 生成控制区 ----
        with gr.Row():
            with gr.Column(scale=1):
                defect_input = gr.Textbox(
                    label="缺陷类型", value="scratch",
                    placeholder="例如: scratch, 轻微划痕, 严重破损...")
                severity_status = gr.Textbox(
                    label="检测到的强度", value="中等 (CFG≈2.0)",
                    interactive=False, lines=1)
                auto_cfg_checkbox = gr.Checkbox(
                    label="启用语义强度映射（根据描述自动调整 CFG）",
                    value=True)
                product_input = gr.Textbox(
                    label="产品类别", value="bottle",
                    placeholder="例如: bottle, cable, metal_nut...")
                cfg_slider = gr.Slider(
                    minimum=0.5, maximum=4.0, step=0.1,
                    label="引导强度 (CFG Scale)", value=2.0)
                generate_btn = gr.Button("🚀 生成", variant="primary", size="lg")

            with gr.Column(scale=2):
                with gr.Row():
                    img_output = gr.Image(label="生成图像", type="numpy")
                    mask_output = gr.Image(label="缺陷掩码", type="numpy")
                gen_status = gr.Textbox(label="状态", interactive=False)

        # ---- 说明 ----
        gr.Markdown("""
        ---
        ### 使用说明

        **标准模式**：输入缺陷类型和产品类别，调整 CFG 强度生成图像。

        **掩码条件模式**（需先训练掩码条件模型 `DiTMaskConditioned`）：
        1. 选择掩码模式：
           - *随机斑点* — 随机位置的高斯斑点，模拟局部缺陷
           - *随机线条* — 随机方向划痕线条
           - *随机噪声点* — 分散的噪声点
           - *手绘/上传* — 在右侧面板绘制或上传掩码（白色=缺陷位置）
        2. CFG 强度越高，缺陷越明显，但可能降低背景质量

        **语义强度映射**：
        在"缺陷类型"中输入带强度描述的自然语言，系统自动调整 CFG 强度：
        - 轻微/轻度/slight → CFG ≈ 1.0
        - 中等/moderate → CFG ≈ 2.0
        - 严重/重度/severe → CFG ≈ 3.5
        - 示例："轻微划痕" "严重破损" "heavy dent"
        - 关闭"启用语义强度映射"复选框可完全手动控制
        """)

        # ---- 事件绑定 ----
        load_btn.click(
            on_load_model,
            inputs=[ckpt_input, vae_input, model_type],
            outputs=[load_status, mask_panel])

        mask_mode.change(
            on_mask_mode_change,
            inputs=[mask_mode],
            outputs=[mask_image])

        # 缺陷文本变化 → 自动检测强度 → 更新 CFG 滑块 + 强度标签
        defect_input.change(
            on_defect_change,
            inputs=[defect_input],
            outputs=[cfg_slider, severity_status])

        generate_btn.click(
            on_generate,
            inputs=[defect_input, product_input, cfg_slider,
                    mask_mode, mask_image, auto_cfg_checkbox],
            outputs=[img_output, mask_output, gen_status])

    return demo


if __name__ == "__main__":
    ui = create_ui()
    ui.launch(share=True)
