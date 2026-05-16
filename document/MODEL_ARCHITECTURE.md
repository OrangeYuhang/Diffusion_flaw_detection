# DefectDiffu 模型架构详细说明

## 一、标准 DiT（`models_add_cross_concate.py`）

### 1.1 总览

基于 Facebook DiT-XL/2 预训练权重微调，在 ImageNet 类别条件生成能力之上学习工业缺陷图像生成。模型在一个前向传播中同时产出**去噪潜变量**和**缺陷分割掩码**。

```
输入:  Noisy Latent (B,4,32,32) + 文本 Prompt
输出:  Denoised Latent (B,8,32,32) + Defect Mask (B,4,32,32)
参数:  ~675M（DiT-XL/2 骨干）
```

### 1.2 组件详解

#### 输入嵌入层

| 组件 | 类 | 规格 |
|------|------|------|
| Patch Embedding | `PatchEmbed` (timm) | 4ch → 1152d，patch_size=2，产生 256 tokens |
| 位置嵌入 | `pos_embed` (冻结) | (1, 256, 1152) 二维正弦余弦位置编码 |
| 时间步嵌入 | `TimestepEmbedder` | 标量 t → 256d 频率编码 → MLP → 1152d |
| 文本嵌入 | `nn.Linear(1024, 1152)` | CLIP RN50 输出的 1024d → 1152d |

文本嵌入分为三种，对应三层条件机制：

| 嵌入 | 来源 | 用途 |
|------|------|------|
| `y_class` | `"a photo of {product}"` | Blocks 0–9 条件 |
| `y_defect` | `"a photo of {defect}"` | Blocks 10–19 交叉注意力条件 |
| `y_all` | `"a photo of {defect} {product}"` | Blocks 20–27 条件 |

#### DiTBlock（28 层）

```
DiTBlock (hidden_size=1152, num_heads=16, mlp_ratio=4.0)
├── norm1: LayerNorm(1152, eps=1e-6, elementwise_affine=False)
├── attn:  Multi-Head Self-Attention (16 heads, qkv_bias=True)
├── norm2: LayerNorm(1152, eps=1e-6, elementwise_affine=False)
├── mlp:   Mlp(1152 → 4608 → 1152, GELU, drop=0)
└── adaLN_modulation: Sequential(SiLU, Linear(1152 → 6×1152))
```

adaLN-Zero 调制：条件向量 c 经过 `adaLN_modulation` 产生 6 组参数 `(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)`，通过 `modulate(x, shift, scale) = x * (1 + scale) + shift` 注入自注意力和 MLP。

#### CrossAttention（blocks 10–19 专用）

```python
CrossAttention (query_dim=1152, heads=16)
├── to_q: Linear(1152 → 1152)    # 对 patch tokens 投影
├── to_k: Linear(1152 → 1152)    # 对条件向量投影 → (B,1,D)
├── to_v: Linear(1152 → 1152)    # 对条件向量投影 → (B,1,D)
└── scale: sqrt(dim_head) = sqrt(1152/16) ≈ 8.485
```

关键设计：key 和 value 是**单向量**（unsqueeze 到 seq_len=1）。对 N=256 个 query patch 做 attention 时，softmax 在 dim=-2（query 维度）上归一化，产生 (B×16, 256, 1) 的注意力图——即每个 patch 相对于缺陷文本的"关注度"。

#### Cross_Norm（封装 CrossAttention）

```python
Cross_Norm (hidden_size=1152, num_heads=16)
├── norm: LayerNorm(1152, eps=1e-6)
└── cross_attention: CrossAttention(1152, heads=16)
```

参数总计：10 层 Cross_Norm，每层约 8M 参数（QKV 投影各 1152² ≈ 1.3M × 3 ≈ 4M，加上其他开销）。在 blocks 10–19 中，每层的交叉注意力输出累加到 `att_map` 列表中。

#### temp_Adaptive_Mask（掩码解码器）

```python
temp_Adaptive_Mask (input_dim=160, patch_size=2, out_channels=4)
├── norm:   LayerNorm(160, eps=1e-6)
├── mlp:    Sequential(SiLU, Linear(160→320), Linear(320→160))
├── linear: Linear(160 → 2²×4 = 16)
├── reshape + einops 重组: (B,256,16) → (B,4,32,32)
└── tanh:   约束输出到 [-1, 1]
```

输入：10 层交叉注意力图的拼接 `(B, 256, 10×16=160)`。输出：潜空间缺陷掩码 `(B, 4, 32, 32)`。

#### FinalLayer（去噪输出头）

```python
FinalLayer (hidden_size=1152, patch_size=2, out_channels=8)
├── norm_final: LayerNorm(1152, eps=1e-6)
├── adaLN_modulation: Sequential(SiLU, Linear(1152 → 2×1152))
└── linear: Linear(1152 → 2²×8 = 32)
```

输出 8 通道：前 4 通道为预测均值 ε_θ，后 4 通道为学习到的方差 v_θ（供 VB loss 使用）。经 `unpatchify` 重组为 `(B, 8, 32, 32)`。

### 1.3 三阶段条件机制

```
阶段 I   (Blocks 0–9):   c = t_emb + y_class
         纯自注意力，注入"这是什么产品"的信息
         
阶段 II  (Blocks 10–19): c = t_emb + y_defect
         自注意力 + 交叉注意力，模型"看向"缺陷文本
         → 注意力热力图收集到 att_map
         
阶段 III (Blocks 20–27): c = t_emb + y_all
         纯自注意力，融合产品+缺陷的完整语义
```

### 1.4 前向传播路径

#### `forward(x, t, y)` — 训练路径

1. Patch Embed + 位置编码 → `(B, 256, 1152)`
2. 时间步嵌入 + CLIP 文本三重编码
3. 28 层 DiT Block 顺序处理，blocks 10–19 中收集 `att_map`
4. 主路径 → `FinalLayer` → `unpatchify` → 去噪潜变量 `(B, 8, 32, 32)`
5. 侧路径 → `att_map` 拼接 `(B, 256, 160)` → `temp_Adaptive_Mask` → 掩码 `(B, 4, 32, 32)`
6. 返回 `(x, att_mask, loss_att)`

#### `forward_with_cfg_2(x, t, y, cfg_scale)` — 推理路径（2 分支 CFG）

将 batch 对半拆分：条件分支（有缺陷文本）+ 无条件分支（文本替换为 "good"），联合前向。对前 4 通道做 CFG：
```
eps = uncond_eps + cfg_scale × (cond_eps - uncond_eps)
```

#### `forward_with_cfg_3(x, t, y, cfg_scale)` — 推理路径（3 分支 CFG）

增加 `uncond_defect` 分支（仅去掉缺陷词，保留产品词），三路加权。

### 1.5 权重初始化

| 组件 | 初始化方式 |
|------|------|
| 所有 Linear（默认） | Xavier uniform |
| pos_embed | 正弦余弦 2D 位置编码（冻结） |
| x_embedder.proj | Xavier uniform |
| t_embedder MLP | Normal(std=0.02) |
| adaLN 调制层（最后 Linear） | **零初始化**（weight=0, bias=0） |
| final_layer 输出层 | **零初始化** |

adaLN 和输出层的零初始化保证训练初期模型行为等同恒等映射，稳定训练。

---

## 二、DiTMaskConditioned（`models_mask_condition.py`）

### 2.1 总览

继承自标准 DiT，新增三个模块实现**显式掩码条件注入**。训练时用 ground truth 掩码作为位置条件，推理时用任意掩码（斑点/线条/噪声/手绘）控制缺陷生成位置。

```
输入:  Noisy Latent (B,4,32,32) + 文本 Prompt + mask_cond (B,1,32,32)
输出:  Denoised Latent (B,8,32,32) + Defect Mask (B,4,32,32)
新增参数: ~35M（MaskEncoder + mask_fusion×28 + mask_to_tokens）
```

### 2.2 新增组件

#### MaskEncoder

```python
MaskEncoder (hidden_size=1152, input_size=32)
├── conv: Sequential(
│     Conv2d(1→64,   k3,s2,p1) + SiLU    # 32→16
│     Conv2d(64→128, k3,s2,p1) + SiLU    # 16→8
│     Conv2d(128→256,k3,s2,p1) + SiLU    # 8→4
│     Conv2d(256→512,k3,s2,p1) + SiLU    # 4→2
│   )
├── global_proj: Sequential(
│     AdaptiveAvgPool2d(1)  # (B,512,2,2) → (B,512,1,1)
│     Flatten               # (B,512)
│     Linear(512→1152) + SiLU
│     Linear(1152→1152)
│   )
└── spatial_proj: Conv2d(512→1152, k1)  # (B,512,2,2) → (B,1152,2,2)
```

输入 `mask_cond (B,1,32,32)`，输出：
- `global_cond (B,1152)` — 全局掩码嵌入，与时间/文本条件拼接
- `spatial_cond (B,1152,2,2)` — 空间掩码特征，上采样后注入 token 序列

#### mask_fusion（28 层）

```python
mask_fusion[i]: Sequential(
    Linear(1152×2 → 1152)   # 拼接 global_cond + 基础条件
    SiLU
    Linear(1152 → 6×1152)   # 输出 6 组 adaLN 参数
)
```

每一层 DiT Block 都有独立的 `mask_fusion` 层。取代原 DiT Block 的 `adaLN_modulation`：将基础条件 `c_base`（时间 + 文本）与 `global_cond` 拼接，通过 MLP 融合后产生 adaLN 的 6 组参数。

对比：

| | 标准 DiT | DiTMaskConditioned |
|------|------|------|
| 条件来源 | `t_emb + text` | `concat(t_emb + text, global_cond)` |
| 生成参数 | `adaLN_modulation` 一层 Linear | `mask_fusion[i]` 两层 MLP |
| 层数 | 仅 blocks 自带 | 28 层独立融合 |

#### mask_tokens（空间偏置注入）

```python
# spatial_cond (B,1152,2,2) → 上采样到 (B,1152,16,16) → 展平 → (B,256,1152)
x = x + 0.1 * mask_tokens   # 微小空间偏置
```

将掩码的空间特征上采样匹配 DiT 的 patch 网格 (16×16 = 256 tokens)，以 0.1 的微小权重加到 token 序列上，提供空间位置引导。

### 2.3 完整前向路径：`forward_with_mask`

```
1. Patch Embedding + 位置嵌入                      → (B,256,1152)
2. 时间步嵌入 + CLIP 三重文本编码
3. mask_cond ──→ MaskEncoder ──→ global_cond (B,1152)
   │                            └─→ spatial_cond (B,1152,2,2)
   │                                     │
   │                              Upsample 16×16 → flatten
   │                                     │
   │                              mask_tokens (B,256,1152)
   │                                     │
   │                              x = x + 0.1 * mask_tokens
   │
4. 28 层 DiT Block:
   ├── Blocks 0–9:   c_base = t_emb + y_class
   │                 adaLN = mask_fusion[i](concat(c_base, global_cond))
   │
   ├── Blocks 10–19: c_base = t_emb + y_defect
   │                 adaLN = mask_fusion[i](concat(c_base, global_cond))
   │                 CrossAttention(x, c_base) → att_map 收集
   │
   └── Blocks 20–27: c_base = t_emb + y_all
                     adaLN = mask_fusion[i](concat(c_base, global_cond))

5. FinalLayer(x, t_emb + y_all) → unpatchify → (B,8,32,32)
6. att_map 拼接 → temp_Adaptive_Mask → tanh → (B,4,32,32)
7. 返回 (x, att_mask, loss_att)
```

### 2.4 `forward()` 重载 — 与训练管线兼容

```python
def forward(self, x, t, y, mask_cond=None):
    if mask_cond is not None:
        return self.forward_with_mask(x, t, y, mask_cond)
    return super().forward(x, t, y)
```

`diffusion.training_losses` 调用 `model(x_t, t, **model_kwargs)` 时，若 `model_kwargs` 中包含 `mask_cond`，自动路由到掩码条件路径；否则回退到标准 DiT 的 `forward()`。这使得同一套训练扩散基础设施无需修改即可支持掩码条件训练。

---

## 三、两模型核心差异对比

| 维度 | 标准 DiT | DiTMaskConditioned |
|------|------|------|
| **输入** | 文本（缺陷+产品） | 文本 + 空间掩码 `mask_cond` |
| **条件注入** | adaLN 仅含时间+文本 | adaLN 含时间+文本+掩码全局嵌入 |
| **空间引导** | 无（仅交叉注意力软约束） | `mask_tokens` 空间偏置直接注入 token 序列 |
| **交叉注意力** | 10 层，key 从缺陷文本嵌入投影 | 相同结构，但额外接收 mask_fusion 调制 |
| **推理可控性** | 无法指定缺陷位置 | 可用随机/手绘掩码控制位置 |
| **总参数量** | ~675M | ~710M |
| **训练数据需求** | 缺陷图像 + GT 掩码（仅监督） | 缺陷图像 + GT 掩码（监督 + 输入条件） |
| **输出** | 去噪潜变量 + 缺陷掩码 | 相同 |
| **checkpoint 兼容** | 独立 checkpoint | 加载标准 DiT checkpoint（strict=False），MaskEncoder/mask_fusion 随机初始化微调 |

## 四、关键设计决策

### 为什么掩码是两个独立输出

去噪潜变量（任务：重建干净图像）和缺陷掩码（任务：分割缺陷区域）是两个**不同语义**的目标。共享主干预训练特征表示（"这是什么缺陷、在哪"），但各自需要专用的输出头来解码。

### 为什么 att_map 用 softmax(dim=-2)

标准交叉注意力 softmax 在 key 维度（dim=-1），但当 key 只有一个向量时，dim=-1 的 softmax 恒为 1.0。改为 dim=-2 在 query（空间位置）维度上归一化，使得不同空间位置的注意力有区分度——这正是生成空间热力图所需的。

### 为什么 temp_Adaptive_Mask 输出加 tanh

VAE 潜空间值域约 [-3, 3]，tanh 约束到 [-1, 1] 在保持语义的同时限制了极端值，避免后续 Dice loss 计算中的数值不稳定。

### 为什么 mask_fusion 是 28 层而非仅 10 层

位置信息对每一层变换都有影响。仅在交叉注意力层（10–19）注入掩码条件会导致早期（0–9）和后期（20–27）层的特征表示缺少空间感知，降低生成缺陷与指定位置的对应精度。全层注入确保位置意识贯穿整个 Transformer。
