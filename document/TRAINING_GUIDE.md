# DefectDiffu 训练流程指南

## 一、标准 DiT 训练（`train.py`）

### 目标

在 DiT-XL/2 预训练权重上微调，学会根据文本描述生成工业缺陷图像及其分割掩码。

### 命令

```bash
python train.py \
  --ckpt ./DiT-256/DiT-XL-2-256x256.pt \
  --batchsize 4 \
  --vae ./VAE \
  --data ./data/mvtec \
  --bf16 \
  --free 1 \
  --good_ratio 0.4 \
  --feature_loss \
  --feature_loss_freq 10 \
  --tensorboard \
  --log_dir ./runs/train
```

### 关键参数

| 参数 | 推荐值 | 说明 |
|------|:---:|------|
| `batchsize` | 4（FP32）/ 6（BF16） | 24GB 显存上限 |
| `free` | 1 | 2 分支 CFG，推理时可调节强度 |
| `good_ratio` | 0.3–0.4 | 正常样本比例，抑制背景伪影 |
| `feature_loss_freq` | 10 | 匹配快速收敛特性 |
| `bf16` | 启用 | 省 30% 显存，已验证数值稳定 |

### 预期收敛

```
epoch  0:  loss≈4.5,  mask_mse≈2.6,  mask_dice≈0.33
epoch  5:  loss≈2.1,  mask_mse≈0.8,  mask_dice≈0.22
epoch 30:  loss≈1.9,  mask_mse≈0.7,  mask_dice≈0.22
```

mask_dice 和 att_align 在 epoch 5 后基本停滞——这是标准 DiT 的架构上限（纯文本语义推断位置），属于正常现象。

### 输出

| 产物 | 路径 |
|------|------|
| 训练日志 | `loss_log.csv` |
| 中间 checkpoint（每 100 epoch） | `checkpoint/model_{epoch}.pth` |
| 最终模型 | `checkpoint/model_final.pth` |
| 损失曲线 | `img/loss_curve.png` |
| TensorBoard | `./runs/train/` |

---

## 二、掩码条件 DiT 训练（`train_mask_condition.py`）— 两阶段

### 设计思想

DiTMaskConditioned 在标准 DiT 基础上增加 MaskEncoder + mask_fusion×28 + mask_to_tokens。采用类似 ControlNet 的两阶段策略：

- **Stage 1**：冻结预训练基类 DiT，高学习率训练新模块——迫使新模块学会将 mask_cond "翻译"为基类理解的 adaLN 条件信号
- **Stage 2**：解冻全参数，低学习率微调——消除基类与新模块之间的阻抗不匹配

### 基类冻结范围（Stage 1）

```
冻结（359 key）:
  x_embedder, t_embedder, y_embedders
  blocks ×28, cross_defect ×10
  adapt_mask (temp_Adaptive_Mask), final_layer

训练（128 key）:
  mask_encoder, mask_fusion ×28, mask_to_tokens
```

### Stage 1 命令

```bash
python train_mask_condition.py \
  --stage 1 \
  --ckpt ./checkpoint/model_400.pth \
  --batchsize 4 \
  --vae ./VAE \
  --data ./data/mvtec \
  --bf16 \
  --free 1 \
  --good_ratio 0.4 \
  --feature_loss \
  --feature_loss_freq 10 \
  --tensorboard \
  --log_dir ./runs/train_mask
```

| 参数 | Stage 1 值 | 说明 |
|------|:---:|------|
| `stage` | 1 | 冻结基类 |
| `lr` | **5e-4** | 高 lr，新模块快速收敛 |
| `ckpt` | 标准 DiT checkpoint | 提供基类权重 |
| 训练参数量 | ~35M / 710M | 仅 MaskEncoder + mask_fusion + mask_to_tokens |

预期 epoch 50–80 时 mask_dice 开始松动下降。

### Stage 2 命令

```bash
python train_mask_condition.py \
  --stage 2 \
  --ckpt ./checkpoint/model_mask_cond_s1_final.pth \
  --batchsize 4 \
  --vae ./VAE \
  --data ./data/mvtec \
  --bf16 \
  --free 1 \
  --good_ratio 0.4 \
  --feature_loss \
  --feature_loss_freq 10 \
  --tensorboard \
  --log_dir ./runs/train_mask
```

| 参数 | Stage 2 值 | 说明 |
|------|:---:|------|
| `stage` | 2 | 全参数解冻 |
| `lr` | **2e-5** | 低 lr，微调对齐 |
| `ckpt` | Stage 1 的 checkpoint | 新模块已预训练 |
| 训练参数量 | 710M / 710M | 全部参数 |

### 输出

| 产物 | Stage 1 | Stage 2 |
|------|------|------|
| 日志 | `loss_log_mask_cond.csv` | 同 |
| 中间 ckpt | `model_mask_cond_s1_{epoch}.pth` | `model_mask_cond_s2_{epoch}.pth` |
| 最终 ckpt | `model_mask_cond_s1_final.pth` | `model_mask_cond_s2_final.pth` |
| 损失曲线 | `img/loss_curve_mask_cond.png` | 同 |

### 预期收敛对比

| 指标 | 标准 DiT 天花板 | Stage 1 预期 | Stage 2 预期 |
|------|:---:|:---:|:---:|
| mask_dice（收敛） | 0.218 | 0.10–0.15 | **0.05–0.10** |
| att_align | 0.0376 | 开始松动 | **突破 0.03** |
| 位置可控性 | 无 | **初步** | **精准** |

---

## 三、训练监控

```bash
# 启动 TensorBoard
tensorboard --logdir ./runs --port 6006 --bind_all
# 浏览器 http://localhost:6006
```

CSV 列含义：

| 列 | 说明 | 健康范围 |
|------|------|:---:|
| `loss` | 总损失 | 2→1（递减） |
| `mse` | 扩散去噪 MSE | < 0.1 |
| `mask_mse` | 掩码像素精度（潜空间） | 2.6→0.7（递减） |
| `mask_dice` | 1−Dice，掩码形状质量 | 0.3→<0.1（递减） |
| `att_align` | 交叉注意力对齐 | 0.037→<0.03（递减） |
| `vb` | 变分下界 | ~0.005 |
| `feat` | 特征一致性 | ~14（稳定即正常） |

**异常信号：**
- `mask_dice` 持续 > 0.5 超 100 epoch → 新模块没学到东西，检查数据加载
- `feat` 突然飙升 → 背景纹理退化
- `att_align` 上升 → 注意力发散（极罕见）
