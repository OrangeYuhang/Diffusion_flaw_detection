# DefectDiffu 项目文件与目录说明

> 完整索引项目中每一个文件与目录的作用。

---

## 一、目录结构总览

```
DefectDiffu-main/
│
├── 📁 diffusion/          # 扩散过程核心（DDPM 前向/反向、采样、loss）
├── 📁 clip/               # CLIP 文本编码器（vendored）
├── 📁 DiT-256/            # 预训练 DiT-XL/2 权重存放目录
├── 📁 VAE/                # Stable Diffusion VAE 权重（sd-vae-ft-mse）
├── 📁 data/mvtec/         # MVTec AD 数据集
├── 📁 checkpoint/         # 训练中间检查点
├── 📁 model_para/         # 已训练模型参数（最终/推理用检查点）
├── 📁 img/                # 生成图像输出目录
├── 📁 document/           # 文档集中目录（论文、说明、实验设计）
├── 📁 __pycache__/        # Python 字节码缓存（自动生成，可忽略）
├── 📁 .gradio/            # Gradio HTTPS 证书（自动生成，可忽略）
├── 📁 .vscode/            # VS Code 编辑器配置
│
├── 📄 *.py                # Python 源码（详见下方分类）
├── 📄 *.yml               # Conda 环境配置
├── 📄 Dockerfile          # Docker 镜像构建文件
├── 📄 .dockerignore       # Docker 构建忽略规则
├── 📄 .cmd                # 常用命令速查
└── 📄 diffusion.zip       # diffusion 目录压缩包（分发用）
```

---

## 二、目录详解

### `diffusion/` — 扩散过程核心

| 文件 | 作用 |
|------|------|
| `__init__.py` | 暴露 `create_diffusion()` 工厂函数，创建配置好的 `SpacedDiffusion` 实例 |
| `gaussian_diffusion.py` | **核心模块**。`GaussianDiffusion` 类实现 DDPM：前向扩散 q(x_t\|x_0)、反向采样 p(x_{t-1}\|x_t)、训练 loss（含改进的 fg/bg 分离 + Dice mask loss） |
| `diffusion_utils.py` | 工具函数：KL 散度、高斯 log-likelihood、标准正态 CDF 近似 |
| `respace.py` | `SpacedDiffusion` 类：支持 respaced timesteps（如 1000 步 → 50 步），通过 `_WrappedModel` 做 timestep 映射 |
| `timestep_sampler.py` | 训练时 timestep 采样策略：`UniformSampler`（均匀）、`LossSecondMomentResampler`（按 loss 加权） |

### `clip/` — CLIP 文本编码器

| 文件 | 作用 |
|------|------|
| `__init__.py` | 暴露 `clip.load()` 接口 |
| `clip.py` | CLIP 模型加载与预处理逻辑 |
| `model.py` | CLIP 模型架构（Transformer + ViT） |
| `simple_tokenizer.py` | BPE 分词器，将文本转为 token IDs |

### `DiT-256/` — 预训练 DiT 权重

| 文件 | 作用 |
|------|------|
| `DiT-XL-2-256x256.pt` | Facebook 官方 DiT-XL/2 预训练权重（256×256 ImageNet），作为微调起点 |

### `VAE/` — Stable Diffusion VAE 权重

| 文件 | 作用 |
|------|------|
| `config.json` | VAE 模型配置（kl-f8，4 通道潜空间，8× 下采样） |
| `diffusion_pytorch_model.bin` | VAE 权重（PyTorch 格式） |
| `diffusion_pytorch_model.safetensors` | VAE 权重（safetensors 格式，HF 推荐） |
| `gitattributes` | HuggingFace LFS 配置 |
| `README.md` | Stability AI 官方 VAE 说明（sd-vae-ft-mse） |

### `data/mvtec/` — MVTec AD 数据集

| 路径 | 内容 |
|------|------|
| `<class>/train/good/` | 正常（无缺陷）训练图像 |
| `<class>/test/<defect>/` | 各类缺陷的测试图像 |
| `<class>/ground_truth/<defect>/` | 像素级缺陷分割掩码 |

### `checkpoint/` — 训练中间检查点

存放训练过程中周期性保存的检查点（`model_*.pth`），训练完成后可将最终检查点移入 `model_para/`。

### `model_para/` — 已训练模型参数

| 文件 | 作用 |
|------|------|
| `model_1500.pth` | 训练 1500 epoch 的最终检查点，供推理使用 |

### `img/` — 生成输出

| 文件 | 作用 |
|------|------|
| `fig 2 architecture.jpg` | 论文中的架构图 |

### `document/` — 文档集中目录

| 文件 | 作用 |
|------|------|
| `FILE_INDEX.md` | 本文件 — 项目文件与目录索引 |
| `INNOVATIONS.md` | 优化与创新技术说明。原模型不足分析、改进原理、解决的问题 |
| `实验流程.md` | 实验设计框架。5 组实验方案（few-shot 消融、合成样本数量、传统增强对比等） |
| `defectdiffu.pdf` | 论文 PDF（ECCV 2024, Few-shot Defect Image Generation based on Consistency Modeling） |
| `defectdiffu-mono.pdf` | 论文 PDF（单色版） |
| `defectdiffu-dual.pdf` | 论文 PDF（双色版） |
| `扩散检瑕-陈宇航(1).docx` | 扩散检瑕相关文档 |
| `工作流程.md` | **新增**。完整端到端工作流程：四阶段路线图、依赖关系、命令、时间线、交付物清单 |

### `.vscode/` — 编辑器配置

| 文件 | 作用 |
|------|------|
| `settings.json` | 指定 Python 环境管理器为 Conda |

### `.gradio/` — Gradio 临时文件

| 文件 | 作用 |
|------|------|
| `certificate.pem` | 本地 HTTPS 自签名证书（Gradio 自动生成） |

---

## 三、核心 Python 文件

### 模型定义

| 文件 | 作用 |
|------|------|
| `models_add_cross_concate.py` | **主模型文件**。`DiT` 类：28 层 DiT-XL/2，含 `CrossAttention`、`Cross_Norm`、`temp_Adaptive_Mask` 掩码解码器。提供 3 种前向路径：`forward`（训练）、`forward_free_2`（2 分支 CFG）、`forward_with_cfg_2`（2 分支 CFG 引导）、`forward_free_3`（3 分支）、`forward_with_cfg_3`（3 分支 CFG 引导）。已改为全通道（4ch）CFG |
| `models_mask_condition.py` | **新增**。`DiTMaskConditioned`：继承自 DiT，增加 `MaskEncoder` 实现显式掩码条件注入，支持推理时用随机掩码控制缺陷位置。含 `generate_random_mask()` 工具 |
| `autoencoder.py` | VAE 相关模块（`Encoder`、`Decoder`、`FrozenAutoencoderKL`、`LinearAttention` 等）。实际训练/推理使用的是 `diffusers.models.AutoencoderKL` |

### 训练

| 文件 | 作用 |
|------|------|
| `train.py` | **主训练脚本**。`Dataset_self` 类：加载 MVTec 数据（缺陷 + 正常），预处理图像和掩码。训练循环：CLIP 文本编码 → VAE 潜空间编码 → DiT 去噪 → fg/bg 分离 loss + Dice mask loss。支持 `--good_ratio`（正常样本比例）、`--feature_loss`（特征空间一致性）、`--free`（CFG 分支策略） |
| `train.bak.py` | 更早版本的训练脚本备份 |
| `feature_loss.py` | **新增**。`PatchCoreFeatureExtractor`：WideResNet-50 layer2+layer3 特征提取。`compute_normal_statistics()`：预计算正常样本特征均值与协方差逆矩阵。`FeatureConsistencyLoss`：Mahalanobis 距离 loss，约束非缺陷区域特征分布。`build_feature_loss()`：一键构建函数 |
| `train_mask_condition.py` | **新增**。掩码条件 DiT 训练脚本。基于 ground truth 掩码训练 `DiTMaskConditioned`，使模型学会在指定位置生成缺陷。复用 fg/bg 分离 loss、Dice mask loss、正常样本训练、特征一致性 loss。输出 `checkpoint/model_mask_cond_*.pth` |
| `consistency_distill.py` | **新增**。一致性蒸馏训练：`consistency_loss()` + `boundary_loss_fn()` + `fast_sample()` 4 步快速采样器。`MVTecDistillDataset` 轻量数据集类 |

### 推理/生成

| 文件 | 作用 |
|------|------|
| `test.py` | **主测试脚本**。遍历 MVTec 所有 class × defect 组合，为每个类别生成图像 + 掩码。使用 `binarize_tensor_iterative()` 对掩码做自适应阈值二值化。输出到 `./img/` |
| `test_gen.py` | 简化版单图生成脚本。从命令行指定的文本 prompt 生成一张缺陷图像。适合快速调试 |
| `generate_diverse.py` | **新增**。多轴多样性生成：CFG scale sweep（0.5–4.0）、noise scale sweep（0.7–1.8）、seed variation、cross-product grid。适用于系统性生成多样化训练数据 |
| `app.py` | Gradio Web UI。支持标准 DiT 与掩码条件 DiT 双模式。交互式界面：缺陷类型 + 产品类别 + CFG scale + 掩码位置控制（斑点/线条/噪声/手绘），实时生成图像和掩码 |

### 数据与流程

| 文件 | 作用 |
|------|------|
| `prepare_dataset.py` | 数据集准备脚本。复制 MVTec 原始正常样本到 `train/good/`；复制测试集、ground_truth；用 DefectDiffu 生成合成缺陷样本存入 `train/bad/`。输出适配 anomalib 的目录结构 |
| `adaptive_generate.py` | **新增**。闭环反馈自适应生成。迭代循环：生成样本 → PatchCore 评估 → 按 AUROC 反比例分配预算 → 下一轮生成。`allocate_budget()` 实现自适应策略 |

---

## 四、配置文件

| 文件 | 作用 |
|------|------|
| `environment.yml` | Conda 环境配置（主）。Python 3.10 + PyTorch 2.12 nightly CUDA 12.8 + diffusers、timm、einops、gradio 等 |
| `envRTX5070.yml` | RTX 5070 环境配置（含清华镜像 channels，固定版本号） |
| `envRTX5070_linux.yml` | RTX 5070 Linux 环境配置（精简版） |
| `Dockerfile` | Docker 镜像构建。基于 `pytorch/pytorch:2.4.0-cuda12.4`，安装依赖、挂载数据/模型目录 |
| `.dockerignore` | Docker 构建排除：`data/`、`checkpoint/`、`*.pt`、`*.pth`、`*.safetensors`、`__pycache__`、`*.pyc` |
| `.vscode/settings.json` | VS Code：指定 Conda 为 Python 环境管理器 |
| `.cmd` | 常用命令速查（训练/测试/生成/蒸馏/Docker/环境安装） |
| `diffusion.zip` | diffusion 目录的压缩包（方便分发和部署） |

---

## 五、文档（均位于 `document/` 目录）

| 文件 | 作用 |
|------|------|
| `FILE_INDEX.md` | 本文件 — 项目文件与目录索引 |
| `INNOVATIONS.md` | 优化与创新技术说明。原模型不足分析、改进原理、解决的问题、公式与代码片段 |
| `实验流程.md` | 实验设计框架。5 组实验方案（few-shot 消融、合成样本数量、传统增强对比、长尾覆盖、质量消融） |
| `defectdiffu.pdf` | 论文 PDF（ECCV 2024, Few-shot Defect Image Generation based on Consistency Modeling） |
| `defectdiffu-mono.pdf` | 论文 PDF（单色版） |
| `defectdiffu-dual.pdf` | 论文 PDF（双色版） |
| `扩散检瑕-陈宇航(1).docx` | 扩散检瑕相关文档 |
| `工作流程.md` | 完整端到端工作流程：四阶段路线图、依赖关系、命令速查、时间线、交付物清单 |

此外：
| `README.md` | 根目录官方 README。论文简介、预训练模型下载链接、训练/测试命令、引用格式 |
| `CLAUDE.md` | 根目录 Claude Code 指导文件。项目概览、命令、架构说明、环境配置、关键细节 |
| `VAE/README.md` | VAE 附带说明文档（sd-vae-ft-mse 训练细节与评估指标） |

---

## 六、依赖关系图

```
训练流程:
  train.py / train_mask_condition.py
  ├── models_add_cross_concate.py (DiT 模型) 或 models_mask_condition.py (DiTMaskConditioned)
  ├── diffusion/ (DDPM 扩散过程 + 训练 loss)
  ├── feature_loss.py ─── 可选：PatchCore 对齐 loss
  ├── clip/ (CLIP 文本编码)
  ├── autoencoder.py (VAE 辅助类)
  ├── DiT-256/DiT-XL-2-256x256.pt (预训练权重)
  ├── VAE/ (SD VAE 权重)
  └── data/mvtec/ (训练数据 + ground_truth 掩码)

生成流程:
  test.py / test_gen.py / generate_diverse.py / app.py
  ├── models_add_cross_concate.py
  ├── models_mask_condition.py ─── app.py 掩码控制模式
  ├── diffusion/ (采样用)
  ├── clip/
  ├── checkpoint/ 或 model_para/ (微调后的检查点)
  └── VAE/

闭环流程:
  adaptive_generate.py
  ├── models_add_cross_concate.py (生成)
  ├── model_para/ 或 checkpoint/ (检查点)
  └── anomalib/PatchCore (外部依赖，检测评估)

蒸馏流程:
  consistency_distill.py
  ├── models_add_cross_concate.py (教师 + 学生)
  ├── diffusion/
  ├── VAE/
  └── model_para/ 或 checkpoint/ (教师权重)

位置解耦:
  models_mask_condition.py
  ├── models_add_cross_concate.py (基类 DiT)
  └── 新增 MaskEncoder + DiTMaskConditioned
```
