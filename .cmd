# =====================================================================
#  DefectDiffu 命令速查（2026/05 更新）
# =====================================================================

# ============================
# 阶段一：标准 DiT 训练
# ============================

# 本地训练（24GB 显存，batchsize 4-6）
python train.py --ckpt ./DiT-256/DiT-XL-2-256x256.pt --batchsize 4 --vae ./VAE --data ./data/mvtec

# 云端训练（AutoDL / 共享存储，num_workers=0 防死锁）
python train.py --ckpt ./DiT-256/DiT-XL-2-256x256.pt --batchsize 4 --vae ./VAE --data ./data/mvtec --num_workers 0

# 调整正常样本比例
python train.py --ckpt ./DiT-256/DiT-XL-2-256x256.pt --batchsize 4 --vae ./VAE --data ./data/mvtec --good_ratio 0.4

# 启用 PatchCore 特征空间一致性损失（建议 feq≥50，跳过 epoch 0）
python train.py --ckpt ./DiT-256/DiT-XL-2-256x256.pt --batchsize 2 --vae ./VAE --data ./data/mvtec --feature_loss --feature_loss_freq 100

python train.py --ckpt ./DiT-256/DiT-XL-2-256x256.pt --batchsize 8 --vae ./VAE --data ./mvtec --feature_loss --feature_loss_freq 10 --num_workers 4 --good_ratio 0.4 --tensorboard --log_dir ./runs/train --bf16

# ============================
# 阶段三：掩码条件 DiT 训练（两阶段）
# ============================

# —— Stage 1: 冻结基类 DiT，仅训练 MaskEncoder + mask_fusion，lr=5e-4 ——
python train_mask_condition.py --stage 1 --ckpt ./model_para/model_400.pth --batchsize 4 --vae ./VAE --data ./data/mvtec --bf16 --good_ratio 0.4 --feature_loss --feature_loss_freq 10

# —— Stage 2: 全参数微调，lr=2e-5（需 Stage1 的 checkpoint） ——
python train_mask_condition.py --stage 2 --ckpt ./checkpoint/model_mask_cond_s1_final.pth --batchsize 4 --vae ./VAE --data ./data/mvtec --bf16 --good_ratio 0.4 --feature_loss --feature_loss_freq 10

# ============================
# 生成与测试
# ============================

# 全类别生成（test.py）
python test.py --ckpt ./model_para/model_plus.pth --vae ./VAE --data ./data/mvtec

# 单图快速验证
python test_gen.py

# 多样性生成（CFG / noise / seed 扫描）
python generate_diverse.py --ckpt ./model_para/model_plus.pth --vae ./VAE --defect scratch --product bottle --num_per_axis 5

# ============================
# 阶段二：闭环自适应生成 & 消融实验
# ============================

python adaptive_generate.py --ckpt ./model_para/model_plus.pth --vae ./VAE --data ./data/mvtec --num_iterations 3 --total_budget 200

# ============================
# 阶段四：一致性蒸馏（50 步 → 4 步）
# ============================

python consistency_distill.py --teacher_ckpt ./model_para/model_plus.pth --vae ./VAE --data ./data/mvtec --num_student_steps 4 --num_epochs 50

# ============================
# Gradio 交互演示
# ============================

# 启动 Web UI（语义强度映射 + 掩码位置控制）
python app.py

# ============================
# Docker
# ============================

docker run --gpus all -it -v ./data/mvtec:/workspace/data/mvtec -v ./DiT-256:/workspace/DiT-256 -v ./VAE:/workspace/VAE -v ./checkpoint:/workspace/checkpoint -v ./model_para:/workspace/model_para defectdiffu:linux

# ============================
# TensorBoard 实时监控
# ============================

# 训练时启用 TensorBoard（需加 --tensorboard）
python train.py ... --tensorboard --log_dir ./runs/train
python train_mask_condition.py ... --tensorboard --log_dir ./runs/train_mask

# 另开终端启动 TensorBoard（端口 6006）
tensorboard --logdir ./runs/train --port 6006 --bind_all
# 浏览器打开 http://localhost:6006 实时查看 loss/vb/mse/mask_mse/mask_dice/att_align/feat 曲线

# ============================
# 环境安装
# ============================

# PyTorch 2.x nightly + CUDA 12.8
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall

# 完整环境（Conda）
conda env create -f environment.yml
export LD_PRELOAD=/root/miniconda3/envs/defectdiffu/lib/libstdc++.so.6

# ============================
# 参数说明
# ============================
# --batchsize         : 训练批大小（24GB: 2-4 标准/2 掩码条件）
# --num_workers       : DataLoader 进程数（云端 NFS 建议 0，本地 2-4）
# --good_ratio        : 正常样本相对缺陷样本的比例（默认 0.3）
# --feature_loss      : 启用 PatchCore 特征空间一致性损失
# --feature_loss_freq : 特征损失计算间隔 epoch（建议≥50）
# --free              : CFG 分支策略 0/1/2
