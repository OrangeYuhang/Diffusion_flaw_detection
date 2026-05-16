# 使用 PyTorch 官方镜像，CUDA 12.1 + PyTorch 2.4 稳定版（可自行替换为 nightly）
# 若要匹配你的 torch 2.12.0.dev，请使用 nightly 标签（但稳定版更可靠）
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# 设置工作目录
WORKDIR /workspace

# 复制项目代码（不包括数据、模型大文件）
COPY . /workspace

# 安装系统依赖（可选，用于 OpenCV 等）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
RUN pip uninstall torch torchvision torchaudio -y || true && \
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
# 此处使用基础镜像自带的 torch 2.4.0（兼容 RTX 4090）
RUN pip install --no-cache-dir \
    accelerate==1.13.0 \
    diffusers==0.37.1 \
    einops==0.8.2 \
    omegaconf==2.3.0 \
    opencv-python==4.13.0.92 \
    matplotlib==3.10.8 \
    seaborn==0.13.2 \
    timm==1.0.26 \
    tqdm==4.67.3 \
    pillow==12.2.0 \
    numpy==2.2.6 \
    huggingface-hub==1.10.1 \
    safetensors==0.7.0 \
    ftfy==6.3.1 \
    regex==2026.4.4 \
    requests==2.33.1 \
    pyyaml==6.0.3

# 复制本地已下载的 taming-transformers 目录到镜像
COPY taming-transformers /workspace/taming-transformers
# 安装它（可编辑模式）
RUN pip install --no-cache-dir -e /workspace/taming-transformers

# 创建数据与模型挂载点
RUN mkdir -p /workspace/data/mvtec /workspace/checkpoint /workspace/DiT-256 /workspace/VAE

# 设置默认命令
CMD ["/bin/bash"]