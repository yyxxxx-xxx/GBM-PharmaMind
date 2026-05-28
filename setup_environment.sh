#!/bin/bash
# ============================================================================
# GBM-PharmaMind 环境配置脚本
# 运行前请确保已安装 Python 3.10+ 和 CUDA 11.8+ 环境
# ============================================================================

set -e

echo "=========================================="
echo "GBM-PharmaMind 环境配置"
echo "=========================================="

# --------------------------------------------------------------------------
# 1. 创建并激活 conda 环境（可选，如已存在可跳过）
# --------------------------------------------------------------------------
ENV_NAME="gbm_pharmamind"

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[1/5] conda 环境 '${ENV_NAME}' 已存在，跳过创建"
else
    echo "[1/5] 创建 conda 环境 '${ENV_NAME}'（Python 3.11）..."
    conda create -n ${ENV_NAME} python=3.11 -y
fi

echo "激活 conda 环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# --------------------------------------------------------------------------
# 2. 安装 PyTorch（CUDA 11.8）
# --------------------------------------------------------------------------
echo "[2/5] 安装 PyTorch（CUDA 11.8）..."
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# --------------------------------------------------------------------------
# 3. 安装核心依赖
# --------------------------------------------------------------------------
echo "[3/5] 安装核心依赖..."
pip install \
    transformers==4.40.0 \
    peft==0.10.0 \
    bitsandbytes==0.43.1 \
    accelerate==0.30.1 \
    sentencepiece==0.1.99 \
    protobuf==4.25.3 \
    accelerate==0.30.1

# --------------------------------------------------------------------------
# 4. 安装 RDKit
# --------------------------------------------------------------------------
echo "[4/5] 安装 RDKit..."
pip install rdkit

# --------------------------------------------------------------------------
# 5. 安装其他辅助依赖
# --------------------------------------------------------------------------
echo "[5/5] 安装辅助依赖（matplotlib、seaborn、pandas 等）..."
pip install \
    matplotlib \
    seaborn \
    pandas \
    numpy \
    pyyaml \
    scipy \
    scikit-learn \
    jupyterlab

echo ""
echo "=========================================="
echo "环境配置完成！"
echo "=========================================="
echo "激活环境命令："
echo "  conda activate ${ENV_NAME}"
echo ""
echo "验证安装（运行以下命令检查）："
echo "  python -c 'import torch; print(torch.cuda.is_available())'"
echo "  python -c 'import rdkit; print(rdkit.__version__)'"
echo "  python -c 'import transformers; print(transformers.__version__)'"
echo ""
