#!/usr/bin/env python3
"""
Correct GBM LoRA fine-tuning using pre-trained Llamole adapter as base.
This script addresses the issue where LoRA fine-tuning was done on raw Qwen2 instead of pre-trained Llamole.
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.gbm_lora_finetuner import run_gbm_lora_finetuning


def main():
    # 正确的训练配置
    base_model_path = "/root/Llamole-main/models/Qwen2-7B-Instruct"
    llamole_adapter_path = "/root/Llamole-main/saves/Llamole-Qwen2-7B-Instruct-Adapter"
    dataset_path = "/root/Llamole-main/gbm_project/data/lora_datasets/gbm_evaluation_expanded_lora_train.json"
    output_dir = "/root/Llamole-main/saves/Llamole-Qwen2-7B-Instruct-Adapter-gbm-evaluation-corrected"

    print("=" * 60)
    print("GBM LoRA Fine-tuning - CORRECTED VERSION")
    print("=" * 60)
    print(f"Base model: {base_model_path}")
    print(f"Llamole adapter: {llamole_adapter_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    print()

    # 验证文件存在性
    if not os.path.exists(llamole_adapter_path):
        raise FileNotFoundError(f"Llamole adapter not found: {llamole_adapter_path}")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print("✅ All required files found")

    # LoRA配置 - 为更大数据集优化
    lora_config = {
        "r": 16,  # 标准rank
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "num_epochs": 1,  # 1个epoch，因为数据集更大
        "batch_size": 2,
        "learning_rate": 2e-4,
        "gradient_accumulation_steps": 8,  # 增加梯度累积以处理更大batch
        "save_steps": 500,  # 更频繁保存
        "logging_steps": 50,  # 更频繁日志
        "fp16": True
    }

    print(f"LoRA config: r={lora_config['r']}, alpha={lora_config['lora_alpha']}, lr={lora_config['learning_rate']}")

    # 运行微调
    finetuner = run_gbm_lora_finetuning(
        base_model_path=base_model_path,
        llamole_adapter_path=llamole_adapter_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        lora_config=lora_config
    )

    print("\n" + "=" * 60)
    print("Fine-tuning completed successfully!")
    print(f"Model saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
