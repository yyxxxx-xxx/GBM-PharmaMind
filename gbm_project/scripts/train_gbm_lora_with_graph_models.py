#!/usr/bin/env python3
"""
GBM LoRA微调脚本 - 集成完整Llamole图模型支持
确保在已经经过一轮LoRA微调的Llamole适配器上进行微调
"""

import os
import sys
import torch

# Add the gbm_project directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.gbm_lora_finetuner import run_gbm_lora_finetuning


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # 模型路径 - 使用原始Qwen2模型作为基础
    base_model_path = os.path.join(os.path.abspath(os.path.join(base_dir, "..")), "models", "Qwen2-7B-Instruct")

    # Llamole适配器路径 - 已经在Qwen2基础上训练过一次的LoRA适配器
    llamole_adapter_path = os.path.join(os.path.abspath(os.path.join(base_dir, "..")), "saves", "Llamole-Qwen2-7B-Instruct-Adapter")

    # 图模型路径
    graph_decoder_path = os.path.join(os.path.abspath(os.path.join(base_dir, "..")), "saves", "graph_decoder")
    graph_encoder_path = os.path.join(os.path.abspath(os.path.join(base_dir, "..")), "saves", "graph_encoder")
    graph_predictor_path = os.path.join(os.path.abspath(os.path.join(base_dir, "..")), "saves", "graph_predictor")
    graph_lm_connector_path = os.path.join(os.path.abspath(os.path.join(base_dir, "..")), "saves", "Llamole-Qwen2-7B-Instruct-Adapter", "connector")

    # 数据集路径
    dataset_path = os.path.join(base_dir, "data/lora_datasets/gbm_evaluation_expanded_lora_train.json")

    # 输出路径
    output_dir = os.path.join(os.path.abspath(os.path.join(base_dir, "..")), "saves", "Llamole-Qwen2-7B-Instruct-Adapter-gbm-with-graph-models")

    # 验证路径存在
    paths_to_check = [
        ("Base model", base_model_path),
        ("Llamole adapter", llamole_adapter_path),
        ("Graph decoder", graph_decoder_path),
        ("Graph encoder", graph_encoder_path),
        ("Graph predictor", graph_predictor_path),
        ("Dataset", dataset_path),
    ]

    print("============================================================")
    print("GBM LoRA Fine-tuning with Full Llamole Graph Model Integration")
    print("============================================================")

    for name, path in paths_to_check:
        if os.path.exists(path):
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name}: {path} (NOT FOUND)")
            return

    # 检查连接器路径
    if os.path.exists(graph_lm_connector_path):
        print(f"✅ Graph-LM connectors: {graph_lm_connector_path}")
    else:
        print(f"⚠️  Graph-LM connectors: {graph_lm_connector_path} (NOT FOUND - will use default)")

    # LoRA配置 - 针对GBM微调的保守配置
    lora_config = {
        "r": 8,  # 更小的rank以减少显存使用
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],  # 只微调注意力层
        "num_epochs": 1,  # 单轮微调
        "batch_size": 1,  # 小批量以适应显存
        "learning_rate": 1e-4,  # 较低的学习率
        "gradient_accumulation_steps": 8,  # 梯度累积
        "save_steps": 100,
        "logging_steps": 10,
        "fp16": True,
        "max_grad_norm": 0.3,
        # 图模型集成配置
        "use_llamole_graph_models": True,
        "graph_decoder_path": graph_decoder_path,
        "graph_encoder_path": graph_encoder_path,
        "graph_predictor_path": graph_predictor_path,
        "graph_lm_connector_path": graph_lm_connector_path,
    }

    print(f"\n🔧 LoRA配置:")
    print(f"   • Rank: {lora_config['r']}")
    print(f"   • Alpha: {lora_config['lora_alpha']}")
    print(f"   • Learning rate: {lora_config['learning_rate']}")
    print(f"   • Batch size: {lora_config['batch_size']} (with gradient accumulation: {lora_config['gradient_accumulation_steps']})")
    print(f"   • Target modules: {lora_config['target_modules']}")
    print(f"   • Use Llamole graph models: {lora_config['use_llamole_graph_models']}")

    print(f"\n📊 训练概览:")
    print(f"   • 基础模型: Qwen2-7B-Instruct")
    print(f"   • Llamole适配器: 已预训练 (分子生成能力)")
    print(f"   • 图模型: GraphDiT + GraphCLIP + GraphPredictor")
    print(f"   • 训练数据: GBM评估导向数据集")
    print(f"   • 输出目录: {output_dir}")

    print("\n🚀 开始训练...")
    print("=" * 60)

    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 设置环境变量以减少显存使用
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["SKIP_KBIT_PREP"] = "1"  # 跳过kbit准备以减少显存

        # 运行LoRA微调
        finetuner = run_gbm_lora_finetuning(
            base_model_path=base_model_path,
            dataset_path=dataset_path,
            output_dir=output_dir,
            llamole_adapter_path=llamole_adapter_path,
            lora_config=lora_config
        )

        print("\n" + "=" * 60)
        print("🎉 训练完成!")
        print("=" * 60)
        print(f"📁 模型保存至: {output_dir}")
        print("🔧 集成了完整的Llamole图模型支持")
        print("🎯 可用于GBM药物分子生成和评估")

    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        print("请检查配置和资源使用情况")
        return

    # 验证训练结果
    print("\n🔍 验证训练结果...")
    adapter_model_path = os.path.join(output_dir, "adapter_model.safetensors")
    if os.path.exists(adapter_model_path):
        print("✅ LoRA适配器已保存")
    else:
        print("⚠️  LoRA适配器文件未找到")

    training_args_path = os.path.join(output_dir, "training_args.bin")
    if os.path.exists(training_args_path):
        print("✅ 训练配置已保存")
    else:
        print("⚠️  训练配置文件未找到")


if __name__ == "__main__":
    main()
