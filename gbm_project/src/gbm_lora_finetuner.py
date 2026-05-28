"""
GBM LoRA微调实现
基于药物-细胞状态连接性数据进行LoRA微调
参考scFOCAL项目的药物语义化和连接性标注方法
集成完整的Llamole图模型支持
"""

import os
import torch
import json
import torch.nn as nn
from typing import Dict, List, Any, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training
)
from datasets import Dataset
import numpy as np

# Import Llamole components
try:
    from src.model.loader import (
        load_graph_decoder,
        load_graph_encoder,
        load_graph_predictor,
        load_language_model
    )
    from src.model.modeling_llamole import GraphLLMForCausalMLM
    from src.hparams.model_args import ModelArguments
    LLAMOLE_AVAILABLE = True
except ImportError as e:
    LLAMOLE_AVAILABLE = False
    print(f"Warning: Llamole components not available ({e}), falling back to standard PEFT")


class GBMLoRAFinetuner:
    """GBM LoRA微调器 - 集成完整Llamole图模型支持"""

    def __init__(self, base_model_path: str, lora_config: Dict[str, Any],
                 use_llamole_graph_models: bool = False,
                 graph_decoder_path: str = None,
                 graph_encoder_path: str = None,
                 graph_predictor_path: str = None,
                 graph_lm_connector_path: str = None):
        self.base_model_path = base_model_path
        self.lora_config = lora_config
        self.use_llamole_graph_models = use_llamole_graph_models
        self.graph_decoder_path = graph_decoder_path
        self.graph_encoder_path = graph_encoder_path
        self.graph_predictor_path = graph_predictor_path
        self.graph_lm_connector_path = graph_lm_connector_path
        self.tokenizer = None
        self.model = None
        self.graph_decoder = None
        self.graph_encoder = None
        self.graph_predictor = None
        self.connectors = None

    def load_model_and_tokenizer(self, llamole_adapter_path: str = None):
        """加载基础模型和tokenizer"""
        print(f"Loading model from {self.base_model_path}")

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            trust_remote_code=True,
            padding_side="right",
            local_files_only=True
        )

        # 添加特殊token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载基础模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )

        # 如果提供了Llamole适配器，加载它
        if llamole_adapter_path:
            print(f"Loading Llamole adapter from {llamole_adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, llamole_adapter_path)

        print(f"Model loaded: {self.model.__class__.__name__}")
        print(f"Model parameters: {self.model.num_parameters():,}")

        # 加载Llamole图模型（如果启用）
        self.load_llamole_graph_models()

        # 设置完整的Llamole模型（如果图模型加载成功）
        self.setup_llamole_model()

    def load_llamole_graph_models(self):
        """加载Llamole图模型组件"""
        if not self.use_llamole_graph_models or not LLAMOLE_AVAILABLE:
            print("Skipping Llamole graph model loading (not enabled or not available)")
            return

        print("Loading Llamole graph models...")

        # 创建ModelArguments对象 - 只使用必要的参数
        model_args = ModelArguments(
            model_name_or_path=self.base_model_path,
            graph_decoder_path=self.graph_decoder_path,
            graph_encoder_path=self.graph_encoder_path,
            graph_predictor_path=self.graph_predictor_path,
            graph_lm_connector_path=self.graph_lm_connector_path
        )

        # 设置额外的属性（如果需要）
        if hasattr(model_args, 'compute_dtype'):
            model_args.compute_dtype = torch.float16
        if hasattr(model_args, 'disable_graph_model_gradient'):
            model_args.disable_graph_model_gradient = True
        if hasattr(model_args, 'print_param_status'):
            model_args.print_param_status = False

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 加载图模型
        try:
            self.graph_decoder = load_graph_decoder(model_args, self.graph_decoder_path, device)
            print("✓ Graph decoder loaded")

            self.graph_encoder = load_graph_encoder(model_args, self.graph_encoder_path, device)
            print("✓ Graph encoder loaded")

            self.graph_predictor = load_graph_predictor(model_args, self.graph_predictor_path, device)
            print("✓ Graph predictor loaded")

            # 加载连接器权重
            if self.graph_lm_connector_path and os.path.exists(self.graph_lm_connector_path):
                self.connectors = {}
                connector_files = ['graph_to_lm_connector.pt', 'lm_to_graph_decoder.pt', 'lm_to_graph_predictor.pt']

                for connector_file in connector_files:
                    connector_path = os.path.join(self.graph_lm_connector_path, connector_file)
                    if os.path.exists(connector_path):
                        self.connectors[connector_file] = torch.load(connector_path, map_location=device)
                        print(f"✓ Connector {connector_file} loaded")
                    else:
                        print(f"⚠ Connector {connector_file} not found")

            print("✓ All Llamole graph models loaded successfully")

        except Exception as e:
            print(f"❌ Failed to load Llamole graph models: {e}")
            print("Falling back to language model only")
            self.use_llamole_graph_models = False

    def setup_llamole_model(self):
        """设置完整的Llamole模型"""
        if not self.use_llamole_graph_models:
            return

        print("Setting up complete Llamole model with graph components...")

        # 创建token_id_dict（用于特殊token）
        token_id_dict = {}
        new_special_tokens = ["<design_start>", "<design_end>", "<design_body>", "<molecule>", "<retro_start>", "<retro_end>", "<retro_body>", "<rollback_start>", "<rollback_end>"]
        for token in new_special_tokens:
            if token in self.tokenizer.get_vocab():
                token_id_dict[token] = self.tokenizer.get_vocab()[token]

        # 创建Llamole模型
        try:
            llamole_model = GraphLLMForCausalMLM(
                model_args=None,  # 我们不需要完整的model_args
                finetuning_args=None,
                data_args=None,
                language_model=self.model,
                graph_decoder=self.graph_decoder,
                graph_predictor=self.graph_predictor,
                graph_encoder=self.graph_encoder,
                token_id_dict=token_id_dict,
                tokenizer=self.tokenizer,
            )

            # 设置连接器
            if self.connectors:
                if 'graph_to_lm_connector.pt' in self.connectors:
                    llamole_model.graph_to_lm_connector.load_state_dict(self.connectors['graph_to_lm_connector.pt'])
                if 'lm_to_graph_decoder.pt' in self.connectors:
                    llamole_model.lm_to_graph_decoder.load_state_dict(self.connectors['lm_to_graph_decoder.pt'])
                if 'lm_to_graph_predictor.pt' in self.connectors:
                    llamole_model.lm_to_graph_predictor.load_state_dict(self.connectors['lm_to_graph_predictor.pt'])

            self.model = llamole_model
            print("✓ Complete Llamole model setup successful")

        except Exception as e:
            print(f"❌ Failed to setup Llamole model: {e}")
            print("Falling back to language model only")
            self.use_llamole_graph_models = False

    def setup_lora_config(self) -> LoraConfig:
        """设置LoRA配置"""
        lora_config = LoraConfig(
            r=self.lora_config.get("r", 16),
            lora_alpha=self.lora_config.get("lora_alpha", 32),
            target_modules=self.lora_config.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            lora_dropout=self.lora_config.get("lora_dropout", 0.05),
            bias=self.lora_config.get("bias", "none"),
            task_type=self.lora_config.get("task_type", "CAUSAL_LM")
        )

        return lora_config

    def prepare_model_for_training(self):
        """准备模型进行LoRA训练"""
        # 应用LoRA配置
        lora_config = self.setup_lora_config()
        self.model = get_peft_model(self.model, lora_config)

        # 打印可训练参数
        self.model.print_trainable_parameters()

        # 根据环境变量决定是否准备模型用于kbit训练（内存敏感环境可跳过）
        skip_kbit = os.environ.get("SKIP_KBIT_PREP", "0") == "1"
        if skip_kbit:
            print("SKIP_KBIT_PREP=1 -> 跳过 prepare_model_for_kbit_training()（减少显存峰值）")
        else:
            try:
                # 准备模型用于kbit训练（如果需要）
                self.model = prepare_model_for_kbit_training(self.model)
            except Exception as e:
                # 如果在准备过程中出现 OOM 或其他错误，记录并重新抛出以便上层处理
                print(f"Warning: prepare_model_for_kbit_training failed: {e}")
                raise

        return self.model

    def load_dataset(self, dataset_path: str) -> Dataset:
        """加载训练数据集 - 支持新旧两种格式"""
        print(f"Loading dataset from {dataset_path}")

        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 转换为datasets格式
        formatted_data = []
        for item in data:
            # 检查数据集格式
            if 'response' in item and 'instruction' in item:
                # 新格式: instruction + response (评估指标导向数据集)
                text = f"Instruction: {item['instruction']}\nResponse: {item['response']}"

                formatted_data.append({
                    "text": text,
                    "instruction": item["instruction"],
                    "input": "",  # 空输入
                    "output": item["response"]
                })
            elif 'input' in item and 'output' in item:
                # 旧格式: instruction + input + output
                text = f"Instruction: {item['instruction']}\nInput: {item['input']}\nResponse: {item['output']}"

                formatted_data.append({
                    "text": text,
                    "instruction": item["instruction"],
                    "input": item["input"],
                    "output": item["output"]
                })
            else:
                raise ValueError(f"Unsupported dataset format for item: {item.keys()}")

        dataset = Dataset.from_list(formatted_data)
        print(f"Loaded {len(dataset)} training samples")

        return dataset

    def tokenize_function(self, examples):
        """token化函数"""
        # 构建完整文本
        texts = []
        for instruction, input_text, output in zip(
            examples["instruction"],
            examples["input"],
            examples["output"]
        ):
            text = f"Instruction: {instruction}\nInput: {input_text}\nResponse: {output}"
            texts.append(text)

        # token化
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=1024,
            return_tensors=None
        )

        # 为因果语言模型设置 labels（等于 input_ids）
        if "input_ids" in tokenized:
            tokenized["labels"] = [list(ids) for ids in tokenized["input_ids"]]

        return tokenized

    def setup_training_args(self, output_dir: str) -> TrainingArguments:
        """设置训练参数"""
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.lora_config.get("num_epochs", 3),
            per_device_train_batch_size=self.lora_config.get("batch_size", 4),
            per_device_eval_batch_size=self.lora_config.get("batch_size", 4),
            gradient_accumulation_steps=self.lora_config.get("gradient_accumulation_steps", 1),
            optim=self.lora_config.get("optim", "adamw_torch"),
            save_steps=self.lora_config.get("save_steps", 100),
            logging_steps=self.lora_config.get("logging_steps", 10),
            learning_rate=self.lora_config.get("learning_rate", 2e-4),
            fp16=self.lora_config.get("fp16", True),
            bf16=self.lora_config.get("bf16", False),
            max_grad_norm=self.lora_config.get("max_grad_norm", 0.3),
            warmup_ratio=self.lora_config.get("warmup_ratio", 0.03),
            lr_scheduler_type=self.lora_config.get("lr_scheduler_type", "cosine"),
            remove_unused_columns=self.lora_config.get("remove_unused_columns", False),
            evaluation_strategy=self.lora_config.get("evaluation_strategy", "no"),
            save_strategy=self.lora_config.get("save_strategy", "steps"),
            load_best_model_at_end=self.lora_config.get("load_best_model_at_end", False),
            report_to=self.lora_config.get("report_to", "none")
        )

        return training_args

    def train(self, dataset_path: str, output_dir: str, llamole_adapter_path: str = None):
        """执行LoRA微调"""
        # 加载模型和tokenizer
        self.load_model_and_tokenizer(llamole_adapter_path)

        # 准备模型训练
        self.prepare_model_for_training()

        # 加载数据集
        dataset = self.load_dataset(dataset_path)

        # 分割训练集和验证集
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]

        # token化数据集
        tokenized_train = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )

        tokenized_eval = eval_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=eval_dataset.column_names
        )

        # 数据整理器
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            max_length=1024
        )

        # 设置训练参数
        training_args = self.setup_training_args(output_dir)

        # 创建trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            data_collator=data_collator,
        )

        # 开始训练
        print("Starting LoRA fine-tuning...")
        trainer.train()

        # 保存模型
        trainer.save_model(output_dir)
        print(f"Model saved to {output_dir}")

        return trainer

    def load_finetuned_model(self, model_path: str, llamole_adapter_path: str = None):
        """使用原始Llamole方法加载微调后的完整模型

        Args:
            model_path: GBM LoRA适配器路径
            llamole_adapter_path: 预训练的Llamole适配器路径
        """
        print(f"Loading fine-tuned model using original Llamole method from {model_path}")

        # 创建兼容的模型参数 (使用原始Llamole的ModelArguments)
        from src.hparams.model_args import ModelArguments

        # 基础模型参数 - 使用原始Qwen2
        model_args = ModelArguments(
            model_name_or_path=self.base_model_path,
            graph_decoder_path=self.graph_decoder_path if self.use_llamole_graph_models else "",
            graph_encoder_path=self.graph_encoder_path if self.use_llamole_graph_models else "",
            graph_predictor_path=self.graph_predictor_path if self.use_llamole_graph_models else "",
            graph_lm_connector_path=self.graph_lm_connector_path if self.use_llamole_graph_models else "",
            adapter_name_or_path=",".join([llamole_adapter_path, model_path] if llamole_adapter_path else [model_path]),
            flash_attn="disabled",  # 强制使用eager注意力机制
            attn_implementation=None,
            cache_dir=None,
            use_fast_tokenizer=True,
            disable_graph_model_gradient=True,
            resize_vocab=False,
            split_special_tokens=False,
            model_revision="main",
            low_cpu_mem_usage=True,
            quantization_method="bitsandbytes",
            quantization_bit=None,
            quantization_type="nf4",
            double_quantization=True,
            rope_scaling=None,
            shift_attn=False,
            mixture_of_depths=None,
            use_unsloth=False,
            new_special_tokens=",".join(["<design_start>", "<design_end>", "<design_body>", "<molecule>", "<retro_start>", "<retro_end>", "<retro_body>", "<rollback_start>", "<rollback_end>"]) if self.use_llamole_graph_models else None
        )

        # 数据参数 (最小化配置)
        from types import SimpleNamespace
        data_args = SimpleNamespace()
        data_args.learned_query_size = 8
        data_args.ignore_pad_token_for_loss = True

        # 训练参数 (推理模式)
        training_args = SimpleNamespace()
        training_args.do_train = False
        training_args.generation_max_length = 2048
        training_args.generation_num_beams = 1

        # 微调参数
        finetuning_args = SimpleNamespace()
        finetuning_args.loss_weight_lm = 1.0
        finetuning_args.loss_weight_design = 1.0
        finetuning_args.loss_weight_retro = 1.0

        # 加载tokenizer
        from src.model.loader import load_tokenizer
        tokenizer_module = load_tokenizer(model_args)
        tokenizer = tokenizer_module["tokenizer"]

        # 使用原始Llamole的from_pretrained方法加载完整模型
        try:
            from src.model.modeling_llamole import GraphLLMForCausalMLM
            print("Using original Llamole GraphLLMForCausalMLM.from_pretrained() method...")

            model = GraphLLMForCausalMLM.from_pretrained(
                tokenizer,
                model_args,
                data_args,
                training_args,
                finetuning_args,
                load_adapter=True  # 加载适配器
            )

            print("✓ Complete Llamole model loaded using original method")

        except Exception as e:
            print(f"❌ Original Llamole method failed: {e}")
            print("Falling back to manual assembly...")

            # 回退到手动组装方法
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True
            )

            # 加载适配器
            if llamole_adapter_path:
                print(f"Loading Llamole adapter: {llamole_adapter_path}")
                base_model = PeftModel.from_pretrained(base_model, llamole_adapter_path)

            print(f"Loading GBM LoRA adapter: {model_path}")
            model = PeftModel.from_pretrained(base_model, model_path)

            # 如果启用Llamole图模型，加载并集成它们
            if self.use_llamole_graph_models and LLAMOLE_AVAILABLE:
                print("Loading Llamole graph models for inference...")
                self.load_llamole_graph_models()
                if self.graph_decoder and self.graph_encoder and self.graph_predictor:
                    self.load_model_and_tokenizer()  # 确保tokenizer已加载
                    self.setup_llamole_model()
                    model = self.model  # 使用完整的Llamole模型
                    print("✓ Complete Llamole model loaded (fallback method)")
                else:
                    print("⚠ Graph models failed to load, using language model only")

        return model, tokenizer

    def generate_with_finetuned_model(self, model, tokenizer, instruction: str, input_text: str = "") -> str:
        """使用微调模型生成响应"""
        prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse:"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取Response部分
        if "Response:" in response:
            response = response.split("Response:")[-1].strip()

        return response


def run_gbm_lora_finetuning(
    base_model_path: str,
    dataset_path: str,
    output_dir: str,
    llamole_adapter_path: str = None,
    lora_config: Optional[Dict[str, Any]] = None
):
    """运行GBM LoRA微调的主函数

    Args:
        base_model_path: 基础模型路径 (Qwen2-7B-Instruct)
        llamole_adapter_path: 预训练的Llamole适配器路径
        dataset_path: 训练数据集路径
        output_dir: 输出目录
        lora_config: LoRA配置
    """

    # 默认LoRA配置
    if lora_config is None:
        lora_config = {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "num_epochs": 3,
            "batch_size": 2,  # 小批量以适应内存
            "learning_rate": 2e-4,
            "gradient_accumulation_steps": 4,
            "save_steps": 50,
            "logging_steps": 10,
            "fp16": True
        }

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 初始化微调器
    finetuner = GBMLoRAFinetuner(base_model_path, lora_config)

    # 执行训练
    trainer = finetuner.train(dataset_path, output_dir, llamole_adapter_path)

    print("GBM LoRA fine-tuning completed!")

    return finetuner


if __name__ == "__main__":
    # 示例用法
    base_model_path = "../../saves/Llamole-Qwen2-7B-Instruct-Adapter"
    dataset_path = "../data/lora_datasets/gbm_connectivity_english.json"
    output_dir = "../../saves/GBM_LoRA_Connectivity"

    # 运行微调
    finetuner = run_gbm_lora_finetuning(
        base_model_path=base_model_path,
        dataset_path=dataset_path,
        output_dir=output_dir
    )

    # 测试微调效果
    print("\nTesting fine-tuned model...")
    model, tokenizer = finetuner.load_finetuned_model(output_dir)

    test_instruction = "Design a GBM therapeutic molecule targeting Mesenchymal-like GBM cells phenotype."
    test_input = "Drug: afatinib, Cell State: MES, Score: 0.750"

    response = finetuner.generate_with_finetuned_model(
        model, tokenizer, test_instruction, test_input
    )

    print(f"Test Instruction: {test_instruction}")
    print(f"Test Input: {test_input}")
    print(f"Generated Response: {response}")
