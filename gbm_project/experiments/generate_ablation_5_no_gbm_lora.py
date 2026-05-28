#!/usr/bin/env python3
"""
Ablation 5: Remove GBM-LoRA Adapter (use base + Llamole-Adapter only)
=====================================================================
消融实验 5：去掉 GBM 领域适配器（第二个 LoRA），只用基座模型 + Llamole 通用适配器。

基线行为（ToT）：
  load_models() 加载顺序：
    1. Qwen2-7B-Instruct 基座模型
    2. Llamole-Qwen2-7B-Instruct-Adapter（通用化学适配器）→ merge_and_unload()
    3. Llamole-Qwen2-7B-Instruct-Adapter-gbm-with-graph-models（GBM 领域适配器）
       → 作为 PeftModel 不合并，直接使用

本实验行为：
  - 只加载步骤 1 + 2（基座 + Llamole 通用适配器）
  - 不加载步骤 3（GBM 领域适配器）
  - 即：只用通用化学能力，不使用 GBM 特异性微调

实验目标：验证领域微调是否带来了 GBM 特异性收益，
          而不只是通用分子生成能力。
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "gbm_project" / "scripts"))

from ablation_base import AblationBase, build_base_config, add_common_args, logger

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="peft")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


class NoGBMLoRAConfig(Dict):
    """Config variant that disables GBM adapter loading."""
    pass


class Ablation5NoGBMLoRA(AblationBase):
    """
    Ablation 5: Remove GBM-LoRA adapter (use base + Llamole-Adapter only).

    This experiment uses the base config but sets gbm_adapter_path to None,
    so load_models() only merges Llamole's general chemistry adapter.
    """

    experiment_name = "ablation_5_no_gbm_lora"
    ablation_description = (
        "Ablation 5: GBM-specific LoRA adapter removed. Only the base Qwen2-7B-Instruct "
        "model and the general-purpose Llamole chemistry adapter are loaded and merged. "
        "The GBM domain adapter (Llamole-Qwen2-7B-Instruct-Adapter-gbm-with-graph-models) "
        "is NOT loaded. This tests whether the domain fine-tuning on GBM tasks provides "
        "specific benefits beyond general molecular generation capability."
    )

    def __init__(self, config: Dict[str, Any]):
        # 设置 GBM adapter 路径为空，禁用 GBM 适配器
        config = dict(config)
        config['gbm_adapter_path'] = ""  # 空路径表示不使用 GBM adapter
        super().__init__(config)
        self._generator_instance = None

    def _create_generator(self):
        from generate_tot_molecules import TreeOfThoughtsGenerator
        self._generator_instance = TreeOfThoughtsGenerator(self.config)
        return self._generator_instance

    def _run_single_attempt(self, generator, target_name):
        return generator.generate_molecules(target_name)


def main():
    parser = argparse.ArgumentParser(
        description="Ablation 5: Remove GBM-LoRA adapter (use base + Llamole-Adapter only)"
    )
    add_common_args(parser)
    args = parser.parse_args()

    config = build_base_config(args)

    Ablation5NoGBMLoRA(config).run()
    logger.info("\nAblation 5 complete!")


if __name__ == "__main__":
    main()
