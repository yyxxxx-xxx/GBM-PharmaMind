#!/usr/bin/env python3
"""
Ablation 2: Remove Physical Feedback Iteration (Refinement Loop)
================================================================
消融实验 2：去掉 ToT 的多轮物理反馈迭代（refinement loop）。

基线行为（ToT）：
  - 每个分支执行 tot_refinement_rounds 轮（第 1 轮生成 SMILES，第 2+ 轮注入
    上一轮最佳分子的 Vina/DILI/BBB 评分作为反馈，重新生成）
  - 反馈通过 build_tot_propose_prompt_with_feedback() 注入 prompt
  - 反馈文本由 _build_physical_feedback_text() 从 PhysicalEvaluationResult 构建

本实验行为：
  - 只执行 1 轮生成（tot_refinement_rounds = 1，禁用多轮反馈）
  - 保留物理评估（用于后验记录），但不将结果注入到下一轮 prompt
  - 即：去掉了 feedback loop / iterative refinement

实验目标：验证 docking、BBB、DILI 等物理评估结果回注到下一轮生成，
          是否能显著提高 reward 和候选质量。
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

from generate_tot_molecules import TreeOfThoughtsGenerator
from ablation_base import AblationBase, build_base_config, add_common_args, logger

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="peft")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


class NoFeedbackGenerator:
    """
    ToT 生成器包装类：禁用多轮物理反馈迭代。

    实现方式：将 tot_refinement_rounds 强制设为 1，
    并且将 _generate_smiles_fallback 中的 physical_feedback 永远设为 None，
    从而切断 feedback loop。
    """

    def __init__(self, tot_generator: TreeOfThoughtsGenerator):
        self.tot = tot_generator
        # 强制禁用多轮反馈迭代
        self.tot.tot_refinement_rounds = 1
        # 覆盖 _generate_smiles_fallback 的反馈注入行为（通过 monkey-patch）
        self._patch_generate_smiles()

    def _patch_generate_smiles(self):
        """Monkey-patch _generate_smiles_fallback to always pass physical_feedback=None."""
        original = self.tot._generate_smiles_fallback

        def patched_generate_smiles(
            finetuned_prompt, *, scaffold, assembly, warhead,
            bbb_enhancers, target_mw, physical_feedback=None, max_new_tokens=450
        ):
            # 强制禁用物理反馈
            return original(
                finetuned_prompt,
                scaffold=scaffold, assembly=assembly, warhead=warhead,
                bbb_enhancers=bbb_enhancers, target_mw=target_mw,
                physical_feedback=None,   # <-- 消融：永远不注入反馈
                max_new_tokens=max_new_tokens
            )

        self.tot._generate_smiles_fallback = patched_generate_smiles
        logger.info("[Ablation 2] Physical feedback loop DISABLED (always None)")

    def generate(self, target_name: str) -> List[Dict[str, Any]]:
        """Generate molecules without physical feedback iteration."""
        return self.tot.generate_molecules(target_name)


class Ablation2NoFeedback(AblationBase):
    """Ablation 2: Remove physical feedback iteration / refinement loop."""

    experiment_name = "ablation_2_no_feedback"
    ablation_description = (
        "Ablation 2: Physical feedback iteration removed. ToT generates SMILES for only "
        "1 round per branch, with NO feedback from Vina/DILI/BBB scores injected into "
        "the next round. This tests whether the iterative refinement loop genuinely "
        "improves molecular quality."
    )

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._generator_instance = None

    def _create_generator(self) -> TreeOfThoughtsGenerator:
        self._generator_instance = TreeOfThoughtsGenerator(self.config)
        return self._generator_instance

    def _run_single_attempt(self, generator: TreeOfThoughtsGenerator, target_name: str) -> List[Dict[str, Any]]:
        gen = NoFeedbackGenerator(generator)
        return gen.generate(target_name)


def main():
    parser = argparse.ArgumentParser(
        description="Ablation 2: Remove physical feedback iteration (refinement loop)"
    )
    add_common_args(parser)
    args = parser.parse_args()

    config = build_base_config(args)
    config['tot_refinement_rounds'] = 1  # 确保只有1轮

    Ablation2NoFeedback(config).run()
    logger.info("\nAblation 2 complete!")


if __name__ == "__main__":
    main()
