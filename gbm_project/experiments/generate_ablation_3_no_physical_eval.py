#!/usr/bin/env python3
"""
Ablation 3: Remove Physical Evaluation Module
==============================================
消融实验 3：去掉物理评估模块，只保留语言模型生成 + 基础 RDKit 过滤。

基线行为（ToT）：
  - 每个生成的 SMILES 都经过 GBMPhysicalEvaluator.evaluate() 评估
  - 评估包括：Vina 对接分数、DILI 风险、BBB 渗透性、hERG 风险
  - 评估结果用于：
      (1) ToT 节点评估（sure/likely/impossible）
      (2) Hard pruning（is_pruned=True 的节点被剪枝）
      (3) Reward 计算
      (4) 反馈文本注入（_build_physical_feedback_text）

本实验行为：
  - 保留 GBMPhysicalEvaluator 的初始化（不报错），但 evaluate() 永远返回
    一个 "fake" 评估结果（verdict=likely, reward=1.0, is_pruned=False）
  - 这样 ToT 的节点评估和 BFS 搜索仍然走通，但所有物理约束都不生效
  - 等效于：只做 RDKit TPSA 格式校验，完全去掉 Vina/ADMET/MPO reward

实验目标：验证 Vina、ADMET、MPO reward、hard prune 这些物理评价约束
          对最终分子可行性和药物性的贡献。
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "gbm_project" / "scripts"))

from generate_tot_molecules import TreeOfThoughtsGenerator
from ablation_base import AblationBase, build_base_config, add_common_args, logger
from gbm_project.src.gbm_physical_evaluator import (
    GBMPhysicalEvaluator, PhysicalEvaluationResult, EvaluationVerdict
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="peft")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


class FakePhysicalEvaluator:
    """
    假的物理评估器：所有评估都返回 'likely' verdict，无硬截断。

    分子仍然会通过 RDKit 的 TPSA 格式校验（generate_tot_molecules 内部执行），
    但 Vina/DILI/BBB/hERG/MPO reward 全都失效。
    """

    def __init__(self, real_evaluator: Optional[GBMPhysicalEvaluator]):
        self.real_evaluator = real_evaluator

    def evaluate(self, smiles: str) -> PhysicalEvaluationResult:
        """Return a fake evaluation that always passes."""
        # 返回一个全通过的结果
        result = PhysicalEvaluationResult(
            smiles=smiles,
            vina_score=-7.5,
            dili_prob=0.1,
            herg_prob=0.1,
            bbb_score=0.8,
            rd_tpsa=70.0,
            rd_mw=400.0,
            rd_logp=3.0,
            rd_hbd=2,
            rd_hba=3,
            rd_n_rotatable=4,
            rd_n_rings=2,
            vina_norm=0.7,
            dili_norm=0.9,
            herg_norm=0.9,
            bbb_norm=0.8,
            tpsa_norm=0.8,
            mw_norm=0.8,
            dili_alert_matches=[],
            herg_alert_matches=[],
            reward=1.0,
            verdict=EvaluationVerdict.LIKELY,
            is_pruned=False,
            prune_reason="",
            vina_error=None,
        )
        return result


class PhysicalEvalFreeGenerator:
    """
    ToT 生成器包装类：用 FakePhysicalEvaluator 替换真实的物理评估器。
    """

    def __init__(self, tot_generator: TreeOfThoughtsGenerator):
        self.tot = tot_generator
        # 保存真实评估器并替换为假评估器
        self._real_evaluator = self.tot.physical_evaluator
        self.tot.physical_evaluator = FakePhysicalEvaluator(self._real_evaluator)
        logger.info(
            "[Ablation 3] Physical evaluator REPLACED with fake (always returns LIKELY, no pruning)"
        )

    def generate(self, target_name: str) -> List[Dict[str, Any]]:
        """Generate molecules without real physical evaluation."""
        return self.tot.generate_molecules(target_name)


class Ablation3NoPhysicalEval(AblationBase):
    """Ablation 3: Remove physical evaluation module."""

    experiment_name = "ablation_3_no_physical_eval"
    ablation_description = (
        "Ablation 3: Physical evaluation module replaced with a fake evaluator "
        "that always returns 'likely' verdict with reward=1.0 and is_pruned=False. "
        "This removes Vina docking, DILI, BBB, hERG, and MPO reward signals from "
        "both the ToT search and the feedback loop. Only RDKit TPSA/MW validation "
        "remains."
    )

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._generator_instance = None

    def _create_generator(self) -> TreeOfThoughtsGenerator:
        self._generator_instance = TreeOfThoughtsGenerator(self.config)
        return self._generator_instance

    def _run_single_attempt(self, generator: TreeOfThoughtsGenerator, target_name: str) -> List[Dict[str, Any]]:
        gen = PhysicalEvalFreeGenerator(generator)
        return gen.generate(target_name)


def main():
    parser = argparse.ArgumentParser(
        description="Ablation 3: Remove physical evaluation module"
    )
    add_common_args(parser)
    args = parser.parse_args()

    config = build_base_config(args)

    Ablation3NoPhysicalEval(config).run()
    logger.info("\nAblation 3 complete!")


if __name__ == "__main__":
    main()
