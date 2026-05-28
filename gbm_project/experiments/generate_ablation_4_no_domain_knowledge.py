#!/usr/bin/env python3
"""
Ablation 4: Remove GBM Knowledge Base / Domain Prompt Injection
==============================================================
消融实验 4：去掉 GBM 知识库注入，即不调用 GBMKnowledgeBase 和 GBMPromptGenerator
生成领域专业 prompt，只用通用 prompt。

基线行为（ToT）：
  - load_prompt_generator() 创建 GBMKnowledgeBase + GBMPromptGenerator
  - generate_domain_prompt(target_name, constraints) 从知识库读取：
      (1) 靶点描述、突变类型、现有药物、临床挑战
      (2) 临床洞察（成功模式、失败教训）
      (3) 结构知识注入（StructuralKnowledgeInjector）
  - build_tot_propose_prompt() 将领域 prompt 注入到 Scaffold/Assembly/SMILES 模板

本实验行为：
  - 不加载 GBMKnowledgeBase 和 GBMPromptGenerator
  - 使用一个通用的最小化 prompt（只包含分子设计基本约束，不含靶点知识）
  - 即：去掉靶点知识、临床挑战、结构先验的注入

实验目标：验证靶点知识、临床洞察、结构先验是否真的帮助模型
          生成更符合 GBM 任务需求的候选分子。
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


# ── 通用最小化 Prompt（不含 GBM 领域知识）──────────────────────────────────

GENERIC_DOMAIN_PROMPT = """You are a medicinal chemist designing drug candidates.

Target: {target_name}

Design Requirements:
- Molecular Weight (MW): 300-500 Da
- Blood-Brain Barrier (BBB) penetration potential: HIGH
- LogP: 2.0-4.0
- Topological Polar Surface Area (TPSA): 40-120 Å²
- Chemically valid and synthesizable

Output your candidates as SMILES strings in the format:
SMILES 1: <smiles>
SMILES 2: <smiles>
SMILES 3: <smiles>
"""


class NoDomainKnowledgeGenerator:
    """
    ToT 生成器包装类：禁用 GBM 知识库和领域 prompt 注入。

    将 prompt_generator 替换为一个轻量对象，该对象的 generate_domain_prompt()
    只返回通用最小化约束，不含任何靶点知识。
    """

    def __init__(self, tot_generator: TreeOfThoughtsGenerator):
        self.tot = tot_generator
        # 替换 prompt_generator
        self.tot.prompt_generator = GenericPromptOnly(self.tot.constraints)
        logger.info("[Ablation 4] Domain knowledge injection DISABLED (using generic prompts)")

    def generate(self, target_name: str) -> List[Dict[str, Any]]:
        """Generate molecules with only generic prompts, no domain knowledge."""
        return self.tot.generate_molecules(target_name)


class GenericPromptOnly:
    """
    轻量 prompt 生成器：只返回通用分子设计约束，不含任何 GBM 靶点知识。
    """

    def __init__(self, constraints: Dict[str, Any]):
        self.constraints = constraints

    def generate_domain_prompt(self, target_name: str, constraints=None) -> str:
        return GENERIC_DOMAIN_PROMPT.format(target_name=target_name)

    def build_tot_propose_prompt(self, domain_prompt: str, current_state: Dict, step_type: str) -> str:
        """Build a generic step-specific prompt without domain knowledge."""
        if step_type == "scaffold":
            return (
                f"{domain_prompt}\n\n"
                "Step: Propose 3 heterocyclic scaffolds suitable for BBB-penetrating drugs.\n"
                "Requirements:\n"
                "  - MW: 100-250 Da\n"
                "  - Known BBB-friendly scaffolds\n"
                "  - Easy to functionalize\n\n"
                "Output format:\n"
                "Scaffold 1:\n"
                "Name: <scaffold_name>\n"
                "Rationale: <one-sentence>\n"
                "Base MW: <N> Da\n"
                "BBB Potential: <high/medium/low>\n\n"
                "Scaffold 2: ...\n"
                "Scaffold 3: ..."
            )
        elif step_type == "assembly":
            scaffold = current_state.get("selected_scaffold", "heterocycle")
            return (
                f"{domain_prompt}\n\n"
                f"Selected scaffold: {scaffold}\n\n"
                "Step: Design molecular assembly with warhead and BBB enhancers.\n"
                "  - Warhead: electrophilic group for target binding\n"
                "  - BBB enhancers: lipophilic groups (F, OCH3, CF3)\n"
                "  - Keep total MW 300-500 Da\n\n"
                "Output format:\n"
                "Strategy 1:\n"
                "Warhead: <warhead>\n"
                "BBB Enhancers: <enhancers>\n"
                "Estimated Total MW: <N> Da\n"
                "Expected LogP: <value>\n"
                "Expected TPSA: <value> Å²\n"
                "Rationale: <one-sentence>\n\n"
                "Strategy 2: ...\n"
                "Strategy 3: ..."
            )
        elif step_type == "smiles":
            scaffold = current_state.get("selected_scaffold", "heterocycle")
            warhead = current_state.get("warhead_type", "electrophile")
            enhancers = current_state.get("bbb_enhancers", "lipophilic groups")
            return (
                f"{domain_prompt}\n\n"
                f"Scaffold: {scaffold}\n"
                f"Warhead: {warhead}\n"
                f"BBB Enhancers: {enhancers}\n\n"
                "Step: Generate the final SMILES string.\n"
                "Requirements:\n"
                "  - TPSA: 40-120 Å²\n"
                "  - MW: 300-500 Da\n"
                "  - LogP: 2.0-4.0\n\n"
                "Output format:\n"
                "SMILES 1: <valid_smiles>\n"
                "SMILES 2: <valid_smiles>\n"
                "SMILES 3: <valid_smiles>"
            )
        return domain_prompt

    def build_tot_propose_prompt_with_feedback(
        self, domain_prompt: str, current_state: Dict,
        step_type: str, physical_feedback=None
    ) -> str:
        base = self.build_tot_propose_prompt(domain_prompt, current_state, step_type)
        if physical_feedback:
            base += f"\n\nPhysical feedback from previous round: {physical_feedback}"
        return base


class Ablation4NoDomainKnowledge(AblationBase):
    """Ablation 4: Remove GBM knowledge base / domain prompt injection."""

    experiment_name = "ablation_4_no_domain_knowledge"
    ablation_description = (
        "Ablation 4: GBM knowledge base and domain prompt injection removed. "
        "The generator uses only a generic molecular design prompt with basic "
        "constraints (MW, LogP, TPSA, BBB). No target-specific knowledge, "
        "clinical insights, or structural priors are injected. This tests "
        "whether the domain-specific prompt engineering genuinely helps."
    )

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._generator_instance = None

    def _create_generator(self) -> TreeOfThoughtsGenerator:
        self._generator_instance = TreeOfThoughtsGenerator(self.config)
        return self._generator_instance

    def _run_single_attempt(self, generator: TreeOfThoughtsGenerator, target_name: str) -> List[Dict[str, Any]]:
        gen = NoDomainKnowledgeGenerator(generator)
        return gen.generate(target_name)


def main():
    parser = argparse.ArgumentParser(
        description="Ablation 4: Remove GBM knowledge base / domain prompt injection"
    )
    add_common_args(parser)
    args = parser.parse_args()

    config = build_base_config(args)

    Ablation4NoDomainKnowledge(config).run()
    logger.info("\nAblation 4 complete!")


if __name__ == "__main__":
    main()
