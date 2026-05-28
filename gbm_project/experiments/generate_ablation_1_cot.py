#!/usr/bin/env python3
"""
Ablation 1: CoT Generation (Single-Step & Chain-of-Thought)
==========================================================
消融实验 1：去掉 ToT 的分层搜索结构，对比：
  - Ablation 1a (CoT-1): 单步生成（one-shot），prompt 直接要求输出 SMILES
  - Ablation 1b (CoT-2): Chain-of-Thought，使用 GBMPromptGenerator 的
    generate_cot_prompt() 生成带完整领域知识的 CoT 推理 prompt

对比基线（ToT）：通过 Scaffold -> Assembly -> SMILES 三层 BFS 搜索 + 评估过滤

实验目标：验证 ToT 的分层搜索（Scaffold -> Assembly -> SMILES）
          是否真正提升了分子质量与生成稳定性。

CoT 配置参考 COT_MECHANISM.md：
  - GBMPromptGenerator.generate_cot_prompt() 提供：
      (1) GBM 领域背景 + 靶点知识 + 临床洞察 + 结构知识注入
      (2) cot_reasoning_templates.step_by_step_design（7 步市售药物参考 + 分步设计 SOP）
      (3) 靶点特定推理（相似分子、耐药机制、选择性要求）
"""

import sys
import re
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import torch
from rdkit import Chem
from rdkit.Chem import Descriptors

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "gbm_project" / "scripts"))

from generate_tot_molecules import TreeOfThoughtsGenerator
from ablation_base import AblationBase, build_base_config, add_common_args, logger

# Silencing verbose warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="peft")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


# ─────────────────────────────────────────────────────────────────────────────
# Ablation 1a: Single-Step (One-Shot) — uses GBMPromptGenerator for fair baseline
# ─────────────────────────────────────────────────────────────────────────────

class SingleStepGenerator:
    """
    单步生成（One-Shot）：直接要求模型生成 SMILES，
    不经过任何中间推理步骤。

    使用 GBMPromptGenerator.generate_domain_prompt() 提供领域知识上下文，
    但不使用 CoT 模板——这与 ToT 的分层 BFS 结构形成对比。
    """

    def __init__(self, tot_generator: TreeOfThoughtsGenerator):
        self.tot = tot_generator

    def generate(self, target_name: str) -> List[Dict[str, Any]]:
        """Generate molecules using single-step domain prompt (no CoT)."""
        self.tot.load_prompt_generator(target_name)
        domain_prompt = self.tot.prompt_generator.generate_domain_prompt(
            target_name, self.tot.constraints
        )

        # 使用领域 prompt 作为上下文，但直接要求输出 SMILES
        prompt = (
            "You are a medicinal chemist designing GBM (Glioblastoma) drug candidates.\n\n"
            "Design a novel, drug-like small molecule for the target: {target_name}\n\n"
            "Requirements:\n"
            "- Molecular Weight (MW): 300-500 Da\n"
            "- Blood-Brain Barrier (BBB) penetration: HIGH\n"
            "- LogP: 2.0-4.0\n"
            "- Topological Polar Surface Area (TPSA): 40-120 Å²\n"
            "- Must be chemically valid and synthesizable\n\n"
            "Output format (IMPORTANT - follow EXACTLY):\n"
            "SMILES 1: <your_smiles_here>\n\n"
            "Generate exactly 3 different candidate molecules. "
            "Output only the SMILES strings in the format shown above, nothing else.\n\n"
            "--- Domain Context ---\n"
            f"{domain_prompt[:2000]}\n\n"
            "--- Now Generate ---\n"
        ).format(target_name=target_name)

        response = self.tot.generate_with_model(prompt, max_new_tokens=600)

        smiles_list = self._parse_smiles(response)
        molecules = []
        for sm in smiles_list:
            ok, tpsa, mw, err = self._validate(sm)
            if ok:
                try:
                    rd_logp = round(Descriptors.MolLogP(Chem.MolFromSmiles(sm)), 2)
                except Exception:
                    rd_logp = None
                molecules.append({
                    "smiles": sm,
                    "tpsa": round(tpsa, 2),
                    "mw": round(mw, 2),
                    "logp": rd_logp,
                    "target": target_name,
                    "generation_method": "single_step_one_shot",
                    "raw_response": response[:500],
                    "cot_chain": "",
                    "tot_path": [{"level": 0, "content": "single_step", "evaluation": "N/A"}],
                    "physical_evaluation": {},
                    "physical_feedback": "",
                })

        logger.info(f"[Single-Step] Generated {len(molecules)} valid molecules from {len(smiles_list)} candidates")
        return molecules

    def _parse_smiles(self, response: str) -> List[str]:
        pattern = r"SMILES\s+\d+:\s*([^\n]+)"
        matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
        results = []
        for m in matches:
            cleaned = re.sub(r"\s+", "", m.strip()).strip(" ,.;")
            if cleaned and self._basic_validate(cleaned):
                results.append(cleaned)
        if not results:
            pattern2 = r"\b([CNOSPFI][A-Za-z0-9@+\-\[\]\(\)=#%/\.]{10,})\b"
            for match in re.finditer(pattern2, response):
                cand = match.group(1).strip()
                if self._basic_validate(cand):
                    results.append(cand)
        return list(dict.fromkeys(results))[:5]

    def _basic_validate(self, smiles: str) -> bool:
        if len(smiles) < 5 or len(smiles) > 600:
            return False
        if not any(c in smiles for c in "CBNOSPFI"):
            return False
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                return False
            Chem.SanitizeMol(mol, catchErrors=True)
            tpsa = Descriptors.TPSA(mol)
            if not (40 <= tpsa <= 120):
                return False
        except Exception:
            return False
        return True

    def _validate(self, smiles: str):
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                return False, 0.0, 0.0, "parse failed"
            Chem.SanitizeMol(mol, catchErrors=True)
            tpsa = Descriptors.TPSA(mol)
            mw = Descriptors.MolWt(mol)
            if not (40 <= tpsa <= 120):
                return False, tpsa, mw, "TPSA out of range"
            if not (100 <= mw <= 900):
                return False, tpsa, mw, "MW out of range"
            return True, tpsa, mw, ""
        except Exception as e:
            return False, 0.0, 0.0, str(e)


class Ablation1aSingleStep(AblationBase):
    """Ablation 1a: Single-Step (One-Shot) Generation."""

    experiment_name = "ablation_1a_single_step"
    ablation_description = (
        "Ablation 1a: Single-step generation with GBM domain knowledge. "
        "Uses GBMPromptGenerator.generate_domain_prompt() for target-specific context, "
        "but the prompt directly asks model to output SMILES without any intermediate "
        "reasoning steps. Compares against ToT's multi-level Scaffold -> Assembly -> SMILES search."
    )

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._generator_instance = None

    def _create_generator(self) -> TreeOfThoughtsGenerator:
        self._generator_instance = TreeOfThoughtsGenerator(self.config)
        return self._generator_instance

    def _run_single_attempt(self, generator: TreeOfThoughtsGenerator, target_name: str) -> List[Dict[str, Any]]:
        gen = SingleStepGenerator(generator)
        return gen.generate(target_name)


# ─────────────────────────────────────────────────────────────────────────────
# Ablation 1b: Chain-of-Thought — uses GBMPromptGenerator.generate_cot_prompt()
# ─────────────────────────────────────────────────────────────────────────────

class ChainOfThoughtGenerator:
    """
    Chain-of-Thought (CoT)：使用 GBMPromptGenerator.generate_cot_prompt()
    生成包含完整 GBM 领域知识 + CoT 推理模板的 prompt。

    参考 COT_MECHANISM.md：
      - generate_cot_prompt() 提供：
          (1) GBM 背景与生物学特征（generate_domain_prompt）
          (2) cot_reasoning_templates.step_by_step_design（7 步市售药物参考 SOP）
          (3) 靶点特定推理（_generate_target_reasoning）
          (4) 结构知识注入（StructuralKnowledgeInjector）
      - CoT 链：输出时先输出推理步骤，再输出 SMILES

    与 ToT 的区别：不做 Scaffold -> Assembly -> SMILES 的分层 BFS 搜索与分支评估，
    只通过单个 CoT prompt 引导模型按步骤推理。
    """

    def __init__(self, tot_generator: TreeOfThoughtsGenerator):
        self.tot = tot_generator

    def generate(self, target_name: str) -> List[Dict[str, Any]]:
        """Generate molecules using full CoT prompt from GBMPromptGenerator."""
        self.tot.load_prompt_generator(target_name)

        domain_prompt = self.tot.prompt_generator.generate_domain_prompt(
            target_name, self.tot.constraints
        )

        # 使用 generate_cot_prompt 获得完整的 CoT prompt：
        # 包含 (1) 领域背景 (2) step_by_step_design 模板 (3) 靶点特定推理
        cot_prompt = self.tot.prompt_generator.generate_cot_prompt(
            domain_prompt, target_name, reasoning_type="step_by_step_design"
        )

        # 追加最终生成指令（要求先输出推理，再输出 SMILES）
        instruction = (
            "\n\nBased on the market GBM drug analysis and step-by-step reasoning above, "
            "generate 3 novel GBM candidate molecules.\n\n"
            "IMPORTANT: First provide your Chain-of-Thought reasoning for each candidate, "
            "then output the SMILES string. Follow the format:\n\n"
            "Candidate 1:\n"
            "Chain-of-Thought: <your reasoning here>\n"
            "SMILES 1: <smiles_string>\n\n"
            "Candidate 2:\n"
            "Chain-of-Thought: <your reasoning here>\n"
            "SMILES 2: <smiles_string>\n\n"
            "Candidate 3:\n"
            "Chain-of-Thought: <your reasoning here>\n"
            "SMILES 3: <smiles_string>\n\n"
            "Output EXACTLY 3 candidates following this format. "
            "Only output the reasoning and SMILES, nothing else."
        )

        full_prompt = cot_prompt + instruction

        response = self.tot.generate_with_model(full_prompt, max_new_tokens=1000)

        # 提取 CoT 推理链（第一个 SMILES 之前的所有内容）
        cot_chain = self._extract_cot_chain(response)

        smiles_list = self._parse_smiles(response)
        molecules = []
        for sm in smiles_list:
            ok, tpsa, mw, err = self._validate(sm)
            if ok:
                try:
                    rd_logp = round(Descriptors.MolLogP(Chem.MolFromSmiles(sm)), 2)
                except Exception:
                    rd_logp = None
                molecules.append({
                    "smiles": sm,
                    "tpsa": round(tpsa, 2),
                    "mw": round(mw, 2),
                    "logp": rd_logp,
                    "target": target_name,
                    "generation_method": "chain_of_thought",
                    "raw_response": response[:500],
                    "cot_chain": cot_chain,
                    "tot_path": [{"level": 0, "content": "chain_of_thought", "evaluation": "N/A"}],
                    "physical_evaluation": {},
                    "physical_feedback": "",
                })

        logger.info(f"[CoT] Generated {len(molecules)} valid molecules from {len(smiles_list)} candidates")
        return molecules

    def _extract_cot_chain(self, response: str) -> str:
        """
        从模型输出中提取 CoT 推理链。

        策略（参考 COT_MECHANISM.md 第 5.1 节）：
          1. 优先以编号 SMILES 标记位置（\n\s*\d+[\.\)\:]\s*SMILES）为界，
             将其前的所有内容视为 CoT。
          2. 若没有编号 SMILES，尝试以 "Chain-of-Thought" 标记截断。
          3. 若均无匹配，截取前 800 字符作为 CoT。
        """
        # 策略 1：找到第一个 SMILES 编号标记，截取其前的内容
        pattern_smiles_num = r"\n\s*\d+[\.\)\:]\s*SMILES"
        match = re.search(pattern_smiles_num, response, re.IGNORECASE)
        if match:
            return response[:match.start()].strip()

        # 策略 2：找到第一个 SMILES: 标记，截取其前的内容
        pattern_smiles_colon = r"\n\s*SMILES\s*\d*\s*:"
        match = re.search(pattern_smiles_colon, response, re.IGNORECASE)
        if match:
            return response[:match.start()].strip()

        # 策略 3：截取前 800 字符
        return response[:800].strip()

    def _parse_smiles(self, response: str) -> List[str]:
        """从模型输出中提取 SMILES 字符串。"""
        # 匹配带编号的 SMILES 行
        pattern = r"SMILES\s*\d*\s*:\s*([^\n]+)"
        matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
        results = []
        for m in matches:
            cleaned = re.sub(r"\s+", "", m.strip()).strip(" ,.;")
            if cleaned and self._basic_validate(cleaned):
                results.append(cleaned)

        # 回退：通用 SMILES 提取
        if not results:
            pattern2 = r"\b([CNOSPFI][A-Za-z0-9@+\-\[\]\(\)=#%/\.]{10,})\b"
            for match in re.finditer(pattern2, response):
                cand = match.group(1).strip()
                if self._basic_validate(cand):
                    results.append(cand)

        return list(dict.fromkeys(results))[:5]

    def _basic_validate(self, smiles: str) -> bool:
        if len(smiles) < 5 or len(smiles) > 600:
            return False
        if not any(c in smiles for c in "CBNOSPFI"):
            return False
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                return False
            Chem.SanitizeMol(mol, catchErrors=True)
            tpsa = Descriptors.TPSA(mol)
            if not (40 <= tpsa <= 120):
                return False
        except Exception:
            return False
        return True

    def _validate(self, smiles: str):
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                return False, 0.0, 0.0, "parse failed"
            Chem.SanitizeMol(mol, catchErrors=True)
            tpsa = Descriptors.TPSA(mol)
            mw = Descriptors.MolWt(mol)
            if not (40 <= tpsa <= 120):
                return False, tpsa, mw, "TPSA out of range"
            if not (100 <= mw <= 900):
                return False, tpsa, mw, "MW out of range"
            return True, tpsa, mw, ""
        except Exception as e:
            return False, 0.0, 0.0, str(e)


class Ablation1bChainOfThought(AblationBase):
    """Ablation 1b: Chain-of-Thought Generation using GBMPromptGenerator."""

    experiment_name = "ablation_1b_chain_of_thought"
    ablation_description = (
        "Ablation 1b: Chain-of-Thought generation using GBMPromptGenerator.generate_cot_prompt(). "
        "The full CoT prompt includes: (1) GBM domain knowledge + target-specific context "
        "(from generate_domain_prompt), (2) cot_reasoning_templates.step_by_step_design "
        "7-step SOP with market drug references, and (3) target-specific reasoning "
        "(from _generate_target_reasoning). "
        "The model is guided step-by-step but WITHOUT ToT's hierarchical BFS branching "
        "and state evaluation/pruning. This tests whether the hierarchical search structure "
        "adds value beyond good CoT prompting."
    )

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._generator_instance = None

    def _create_generator(self) -> TreeOfThoughtsGenerator:
        self._generator_instance = TreeOfThoughtsGenerator(self.config)
        return self._generator_instance

    def _run_single_attempt(self, generator: TreeOfThoughtsGenerator, target_name: str) -> List[Dict[str, Any]]:
        gen = ChainOfThoughtGenerator(generator)
        return gen.generate(target_name)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ablation 1: CoT vs ToT comparison (single-step and chain-of-thought)"
    )
    parser.add_argument(
        '--mode', type=str, choices=['single_step', 'cot', 'both'], default='both',
        help='single_step: ablation 1a (one-shot). cot: ablation 1b (chain-of-thought). both: run both.'
    )
    add_common_args(parser)
    args = parser.parse_args()

    config = build_base_config(args)

    if args.mode in ('single_step', 'both'):
        logger.info("=" * 60)
        logger.info("Running Ablation 1a: Single-Step (One-Shot) Generation")
        logger.info("=" * 60)
        Ablation1aSingleStep(config).run()

    if args.mode in ('cot', 'both'):
        logger.info("=" * 60)
        logger.info("Running Ablation 1b: Chain-of-Thought Generation")
        logger.info("=" * 60)
        Ablation1bChainOfThought(config).run()

    logger.info("\nAblation 1 complete!")


if __name__ == "__main__":
    main()
