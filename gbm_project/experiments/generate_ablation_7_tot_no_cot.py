#!/usr/bin/env python3
"""
Ablation 7: Direct Simple Prompt (No CoT, No ToT)
=================================================
消融实验7：禁用CoT推理模板 + 禁用ToT分层搜索，直接用简单prompt生成分子。

Full Model: ToT三层搜索 + CoT step_by_step_design推理链 + 领域知识
Ablation 7: 纯直接生成，无任何中间推理步骤

实验目标：验证CoT+ToT组合对分子质量的边际贡献。
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

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="peft")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


SIMPLE_PROMPTS = {
    "EGFR": (
        "You are a medicinal chemist designing GBM (Glioblastoma) drug candidates targeting EGFR.\n\n"
        "Design a novel, drug-like small molecule targeting EGFR for GBM treatment.\n\n"
        "Requirements:\n"
        "- Molecular Weight (MW): 300-500 Da\n"
        "- Blood-Brain Barrier (BBB) penetration: HIGH\n"
        "- LogP: 2.0-4.0\n"
        "- Topological Polar Surface Area (TPSA): 40-120 A^2\n"
        "- Must be chemically valid and synthesizable\n\n"
        "Output format (IMPORTANT - follow EXACTLY):\n"
        "SMILES 1: <your_smiles_here>\n\n"
        "Generate exactly 3 different candidate molecules.\n"
        "Output only the SMILES strings in the format shown above, nothing else."
    ),
    "IDH1_IDH2": (
        "You are a medicinal chemist designing GBM (Glioblastoma) drug candidates targeting IDH1/IDH2.\n\n"
        "Design a novel, drug-like small molecule targeting IDH1 or IDH2 for GBM treatment.\n\n"
        "Requirements:\n"
        "- Molecular Weight (MW): 300-500 Da\n"
        "- Blood-Brain Barrier (BBB) penetration: HIGH\n"
        "- LogP: 2.0-4.0\n"
        "- Topological Polar Surface Area (TPSA): 40-120 A^2\n"
        "- Must be chemically valid and synthesizable\n\n"
        "Output format (IMPORTANT - follow EXACTLY):\n"
        "SMILES 1: <your_smiles_here>\n\n"
        "Generate exactly 3 different candidate molecules.\n"
        "Output only the SMILES strings in the format shown above, nothing else."
    ),
    "VEGF_VEGFR": (
        "You are a medicinal chemist designing GBM (Glioblastoma) drug candidates targeting VEGF/VEGFR.\n\n"
        "Design a novel, drug-like small molecule targeting VEGF or VEGFR for GBM treatment.\n\n"
        "Requirements:\n"
        "- Molecular Weight (MW): 300-500 Da\n"
        "- Blood-Brain Barrier (BBB) penetration: HIGH\n"
        "- LogP: 2.0-4.0\n"
        "- Topological Polar Surface Area (TPSA): 40-120 A^2\n"
        "- Must be chemically valid and synthesizable\n\n"
        "Output format (IMPORTANT - follow EXACTLY):\n"
        "SMILES 1: <your_smiles_here>\n\n"
        "Generate exactly 3 different candidate molecules.\n"
        "Output only the SMILES strings in the format shown above, nothing else."
    ),
}


class DirectSimpleGenerator:
    """
    直接简单prompt生成：无CoT模板、无ToT框架、无领域知识注入。
    仅通过一条简洁的指令prompt引导模型输出SMILES。
    """

    def __init__(self, tot_generator: TreeOfThoughtsGenerator):
        self.tot = tot_generator

    def generate(self, target_name: str) -> List[Dict[str, Any]]:
        prompt = SIMPLE_PROMPTS.get(target_name)
        if not prompt:
            logger.warning(f"No simple prompt for {target_name}")
            return []

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
                    "generation_method": "direct_simple_prompt",
                    "raw_response": response[:500],
                    "cot_chain": "",
                    "tot_path": [{"level": 0, "content": "direct_simple", "evaluation": "N/A"}],
                    "physical_evaluation": {},
                    "physical_feedback": "",
                })

        logger.info(f"[Direct Simple] {target_name}: {len(molecules)} valid molecules from {len(smiles_list)} candidates")
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


class Ablation7DirectSimple(AblationBase):
    experiment_name = "ablation_7_direct_simple"
    ablation_description = (
        "Ablation 7: Direct simple prompt generation. "
        "No CoT step_by_step_design template, no ToT hierarchical search. "
        "Tests the raw model capability with minimal prompting guidance."
    )

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._generator_instance = None
        self._wrapper_instance = None

    def _create_generator(self) -> TreeOfThoughtsGenerator:
        self._generator_instance = TreeOfThoughtsGenerator(self.config)
        return self._generator_instance

    def _run_single_attempt(
        self, generator: TreeOfThoughtsGenerator, target_name: str
    ) -> List[Dict[str, Any]]:
        if self._wrapper_instance is None:
            self._wrapper_instance = DirectSimpleGenerator(generator)
        return self._wrapper_instance.generate(target_name)


def main():
    parser = argparse.ArgumentParser(
        description="Ablation 7: Direct simple prompt (no CoT, no ToT)"
    )
    add_common_args(parser)
    args = parser.parse_args()

    args.num_molecules = 50
    args.max_attempts = 200
    args.max_no_new_streak = 20
    args.output_dir = None

    config = build_base_config(args)
    Ablation7DirectSimple(config).run()
    logger.info("\nAblation 7 complete!")


if __name__ == "__main__":
    main()
