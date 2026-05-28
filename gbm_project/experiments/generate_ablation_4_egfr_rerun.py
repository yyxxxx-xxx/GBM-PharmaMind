#!/usr/bin/env python3
"""
Ablation 4 EGFR Re-run: Remove GBM Knowledge Base / Domain Prompt Injection
==========================================================================

仅针对 EGFR 靶点重新运行消融实验4。
"""

import sys
import argparse
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


GENERIC_DOMAIN_PROMPT = """You are a medicinal chemist designing drug candidates.

Target: {target_name}

Design Requirements:
- Molecular Weight (MW): 300-500 Da
- Blood-Brain Barrier (BBB) penetration potential: HIGH
- LogP: 2.0-4.0
- Topological Polar Surface Area (TPSA): 40-120 A2
- Chemically valid and synthesizable

Output your candidates as SMILES strings in the format:
SMILES 1: <smiles>
SMILES 2: <smiles>
SMILES 3: <smiles>
"""


class NoDomainKnowledgeGenerator:
    def __init__(self, tot_generator: TreeOfThoughtsGenerator):
        self.tot = tot_generator
        self.tot.prompt_generator = GenericPromptOnly(self.tot.constraints)
        logger.info("[Ablation 4 EGFR] Domain knowledge injection DISABLED")

    def generate(self, target_name: str) -> List[Dict[str, Any]]:
        return self.tot.generate_molecules(target_name)


class GenericPromptOnly:
    def __init__(self, constraints: Dict[str, Any]):
        self.constraints = constraints

    def generate_domain_prompt(self, target_name: str, constraints=None) -> str:
        return GENERIC_DOMAIN_PROMPT.format(target_name=target_name)

    def build_tot_propose_prompt(self, domain_prompt: str, current_state: Dict, step_type: str) -> str:
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
                "Expected TPSA: <value> A2\n"
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
                "  - TPSA: 40-120 A2\n"
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


class Ablation4NoDomainKnowledgeEGFR(AblationBase):
    experiment_name = "ablation_4_no_domain_knowledge"

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
    parser = argparse.ArgumentParser(description="Ablation 4 EGFR re-run: no domain knowledge")
    add_common_args(parser)
    args = parser.parse_args()

    args.targets = ["EGFR"]
    args.num_molecules = 50
    args.max_attempts = 200
    args.output_dir = None
    args.max_no_new_streak = 20

    config = build_base_config(args)
    Ablation4NoDomainKnowledgeEGFR(config).run()
    logger.info("\nAblation 4 EGFR complete!")


if __name__ == "__main__":
    main()
