#!/usr/bin/env python3
"""
Ablation 6: Raw Base Model Only (Direct Prompt → SMILES)
=======================================================
消融实验 6：只使用原始基座模型（Qwen2-7B-Instruct），不做任何 prompt engineering。

行为：
  - 只加载 Qwen2-7B-Instruct 基座模型（不加载任何 Llamole/GBM adapter）
  - 不使用 ToT scaffold/assembly/evaluation 框架
  - 直接把英文 prompt 输入模型，从模型输出中提取 SMILES
  - 不做物理评估（仅做基本的 SMILES 有效性校验）

实验目标：建立无微调基座模型在 GBM 分子生成任务上的原始基线。
"""

import sys
import os
import re
import json
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="peft")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


# ── Target → English prompt mapping ───────────────────────────────────────────

TARGET_PROMPTS = {
    "EGFR": (
        "Generate 10 valid SMILES strings of drug molecules that target EGFR "
        "(Epidermal Growth Factor Receptor) for GBM (Glioblastoma). "
        "Output only the SMILES, one per line, nothing else."
    ),
    "IDH1_IDH2": (
        "Generate 10 valid SMILES strings of drug molecules that target IDH1 or IDH2 "
        "(Isocitrate Dehydrogenase 1/2) for GBM (Glioblastoma). "
        "Output only the SMILES, one per line, nothing else."
    ),
    "VEGF_VEGFR": (
        "Generate 10 valid SMILES strings of drug molecules that target VEGF or VEGFR "
        "(Vascular Endothelial Growth Factor/Receptor) for GBM (Glioblastoma). "
        "Output only the SMILES, one per line, nothing else."
    ),
}


# ── SMILES validation ─────────────────────────────────────────────────────────

def _is_valid_smiles(smiles: str) -> bool:
    """Basic SMILES validity check."""
    smiles = smiles.strip()
    if not smiles or len(smiles) < 5 or len(smiles) > 600:
        return False
    if any(c.isspace() for c in smiles):
        return False
    invalid_words = [
        "chain-of-thought", "analysis", "reasoning", "design",
        "scaffold", "strategy", "rationale", "therefore",
        "synthetic", "procedure", "output", "format", "here is",
        "valid_smiles", "smiles_string", "note:", "note :",
        "example", "generated", "molecule", "compound",
    ]
    s_lower = smiles.lower()
    if any(w in s_lower for w in invalid_words):
        return False
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return False
        Chem.SanitizeMol(mol, catchErrors=True)
        tpsa = Descriptors.TPSA(mol)
        if not (40.0 <= tpsa <= 120.0):
            return False
    except Exception:
        return False
    return True


def _extract_smiles_from_text(text: str) -> List[str]:
    """Extract SMILES-like strings from raw model output."""
    lines = text.strip().split('\n')
    candidates = []
    for line in lines:
        line = line.strip()
        line = re.sub(r'^[Ss][Mm][Ii][Ll][Ee][Ss]\s*[:\-]?\s*', '', line)
        line = re.sub(r'^\d+[\.\)\:]+\s*', '', line)
        line = re.sub(r'^\-\s*', '', line)
        line = re.sub(r'["\'\[\]]', '', line).strip()
        if not line:
            continue
        if _is_valid_smiles(line):
            candidates.append(line)
    return candidates


# ── Model wrapper ─────────────────────────────────────────────────────────────

class BaseModelGenerator:
    """Direct wrapper around the raw base model for ablation 6."""

    def __init__(self, model_path: str, device: str = "cuda:0"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None

    def load(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        logger.info(f"Loading tokenizer from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, padding_side="right"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Loading base model from {self.model_path} on {self.device}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=None,
        )
        torch.cuda.set_device(int(self.device.split(":")[-1]) if ":" in self.device else 0)
        self.model = self.model.to(self.device)
        logger.info(f"Base model loaded: {self.model.__class__.__name__}")

    def generate(self, prompt: str, num_return: int = 10,
                 max_new_tokens: int = 512,
                 temperature: float = 0.8,
                 top_p: float = 0.95) -> List[str]:
        """Directly call the base model with a prompt and extract SMILES."""
        try:
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            formatted = prompt

        inputs = self.tokenizer(formatted, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][input_len:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return _extract_smiles_from_text(response)


# ── Main experiment ────────────────────────────────────────────────────────────

def build_config(args) -> Dict[str, Any]:
    return {
        'gpu_id': args.gpu_id,
        'base_model_path': str(PROJECT_ROOT / "models" / "Qwen2-7B-Instruct"),
        'num_molecules': args.num_molecules,
        'max_attempts': args.max_attempts,
        'targets': args.targets,
        'output_dir': args.output_dir,
    }


def run_ablation_6(config: Dict[str, Any]):
    device = f"cuda:{config['gpu_id']}"
    torch.cuda.set_device(config['gpu_id'])
    torch.cuda.empty_cache()

    generator = BaseModelGenerator(
        model_path=config['base_model_path'],
        device=device,
    )
    generator.load()

    # Setup output
    if config.get('output_dir'):
        out_dir = Path(config['output_dir'])
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = PROJECT_ROOT / "gbm_project" / "experiments" / f"ablation_6_base_model_only_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_cfg = {
        'experiment_name': 'ablation_6_base_model_only',
        'ablation_description': (
            "Ablation 6: Raw Qwen2-7B-Instruct base model with no adapters and no "
            "prompt engineering. Direct prompt → model → SMILES extraction. "
            "Tests the zero-shot capability of the unfine-tuned base model."
        ),
        'targets': config['targets'],
        'num_molecules': config['num_molecules'],
        'gpu_id': config['gpu_id'],
        'timestamp': datetime.now().isoformat(),
    }
    with open(out_dir / "experiment_config.json", 'w', encoding='utf-8') as f:
        json.dump(exp_cfg, f, indent=2, ensure_ascii=False)

    logger.info(f"Output directory: {out_dir}")
    total_molecules = 0

    for target in config['targets']:
        logger.info("=" * 80)
        logger.info(f"[ablation_6] Processing target: {target}")
        logger.info("=" * 80)

        prompt = TARGET_PROMPTS.get(target)
        if not prompt:
            logger.warning(f"No prompt defined for target: {target}, skipping")
            continue

        all_molecules = []
        attempts = 0

        while len(all_molecules) < config['num_molecules'] and attempts < config['max_attempts']:
            attempts += 1
            try:
                smiles_list = generator.generate(prompt, num_return=10)
                existing = {m['smiles'] for m in all_molecules}
                new_mols = []
                for i, smi in enumerate(smiles_list):
                    if smi not in existing:
                        try:
                            mol = Chem.MolFromSmiles(smi, sanitize=False)
                            Chem.SanitizeMol(mol, catchErrors=True)
                            tpsa = Descriptors.TPSA(mol)
                            mw = Descriptors.MolWt(mol)
                        except Exception:
                            tpsa, mw = 0.0, 0.0
                        new_mols.append({
                            'id': len(all_molecules) + len(new_mols),
                            'smiles': smi,
                            'tpsa': tpsa,
                            'mw': mw,
                            'target': target,
                        })
                        existing.add(smi)

                all_molecules.extend(new_mols)
                logger.info(
                    f"  Attempt {attempts}: {len(smiles_list)} extracted, "
                    f"{len(new_mols)} new, total {len(all_molecules)}/{config['num_molecules']}"
                )

                # Checkpoint
                _save_checkpoint(all_molecules, target, out_dir)

            except Exception as e:
                logger.error(f"  Attempt {attempts} error: {e}")
                continue

        logger.info(f"  Target {target} done: {len(all_molecules)} molecules in {attempts} attempts")

        # Save per-target results
        target_dir = out_dir / target
        target_dir.mkdir(exist_ok=True)
        with open(target_dir / "molecules.json", 'w', encoding='utf-8') as f:
            json.dump(all_molecules, f, indent=2, ensure_ascii=False)

        total_molecules += len(all_molecules)

    # Summary
    summary = {
        'experiment_name': 'ablation_6_base_model_only',
        'targets_processed': config['targets'],
        'total_molecules': total_molecules,
        'timestamp': datetime.now().isoformat(),
    }
    with open(out_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'=' * 80}")
    logger.info(f"[ablation_6] Experiment complete!")
    logger.info(f"Output: {out_dir}")
    logger.info(f"Total molecules: {total_molecules}")
    logger.info(f"{'=' * 80}")


def _save_checkpoint(molecules: List[Dict], target: str, out_dir: Path):
    csv_dir = out_dir / "csv_output"
    csv_dir.mkdir(exist_ok=True)
    csv_file = csv_dir / f"{target}.csv"

    existing_smiles = set()
    if csv_file.exists():
        with open(csv_file) as f:
            for line in f.readlines()[1:]:
                if line.strip():
                    parts = line.strip().split(',', 2)
                    if len(parts) >= 2:
                        existing_smiles.add(parts[1])

    with open(csv_file, 'a') as f:
        for mol in molecules:
            smi = mol.get('smiles', '')
            if smi not in existing_smiles:
                tpsa = mol.get('tpsa', 0)
                mw = mol.get('mw', 0)
                f.write(f"{target}_{mol.get('id', 0):05d},{smi},{tpsa:.2f},{mw:.2f},{target}\n")
                existing_smiles.add(smi)


def main():
    parser = argparse.ArgumentParser(
        description="Ablation 6: Raw base model only (direct prompt → SMILES)"
    )
    parser.add_argument('--targets', type=str, nargs='+', default=None,
                        help='Target list (e.g. EGFR IDH1_IDH2 VEGF_VEGFR)')
    parser.add_argument('--num_molecules', type=int, default=50,
                        help='Molecules per target (default: 50)')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU device ID (default: 0)')
    parser.add_argument('--max_attempts', type=int, default=100,
                        help='Max calls per target (default: 100)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: auto timestamp)')
    args = parser.parse_args()

    if not args.targets:
        args.targets = ["EGFR", "IDH1_IDH2", "VEGF_VEGFR"]

    config = build_config(args)
    run_ablation_6(config)


if __name__ == "__main__":
    main()
