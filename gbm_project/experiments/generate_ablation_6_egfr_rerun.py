#!/usr/bin/env python3
"""
Ablation 6 EGFR Re-run: Raw Base Model Only (Direct Prompt -> SMILES)
======================================================================

仅针对 EGFR 靶点重新运行消融实验6（使用原始基座模型，无adapter）。
"""

import sys
import os
import re
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
from rdkit import Chem
from rdkit.Chem import Descriptors

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="peft")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


TARGET_PROMPTS = {
    "EGFR": (
        "Generate 10 valid SMILES strings of drug molecules that target EGFR "
        "(Epidermal Growth Factor Receptor) for GBM (Glioblastoma). "
        "Output only the SMILES, one per line, nothing else."
    ),
}


def _is_valid_smiles(smiles: str) -> bool:
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


def _extract_smiles_from_text(text: str):
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


class BaseModelGenerator:
    def __init__(self, model_path: str, device: str = "cuda:0"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None

    def load(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print(f"Loading tokenizer from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, padding_side="right"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading base model from {self.model_path} on {self.device}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=None,
        )
        torch.cuda.set_device(int(self.device.split(":")[-1]))
        self.model = self.model.to(self.device)
        print(f"Base model loaded: {self.model.__class__.__name__}")

    def generate(self, prompt: str, num_return: int = 10,
                 max_new_tokens: int = 512,
                 temperature: float = 0.8,
                 top_p: float = 0.95):
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


def run_ablation_6_egfr(gpu_id: int, num_molecules: int = 50,
                         max_attempts: int = 100, output_dir: str = None):
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    torch.cuda.empty_cache()

    base_model_path = str(PROJECT_ROOT / "models" / "Qwen2-7B-Instruct")

    generator = BaseModelGenerator(model_path=base_model_path, device=device)
    generator.load()

    if output_dir:
        out_dir = Path(output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = PROJECT_ROOT / "gbm_project" / "experiments" / f"ablation_6_base_model_only_egfr_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_cfg = {
        'experiment_name': 'ablation_6_base_model_only',
        'ablation_description': (
            "Ablation 6 EGFR re-run: Raw Qwen2-7B-Instruct base model, "
            "no adapters, no prompt engineering."
        ),
        'target': 'EGFR',
        'num_molecules': num_molecules,
        'gpu_id': gpu_id,
        'timestamp': datetime.now().isoformat(),
    }
    with open(out_dir / "experiment_config.json", 'w', encoding='utf-8') as f:
        json.dump(exp_cfg, f, indent=2, ensure_ascii=False)

    print(f"Output directory: {out_dir}")

    target = "EGFR"
    prompt = TARGET_PROMPTS.get(target)
    if not prompt:
        print(f"No prompt for {target}, exiting.")
        return

    all_molecules = []
    attempts = 0

    while len(all_molecules) < num_molecules and attempts < max_attempts:
        attempts += 1
        try:
            smiles_list = generator.generate(prompt, num_return=10)
            existing = {m['smiles'] for m in all_molecules}
            new_mols = []
            for smi in smiles_list:
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
            print(f"  Attempt {attempts}: {len(smiles_list)} extracted, "
                  f"{len(new_mols)} new, total {len(all_molecules)}/{num_molecules}")

            _save_checkpoint(all_molecules, target, out_dir)

        except Exception as e:
            print(f"  Attempt {attempts} error: {e}")
            continue

    print(f"  Target {target} done: {len(all_molecules)} molecules in {attempts} attempts")

    target_dir = out_dir / target
    target_dir.mkdir(exist_ok=True)
    with open(target_dir / "molecules.json", 'w', encoding='utf-8') as f:
        json.dump(all_molecules, f, indent=2, ensure_ascii=False)

    summary = {
        'experiment_name': 'ablation_6_base_model_only',
        'target': target,
        'total_molecules': len(all_molecules),
        'timestamp': datetime.now().isoformat(),
    }
    with open(out_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nAblation 6 EGFR complete! Output: {out_dir}")
    return out_dir


def _save_checkpoint(molecules, target, out_dir):
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=3)
    parser.add_argument('--num_molecules', type=int, default=50)
    parser.add_argument('--max_attempts', type=int, default=100)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()
    run_ablation_6_egfr(args.gpu_id, args.num_molecules, args.max_attempts, args.output_dir)
