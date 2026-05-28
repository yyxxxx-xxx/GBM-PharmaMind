#!/usr/bin/env python3
"""
Experiment 8: First LoRA Adapter + Simple Prompt Direct Generation
====================================================================
对比实验：使用第一个LoRA微调保存的参数（Llamole-Qwen2-7B-Instruct-Adapter），
直接输入简单prompt生成GBM药物分子，不使用ToT框架和CoT模板。

模型组合：
  - Base model: Qwen2-7B-Instruct
  - Adapter: saves/Llamole-Qwen2-7B-Instruct-Adapter（第一个微调保存的参数）
  - 不加载 GBM-specific adapter (Adapter-gbm-with-graph-models)

Prompt策略：
  - 简单英文指令："Generate 50 drug molecules targeting xx for GBM"
  - 不使用任何领域知识注入、CoT模板、ToT框架

实验目标：
  - 建立第一个LoRA微调模型在简单prompt下的生成基线
  - 对比 Full Model (ToT + CoT + GBM adapter) 验证各模块贡献
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

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="peft")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


SIMPLE_PROMPTS = {
    "EGFR": (
        "You are a medicinal chemist. Generate exactly 5 valid SMILES strings for drug molecules "
        "targeting EGFR (Epidermal Growth Factor Receptor) for Glioblastoma (GBM) treatment.\n\n"
        "Requirements:\n"
        "- Molecular Weight: 300-500 Da\n"
        "- Blood-Brain Barrier penetration: HIGH\n"
        "- LogP: 2.0-4.0\n"
        "- Synthesizable\n\n"
        "Output format (EXACTLY follow this):\n"
        "SMILES 1: <smiles>\n"
        "SMILES 2: <smiles>\n"
        "SMILES 3: <smiles>\n"
        "SMILES 4: <smiles>\n"
        "SMILES 5: <smiles>\n\n"
        "Only output the numbered SMILES list above, nothing else."
    ),
    "IDH1_IDH2": (
        "You are a medicinal chemist. Generate exactly 5 valid SMILES strings for drug molecules "
        "targeting IDH1 or IDH2 (Isocitrate Dehydrogenase 1/2) for Glioblastoma (GBM) treatment.\n\n"
        "Requirements:\n"
        "- Molecular Weight: 300-500 Da\n"
        "- Blood-Brain Barrier penetration: HIGH\n"
        "- LogP: 2.0-4.0\n"
        "- Synthesizable\n\n"
        "Output format (EXACTLY follow this):\n"
        "SMILES 1: <smiles>\n"
        "SMILES 2: <smiles>\n"
        "SMILES 3: <smiles>\n"
        "SMILES 4: <smiles>\n"
        "SMILES 5: <smiles>\n\n"
        "Only output the numbered SMILES list above, nothing else."
    ),
    "VEGF_VEGFR": (
        "You are a medicinal chemist. Generate exactly 10 valid SMILES strings for drug molecules "
        "targeting VEGF or VEGFR (Vascular Endothelial Growth Factor/Receptor) for Glioblastoma (GBM).\n\n"
        "IMPORTANT: Only output SMILES strings. Each line must be a valid SMILES format starting with an atom symbol (C, N, O, S, P, F, B, Br, Cl, I).\n\n"
        "Requirements:\n"
        "- Molecular Weight: 250-600 Da\n"
        "- Blood-Brain Barrier penetration: HIGH\n"
        "- Synthesizable\n\n"
        "Output format (one SMILES per line, no explanations):\n"
        "SMILES 1: <smiles>\n"
        "SMILES 2: <smiles>\n"
        "SMILES 3: <smiles>\n"
        "SMILES 4: <smiles>\n"
        "SMILES 5: <smiles>\n"
        "SMILES 6: <smiles>\n"
        "SMILES 7: <smiles>\n"
        "SMILES 8: <smiles>\n"
        "SMILES 9: <smiles>\n"
        "SMILES 10: <smiles>\n\n"
        "Only output the numbered SMILES list above, nothing else."
    ),
}


def _is_valid_smiles(smiles: str) -> bool:
    smiles = smiles.strip()
    if any(c.isspace() for c in smiles):
        return False
    if smiles and smiles[0] not in "CNOSFPBIBrCl":
        return False
    if not re.match(r"^[A-Za-z0-9@+\-\[\]\(\)=#%/\\.]+$", smiles):
        return False
    s_lower = smiles.lower()
    invalid = [
        "chain-of-thought", "analysis", "reasoning",
        "scaffold", "strategy", "rationale", "therefore",
        "synthetic", "procedure", "output", "format", "here is",
        "valid_smiles", "smiles_string", "note:", "note :",
        "example", "generated", "molecule", "compound",
        "retrsynthesis", "nmr", "yellow solid",
        "preparation", "intermediate", "furthermore", "specifically",
        "preferably", "similarly", "chloro", "bromo", "fluoro",
        "preparation", "soluble", "solubility", "colorless",
        "tert-butyl", "dimethylamino", "dimethyl",
        "isocyanato", "hydroxysuccinimide", "carboxamide",
        "phenyl", "benzyl", "acetyl", "glycyl", "methyl",
        "dmso", "etoh", "meoh", "thf", "dcm",
    ]
    if any(w in s_lower for w in invalid):
        return False
    if len(smiles) < 5 or len(smiles) > 600:
        return False
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return False
        Chem.SanitizeMol(mol, catchErrors=True)
        tpsa = Descriptors.TPSA(mol)
        if not (20.0 <= tpsa <= 150.0):
            return False
    except Exception:
        return False
    return True


def _extract_smiles(text: str) -> List[str]:
    # 策略1：匹配带编号的SMILES行
    pattern_num = r"SMILES\s*\d*\s*:\s*([^\n]+)"
    matches = re.findall(pattern_num, text, re.IGNORECASE | re.MULTILINE)
    results = []
    for m in matches:
        cleaned = re.sub(r"\s+", "", m.strip()).strip(" ,.;")
        if cleaned and _is_valid_smiles(cleaned):
            results.append(cleaned)

    # 策略2：匹配纯SMILES-like字符串（每行一个，在"Generate"之后的内容）
    if not results:
        lines = text.strip().split('\n')
        for line in lines:
            line = line.strip()
            line = re.sub(r'^[Ss][Mm][Ii][Ll][Ee][Ss]\s*[:\-]?\s*', '', line)
            line = re.sub(r'^\d+[\.\)\:]+\s*', '', line)
            line = re.sub(r'^\-\s*', '', line)
            line = re.sub(r'["\'\[\]]', '', line).strip()
            if not line:
                continue
            if _is_valid_smiles(line):
                results.append(line)

    # 策略3：直接从文本中挖取SMILES-like片段
    if not results:
        pattern_smiles = r"\b([CNOSPFI][A-Za-z0-9@+\-\[\]\(\)=#%/\.]{8,})\b"
        seen = set()
        for match in re.finditer(pattern_smiles, text):
            cand = match.group(1).strip()
            if cand not in seen and _is_valid_smiles(cand):
                results.append(cand)
                seen.add(cand)

    return list(dict.fromkeys(results))[:10]


class FirstLoraSimpleGenerator:
    """
    使用第一个LoRA adapter（不含GBM图模型）的直接生成器。
    """

    def __init__(self, base_model_path: str, lora_adapter_path: str, device: str):
        self.base_model_path = base_model_path
        self.lora_adapter_path = lora_adapter_path
        self.device = device
        self.model = None
        self.tokenizer = None

    def load(self):
        from peft import PeftModel
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"Loading tokenizer from {self.base_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path, trust_remote_code=True, padding_side="right"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading base model from {self.base_model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=self.device,
        )

        print(f"Loading first LoRA adapter from {self.lora_adapter_path}")
        self.model = PeftModel.from_pretrained(
            base_model,
            self.lora_adapter_path,
            torch_dtype=torch.float16,
        )
        self.model.eval()
        print(f"First LoRA adapter loaded successfully")

    def generate(self, prompt: str, num_return: int = 50,
                 max_new_tokens: int = 1200,
                 temperature: float = 0.8,
                 top_p: float = 0.9) -> List[str]:
        """Direct generation with simple prompt, no CoT, no ToT."""
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
        return _extract_smiles(response)


def run_experiment_8(
    gpu_id: int,
    targets: List[str],
    num_molecules: int = 50,
    max_calls_per_target: int = 10,
    molecules_per_call: int = 5,
    output_dir: str = None,
):
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    torch.cuda.empty_cache()

    base_model_path = str(PROJECT_ROOT / "models" / "Qwen2-7B-Instruct")
    lora_adapter_path = str(PROJECT_ROOT / "saves" / "Llamole-Qwen2-7B-Instruct-Adapter")

    generator = FirstLoraSimpleGenerator(
        base_model_path=base_model_path,
        lora_adapter_path=lora_adapter_path,
        device=device,
    )
    generator.load()

    if output_dir:
        out_dir = Path(output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = PROJECT_ROOT / "gbm_project" / "experiments" / f"exp8_first_lora_simple_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_cfg = {
        'experiment_name': 'exp8_first_lora_simple',
        'description': (
            "Experiment 8: First LoRA adapter (Llamole-Qwen2-7B-Instruct-Adapter) "
            "with simple prompt direct generation. No ToT framework, no CoT template, "
            "no GBM-specific adapter. Tests the baseline capability of the first LoRA model."
        ),
        'base_model': base_model_path,
        'lora_adapter': lora_adapter_path,
        'gbm_adapter': None,
        'targets': targets,
        'num_molecules': num_molecules,
        'gpu_id': gpu_id,
        'timestamp': datetime.now().isoformat(),
    }
    with open(out_dir / "experiment_config.json", 'w', encoding='utf-8') as f:
        json.dump(exp_cfg, f, indent=2, ensure_ascii=False)

    print(f"Output directory: {out_dir}")

    total_molecules = 0

    for target in targets:
        print("=" * 80)
        print(f"[exp8] Processing target: {target}")
        print("=" * 80)

        prompt = SIMPLE_PROMPTS.get(target)
        if not prompt:
            print(f"No prompt defined for {target}, skipping")
            continue

        print(f"Starting {target} generation...")

        all_molecules = []
        existing_smiles = set()
        num_calls = 0

        while len(all_molecules) < num_molecules and num_calls < max_calls_per_target:
            num_calls += 1
            try:
                smiles_list = generator.generate(
                    prompt,
                    num_return=molecules_per_call,
                    max_new_tokens=1000,
                )

                new_mols = []
                for i, smi in enumerate(smiles_list):
                    if smi not in existing_smiles:
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
                        existing_smiles.add(smi)

                all_molecules.extend(new_mols)
                print(f"  Call {num_calls}: {len(smiles_list)} extracted, "
                      f"{len(new_mols)} new, total {len(all_molecules)}/{num_molecules}")

                _save_checkpoint(all_molecules, target, out_dir)

            except Exception as e:
                print(f"  Call {num_calls} error: {e}")
                continue

        print(f"  Target {target} done: {len(all_molecules)} molecules in {num_calls} calls")

        target_dir = out_dir / target
        target_dir.mkdir(exist_ok=True)
        with open(target_dir / "molecules.json", 'w', encoding='utf-8') as f:
            json.dump(all_molecules, f, indent=2, ensure_ascii=False)

        total_molecules += len(all_molecules)

    summary = {
        'experiment_name': 'exp8_first_lora_simple',
        'targets_processed': targets,
        'total_molecules': total_molecules,
        'timestamp': datetime.now().isoformat(),
    }
    with open(out_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nExperiment 8 complete! Output: {out_dir}, Total molecules: {total_molecules}")
    return out_dir


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 8: First LoRA adapter + simple prompt direct generation"
    )
    parser.add_argument('--targets', type=str, nargs='+',
                        default=["EGFR", "IDH1_IDH2", "VEGF_VEGFR"],
                        help='Targets to process')
    parser.add_argument('--num_molecules', type=int, default=50,
                        help='Molecules per target (default: 50)')
    parser.add_argument('--gpu_id', type=int, default=1,
                        help='GPU device ID (default: 1)')
    parser.add_argument('--max_calls', type=int, default=10,
                        help='Max model calls per target (default: 10)')
    parser.add_argument('--molecules_per_call', type=int, default=10,
                        help='Molecules to request per call (default: 10)')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    run_experiment_8(
        gpu_id=args.gpu_id,
        targets=args.targets,
        num_molecules=args.num_molecules,
        max_calls_per_target=args.max_calls,
        molecules_per_call=args.molecules_per_call,
        output_dir=args.output_dir,
    )
