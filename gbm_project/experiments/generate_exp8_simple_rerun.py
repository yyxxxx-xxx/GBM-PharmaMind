#!/usr/bin/env python3
"""Experiment 8 rerun: First LoRA Adapter, minimal validation, high diversity."""

import sys, os, re, json, argparse
from pathlib import Path
from datetime import datetime
from typing import List

import torch
import warnings; warnings.filterwarnings("ignore")
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')

PROJECT_ROOT = Path('/root/Llamole-main')
sys.path.insert(0, str(PROJECT_ROOT))

from rdkit import Chem

# Three prompt variants per target to encourage diversity
PROMPTS = {
    "EGFR": [
        "Here are 5 EGFR inhibitor SMILES:\nCC(=O)Nc1ccc(Oc2ccnc3ccc(OCC)nc23)cc1\nCOc1ccc2[nH]c(=O)n(-c3ccc(Cl)cc3)c2c1\nCCOc1ccc(OC)cc1C(=O)Nc1cc(OC)ccc1C#N\nCS(=O)(=O)Nc1ccc(Nc2ccnc3ccc(OCC)nc23)cc1\nCC(C)Nc1ncnc2ccc(-c3cccnc3)cc12\n\nGenerate 5 more different EGFR inhibitor SMILES for GBM. Output ONLY the SMILES strings, one per line:\n",
        "Give me 5 EGFR GBM inhibitor SMILES, one per line:\n",
        "SMILES for EGFR GBM inhibitors:\n",
    ],
    "IDH1_IDH2": [
        "Here are 5 IDH1 inhibitor SMILES:\nCC1=CC(=NN1C1=CC=C(C=C1)C(=O)N1CCC(C)CC1)C(=O)N1CCC(C)CC1\nCC(C)CC1=CC=C(N1C(=O)C2=CC=C(C=C2)C(=O)N1CC(C)C)C=C1\nCC(=O)NC1=CC=C(C=C1)C1=NN(CC2=CC=C(C=C2)C(=O)NC(C)C)C2=CC=CC=C12\nCC1=CC=C(N1C(=O)C2=CC=C(C=C2)C(=O)N1CC(C)C)C=C1\nCC1=CC=C(N1C(=O)C2=CC=CC=C2)C=C1\n\nGenerate 5 more different IDH1/IDH2 inhibitor SMILES for GBM. Output ONLY the SMILES strings, one per line:\n",
        "Give me 5 IDH1/IDH2 GBM inhibitor SMILES, one per line:\n",
        "SMILES for IDH1/IDH2 GBM inhibitors:\n",
    ],
    "VEGF_VEGFR": [
        "Here are 5 VEGFR inhibitor SMILES:\nCC(C)n1c(=O)[nH]c2cc(Oc3ccc(OC)cc3)cnc21\nCC(=O)Oc1ccc2nccc(Oc3ccc(OC)nc3)c2n1\nCOc1ccc(-c2cc(OC)nc(N)n2)cc1\nCC(C)Nc1ccc2c(c1)NC(=O)CO2\nCC1=CC=C(C2=CNN(C2)C2=CC=C(OC)CC=C2)C=C1\n\nGenerate 5 more different VEGFR inhibitor SMILES for GBM. Output ONLY the SMILES strings, one per line:\n",
        "Give me 5 VEGF/VEGFR GBM inhibitor SMILES, one per line:\n",
        "SMILES for VEGF/VEGFR GBM inhibitors:\n",
    ],
}


def _basic_clean(text: str) -> str:
    text = re.sub(r'\s+', '', text)
    return text.strip(' ,.;:"\'`[]')


def _is_valid(smi: str) -> bool:
    if not smi or len(smi) < 5 or len(smi) > 600:
        return False
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        if mol is None:
            return False
        Chem.SanitizeMol(mol, catchErrors=True)
        return True
    except Exception:
        return False


def _extract_smiles(text: str) -> List[str]:
    seen = set()
    results = []

    for m in re.findall(r'SMILES\s*\d*\s*:\s*([^\n]{5,})', text, re.IGNORECASE):
        smi = _basic_clean(m)
        if smi and smi not in seen and _is_valid(smi):
            seen.add(smi); results.append(smi)
    if results:
        return results

    for line in text.strip().split('\n'):
        line = line.strip()
        line = re.sub(r'^[\d\.\-\)\:]+\s*', '', line)
        smi = _basic_clean(line)
        if smi and smi not in seen and _is_valid(smi):
            seen.add(smi); results.append(smi)
    if results:
        return results

    for m in re.finditer(r'([CNOSPFI][A-Za-z0-9@+\-\[\]\(\)=#%/\.\\]{8,})', text):
        cand = m.group(1)
        cand = re.sub(r'\s+', '', cand)
        if cand and cand not in seen and _is_valid(cand):
            seen.add(cand); results.append(cand)
            if len(results) >= 10:
                break

    return results


def _run(gpu_id: int, targets: List[str], num_molecules: int,
         max_calls: int, output_dir: str):
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    torch.cuda.empty_cache()

    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModelForCausalLM

    base_model = str(PROJECT_ROOT / "models" / "Qwen2-7B-Instruct")
    adapter = str(PROJECT_ROOT / "saves" / "Llamole-Qwen2-7B-Instruct-Adapter")

    print(f"Loading tokenizer from {base_model}")
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, padding_side="right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading base model from {base_model}")
    bm = AutoModelForCausalLM.from_pretrained(
        base_model, trust_remote_code=True, torch_dtype=torch.float16, device_map=device
    )

    print(f"Loading LoRA adapter from {adapter}")
    model = PeftModel.from_pretrained(bm, adapter, torch_dtype=torch.float16)
    model.eval()
    print("Model ready!")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = out_dir / "csv_output"
    csv_dir.mkdir(exist_ok=True)

    exp_cfg = {
        'experiment_name': 'exp8_first_lora_simple',
        'description': 'Exp8 rerun: example prompts, minimal validation, high temp for diversity.',
        'base_model': base_model,
        'lora_adapter': adapter,
        'gbm_adapter': None,
        'targets': targets,
        'num_molecules': num_molecules,
        'gpu_id': gpu_id,
        'timestamp': datetime.now().isoformat(),
    }
    with open(out_dir / "experiment_config.json", 'w', encoding='utf-8') as f:
        json.dump(exp_cfg, f, indent=2, ensure_ascii=False)

    print(f"Output: {out_dir}")

    total = 0
    for target in targets:
        print("=" * 60)
        print(f"[exp8] Target: {target} | Goal: {num_molecules}")
        print("=" * 60)

        prompts = PROMPTS.get(target, [])
        if not prompts:
            print(f"  No prompt for {target}, skipping")
            continue

        all_mols = []
        seen = set()
        calls = 0
        prompt_idx = 0

        while len(all_mols) < num_molecules and calls < max_calls:
            calls += 1
            prompt = prompts[prompt_idx % len(prompts)]

            try:
                msgs = [{"role": "user", "content": prompt}]
                formatted = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inputs = tok(formatted, return_tensors="pt").to(device)
                inp_len = inputs["input_ids"].shape[1]

                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=600,
                        temperature=1.0,
                        top_p=0.95,
                        do_sample=True,
                        pad_token_id=tok.pad_token_id,
                        eos_token_id=tok.eos_token_id,
                    )

                resp = tok.decode(out[0][inp_len:], skip_special_tokens=True)
                smis = _extract_smiles(resp)

                new_count = 0
                for smi in smis:
                    if smi not in seen:
                        seen.add(smi)
                        all_mols.append(smi)
                        new_count += 1

                if len(smis) == 0:
                    prompt_idx += 1

                print(f"  Call {calls} (v{prompt_idx % len(prompts)}): "
                      f"extracted={len(smis)}, new={new_count}, total={len(all_mols)}/{num_molecules}")

                _save_csv(all_mols, target, csv_dir)

            except Exception as e:
                print(f"  Call {calls} ERROR: {e}")
                continue

        print(f"  {target} done: {len(all_mols)} unique molecules in {calls} calls")

        target_dir = out_dir / target
        target_dir.mkdir(exist_ok=True)
        mol_list = [{"id": i, "smiles": s, "target": target} for i, s in enumerate(all_mols)]
        with open(target_dir / "molecules.json", 'w', encoding='utf-8') as f:
            json.dump(mol_list, f, indent=2, ensure_ascii=False)

        total += len(all_mols)

    # Final clean CSV rebuild with proper deduplication
    for target in targets:
        csv_path = csv_dir / f"{target}.csv"
        if csv_path.exists():
            seen = set()
            unique_smiles = []
            for line in csv_path.read_text().strip().split('\n'):
                if ',' in line:
                    smi = line.split(',', 2)[1]
                    if smi not in seen:
                        seen.add(smi)
                        unique_smiles.append(smi)
            lines = ["entity_id,mol_id,library_name,smiles,MW,TPSA,LogP,target"]
            for i, smi in enumerate(unique_smiles):
                lines.append(f"llasmol_20260507_160000,{target}_{i:05d},LlaSMol-Qwen2-7B,{smi},0.00,0.00,0.00,{target}")
            csv_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    summary = {
        'experiment_name': 'exp8_first_lora_simple',
        'targets_processed': targets,
        'total_molecules': total,
        'timestamp': datetime.now().isoformat(),
    }
    with open(out_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Output: {out_dir}, Total unique molecules: {total}")


def _save_csv(molecules: List[str], target: str, csv_dir: Path):
    csv_path = csv_dir / f"{target}.csv"
    existing_smiles = set()
    if csv_path.exists():
        for line in csv_path.read_text().strip().split('\n'):
            if ',' in line:
                smi = line.split(',', 2)[1]
                existing_smiles.add(smi)

    with open(csv_path, 'a', encoding='utf-8') as f:
        for smi in molecules:
            if smi not in existing_smiles:
                mol_id = f"{target}_{len(existing_smiles):05d}"
                f.write(f"llasmol_20260507_160000,{mol_id},LlaSMol-Qwen2-7B,{smi},0.00,0.00,0.00,{target}\n")
                existing_smiles.add(smi)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--targets', type=str, nargs='+',
                        default=["EGFR", "IDH1_IDH2", "VEGF_VEGFR"])
    parser.add_argument('--num_molecules', type=int, default=50)
    parser.add_argument('--gpu_id', type=int, default=3)
    parser.add_argument('--max_calls', type=int, default=25)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(PROJECT_ROOT / "gbm_project" / "experiments" /
                              f"exp8_first_lora_simple_{ts}")

    _run(args.gpu_id, args.targets, args.num_molecules,
         args.max_calls, args.output_dir)
