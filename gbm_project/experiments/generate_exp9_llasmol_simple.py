#!/usr/bin/env python3
"""
LlaSMol + Mistral-7B 对比实验
信任模型输出，只做最基础的格式清理。
"""

import sys, re, json, csv
from pathlib import Path
from datetime import datetime
from typing import List, Set

import torch
from rdkit import Chem
from rdkit.Chem import Descriptors

PROJECT_ROOT = Path("/root/Llamole-main")
sys.path.insert(0, str(PROJECT_ROOT))

import warnings
warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


# =============================================================================
# 1. 收集已有 SMILES
# =============================================================================

def collect_existing() -> dict:
    exp_dir = PROJECT_ROOT / "gbm_project" / "experiments"
    existing = {"EGFR": set(), "IDH1_IDH2": set(), "VEGF_VEGFR": set()}

    for exp_path in exp_dir.iterdir():
        if not exp_path.is_dir():
            continue
        csv_out = exp_path / "csv_output"
        if csv_out.exists():
            for cf in csv_out.glob("*.csv"):
                target_name = cf.stem.replace("_re", "")
                matched = next((t for t in existing if t.lower() in target_name.lower()), None)
                if not matched:
                    continue
                try:
                    with open(cf, "r", encoding="utf-8", errors="ignore") as f:
                        reader = csv.reader(f)
                        next(reader, None)  # skip header
                        for row in reader:
                            if len(row) >= 4:
                                smi = row[3].strip()
                            elif len(row) >= 2:
                                smi = row[1].strip()
                            else:
                                continue
                            if smi and _rdkit_ok(smi):
                                existing[matched].add(smi)
                except Exception:
                    pass
    return existing


def _rdkit_ok(smi: str) -> bool:
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        if mol is None:
            return False
        Chem.SanitizeMol(mol, catchErrors=True)
        return True
    except Exception:
        return False


# =============================================================================
# 2. 极简提取：信任模型，只清理格式
# =============================================================================

def extract_smiles(text: str) -> List[str]:
    results = []
    seen = set()

    for chunk in re.split(r"[,;\n]", text):
        chunk = chunk.strip()
        chunk = re.sub(r"^[\-\u2022\u2023\u25e6\*\u2043]\s*", "", chunk)
        chunk = re.sub(r"^[\d]+[\.\)\:]+\s*", "", chunk)
        chunk = re.sub(r'^[Ss][Mm][Ii][Ll][Ee][Ss]\s*[:\-]?\s*', "", chunk)
        chunk = re.sub(r'[\`\`\`]', "", chunk).strip()
        if not chunk or len(chunk) < 5:
            continue
        if not re.search(r"[CNOSPFIcnosp]", chunk):
            continue
        if chunk not in seen and _rdkit_ok(chunk):
            seen.add(chunk)
            results.append(chunk)

    return results


# =============================================================================
# 3. 模型
# =============================================================================

def load_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        trust_remote_code=True,
        padding_side="left",
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("Loading base model (4-bit)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    bm = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        trust_remote_code=True,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    print("Loading LlaSMol adapter...")
    m = PeftModel.from_pretrained(bm, "osunlp/LlaSMol-Mistral-7B", torch_dtype=torch.float16)
    m.eval()
    print("Model loaded!")
    return m, tok


# =============================================================================
# 4. 主流程
# =============================================================================

PROMPTS = {
    "EGFR": "Generate 3 valid SMILES for molecules targeting EGFR for glioblastoma. Output SMILES only, one per line.",
    "IDH1_IDH2": "Generate 3 valid SMILES for molecules targeting IDH1 or IDH2 for glioblastoma. Output SMILES only, one per line.",
    "VEGF_VEGFR": "Generate 3 valid SMILES for molecules targeting VEGF or VEGFR for glioblastoma. Output SMILES only, one per line.",
}


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "gbm_project" / "experiments" / f"exp9_llasmol_simple_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_out = out_dir / "csv_output"
    csv_out.mkdir(exist_ok=True)

    print("=" * 60)
    print("Experiment 9: LlaSMol + Mistral-7B Simple Prompt")
    print(f"Output: {out_dir}")
    print("=" * 60)

    # 收集已有分子
    print("\n[1] Collecting existing molecules...")
    existing = collect_existing()
    for t, s in existing.items():
        print(f"  {t}: {len(s)} existing SMILES")

    # 加载模型
    print("\n[2] Loading model...")
    model, tok = load_model()

    # 生成
    print("\n[3] Generating molecules...")
    all_results = {}

    for target in ["EGFR", "IDH1_IDH2", "VEGF_VEGFR"]:
        print(f"\n--- {target} ---")
        prompt = PROMPTS[target]
        seen = set()  # 本次生成中的去重
        new_mols = []  # 相比已有实验的新分子
        dup_mols = []  # 重复分子
        total_extracted = 0
        num_calls = 0

        while len(new_mols) < 50 and num_calls < 20:
            num_calls += 1
            try:
                inputs = tok(prompt, return_tensors="pt")
                inputs = {k: v.to("cuda:3") for k, v in inputs.items()}

                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=300,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=tok.pad_token_id,
                    )

                resp = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                smis = extract_smiles(resp)
                total_extracted += len(smis)

                for s in smis:
                    if s in existing[target]:
                        if s not in dup_mols:
                            dup_mols.append(s)
                    else:
                        if s not in seen:
                            seen.add(s)
                            new_mols.append(s)

                nov = len(new_mols) / total_extracted * 100 if total_extracted else 0
                print(f"  Call {num_calls}: extracted={len(smis)}, new={len(new_mols)}, "
                      f"dup={len(dup_mols)}, total_new={len(new_mols)}/50, novelty={nov:.0f}%")

            except Exception as e:
                print(f"  Call {num_calls} ERROR: {e}")
                continue

        # 保存 CSV
        csv_path = csv_out / f"{target}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["entity_id", "mol_id", "library_name", "smiles", "MW", "TPSA", "LogP", "target"])
            for i, smi in enumerate(new_mols):
                try:
                    mol = Chem.MolFromSmiles(smi)
                    mw = Descriptors.MolWt(mol)
                    tp = Descriptors.TPSA(mol)
                    lp = Descriptors.MlogP(mol)
                except Exception:
                    mw = tp = lp = 0.0
                w.writerow([
                    f"llasmol_{ts}",
                    f"{target}_{i:05d}",
                    "LlaSMol-Mistral-7B",
                    smi,
                    f"{mw:.2f}", f"{tp:.2f}", f"{lp:.2f}",
                    target,
                ])

        print(f"  Saved {len(new_mols)} new molecules to {csv_path}")
        all_results[target] = {
            "new": len(new_mols),
            "duplicate_vs_prior": len(dup_mols),
            "total_extracted": total_extracted,
            "calls": num_calls,
            "novelty_rate": f"{len(new_mols) / total_extracted * 100:.1f}%" if total_extracted else "N/A",
        }

    # 汇总
    print("\n" + "=" * 60)
    print("Novelty Summary")
    print("=" * 60)
    for t, s in all_results.items():
        print(f"\n{t}:")
        for k, v in s.items():
            print(f"  {k}: {v}")

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "exp9_llasmol_simple",
            "model": "LlaSMol + Mistral-7B-v0.1 (4-bit)",
            "adapter": "osunlp/LlaSMol-Mistral-7B",
            "timestamp": ts,
            "results": all_results,
            "existing_counts": {t: len(s) for t, s in existing.items()},
        }, f, indent=2, ensure_ascii=False)

    print(f"\nDone! {out_dir}")


if __name__ == "__main__":
    main()
