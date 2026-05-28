#!/usr/bin/env python3
"""
从 Llamole 逆合成规划原始输出中提取 SMILES
直接读取已保存的 raw JSON 文件，精确提取目标分子 SMILES
"""

import json
import re
import csv
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

from rdkit import Chem
from rdkit.Chem import Descriptors

import warnings
warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


def basic_validate(smi: str) -> bool:
    """宽松的 RDKit 验证。"""
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        if mol is None:
            return False
        Chem.SanitizeMol(mol, catchErrors=True)
        return True
    except Exception:
        return False


def is_druglike(smi: str) -> bool:
    """检查是否符合药物相似性标准。"""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False
        mw = Descriptors.MolWt(mol)
        tp = Descriptors.TPSA(mol)
        lp = Descriptors.MolLogP(mol)
        return 150 <= mw <= 700 and 20 <= tp <= 140
    except Exception:
        return False


def extract_from_retrosynthesis(text: str) -> List[Tuple[str, str, dict]]:
    """
    从逆合成规划文本中提取目标分子 SMILES。
    返回: [(smiles, strategy, props), ...]
    """
    results = []
    seen = set()

    # ================================================================
    # 策略0: 格式 "Therefore, the designed molecule is:    N: SMILES."
    # ================================================================
    m = re.search(
        r"Therefore,?\s+the\s+designed\s+molecule\s+is\s*[:\.]?\s*",
        text, re.IGNORECASE
    )
    if m:
        after = text[m.end():m.end() + 600].lstrip()
        # 去掉序号 "N: "
        after = re.sub(r"^\d+:\s*", "", after)
        # 提取到第一个句号为止
        smi_match = re.match(r"([A-Za-z0-9@+\-\[\]\(\)=#%/\.\\=]+)\.", after)
        if smi_match:
            smi = smi_match.group(1).strip()
            if basic_validate(smi):
                try:
                    mol = Chem.MolFromSmiles(smi)
                    mw = Descriptors.MolWt(mol)
                    tp = Descriptors.TPSA(mol)
                    lp = Descriptors.MolLogP(mol)
                    results.append((smi, "designed_molecule", {"mw": mw, "tpsa": tp, "logp": lp}))
                    seen.add(smi)
                except Exception:
                    pass

    # ================================================================
    # 策略1: "The applied reaction is:" 后面的完整反应式
    # 格式: "The applied reaction is: SMILES, which requires..."
    # ================================================================
    for m in re.finditer(r"The\s+applied\s+reaction\s+is\s*:\s*", text, re.IGNORECASE):
        segment = text[m.end():m.end() + 600]
        # 去掉 "which requires..." 以后的内容
        segment = re.sub(r",?\s*which\s+requires.*$", "", segment, flags=re.IGNORECASE)
        segment = segment.strip().rstrip(".,")
        if not segment:
            continue

        # 处理 "." 分隔的多个片段
        parts = segment.split(".")
        for part in parts:
            part = part.strip().rstrip(".,")
            if not part or len(part) < 8:
                continue
            # 排除非SMILES的文字
            if re.match(r"^\d", part):  # 数字开头
                continue
            if part.lower() in ["available", "not available"]:
                continue
            if basic_validate(part) and part not in seen:
                try:
                    mol = Chem.MolFromSmiles(part)
                    mw = Descriptors.MolWt(mol)
                    tp = Descriptors.TPSA(mol)
                    lp = Descriptors.MolLogP(mol)
                    results.append((part, "applied_reaction", {"mw": mw, "tpsa": tp, "logp": lp}))
                    seen.add(part)
                except Exception:
                    pass

    # ================================================================
    # 策略2: "requires the reactants:" 中间的 SMILES
    # 格式: "...: SMILES (available), SMILES (not available)"
    # ================================================================
    for m in re.finditer(r"requires\s+the\s+reactants?\s*:\s*", text, re.IGNORECASE):
        segment = text[m.end():m.end() + 600]
        # 提取 ":" 之后的 SMILES-like 片段（到 "This is step" 或句尾）
        segment = re.sub(r"\s*This\s+is\s+step.*$", "", segment, flags=re.IGNORECASE)
        # 按 ", " 切分
        for chunk in re.split(r",\s*", segment):
            chunk = chunk.strip()
            if not chunk:
                continue
            # 去掉 "(available)" / "(not available)" 等括号内容
            chunk_clean = re.sub(r"\s*\([^)]*\)", "", chunk).strip()
            chunk_clean = chunk_clean.rstrip(".,")
            if not chunk_clean or len(chunk_clean) < 8:
                continue
            if chunk_clean not in seen and basic_validate(chunk_clean):
                try:
                    mol = Chem.MolFromSmiles(chunk_clean)
                    mw = Descriptors.MolWt(mol)
                    tp = Descriptors.TPSA(mol)
                    lp = Descriptors.MolLogP(mol)
                    results.append((chunk_clean, "requires_reactants", {"mw": mw, "tpsa": tp, "logp": lp}))
                    seen.add(chunk_clean)
                except Exception:
                    pass

    # ================================================================
    # 策略3: 贪婪模式 - 所有以化学原子符号开头的长片段
    # 专注于 step 1 的目标分子（通常在 "To synthesize" 后)
    # ================================================================
    # 找 "This is step 1" 位置，之前的内容最可能包含目标分子 SMILES
    step1_pos = re.search(r"This\s+is\s+step\s*1", text, re.IGNORECASE)
    if step1_pos:
        # 截取 "Therefore" 到 "step 1" 之间的内容
        therefore_pos = re.search(r"Therefore", text, re.IGNORECASE)
        if therefore_pos:
            search_window = text[therefore_pos.start():step1_pos.end()]
        else:
            search_window = text[:step1_pos.end()]
    else:
        search_window = text[:2000]

    # 在窗口内贪婪提取
    pattern = r"\b([CNOSPFI][A-Za-z0-9@+\-\[\]\(\)=#%/\.\\]{12,})\b"
    for m in re.finditer(pattern, search_window):
        smi = m.group(1).strip()
        if smi not in seen and basic_validate(smi):
            lower = smi.lower()
            # 排除已知非目标关键词
            skip_words = [
                "available", "synthesize", "synthesis", "procedure", "step",
                "requires", "applied", "reaction", "follow", "example",
                "intermediate", "dissolved", "stirred", "heated", "mmol",
                "equiv", "buffer", "column", "chromatography", "concentrated",
                "washed", "solution", "mixture", "compound", "reagent",
                "tert", "butyl", "butoxycarbonyl", "piperazine", "phenyl",
                "methyl", "methoxy", "ethoxy", "ethoxycarbonyl", "indole",
                "purified", "sodium", "potassium", "magnesium", "soluble",
                "hexane", "acetate", "ether", "chloroform", "palladium",
                "chromium", "copper", "bromide", "chloride", "iodide",
                "reduct", "oxid", "catalyst", "ligand", "reagent",
                "chromene", "quinazoline", "sulfonamide", "nicotinamide",
                "benzamide", "pyridine", "pyridazine", "pyridinyl",
                "benzylidene", "acetonitrile", "dichloromethane",
                "dimethylsulfoxide", "triethylamine", "hydrochloride",
                "mmol", "equivalents", "room temperature", "overnight",
                "reflux", "purification", "extraction", "filtration",
                "concentration", "evaporation", "saturated", "brine",
                "magnesium", "anhydrous", "sulfate", "celite",
                "silica", "gradient", "acetonitrile",
            ]
            if not any(s in lower for s in skip_words):
                try:
                    mol = Chem.MolFromSmiles(smi)
                    mw = Descriptors.MolWt(mol)
                    tp = Descriptors.TPSA(mol)
                    lp = Descriptors.MolLogP(mol)
                    results.append((smi, "greedy_step1", {"mw": mw, "tpsa": tp, "logp": lp}))
                    seen.add(smi)
                except Exception:
                    pass

    return results


def process_experiment(exp_dir: str, output_suffix: str = ""):
    """处理一个实验目录的所有原始响应。"""
    exp_path = Path(exp_dir)
    raw_dir = exp_path / "raw_responses"
    if not raw_dir.exists():
        print(f"  No raw_responses dir in {exp_dir}")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = exp_path / f"extracted{output_suffix}_{ts}"
    out_dir.mkdir(exist_ok=True)
    csv_out = out_dir / "csv_output"
    csv_out.mkdir(exist_ok=True)

    # 收集已有分子
    PROJECT_ROOT = Path("/root/Llamole-main")
    exp_parent = PROJECT_ROOT / "gbm_project" / "experiments"
    existing = {"EGFR": set(), "IDH1_IDH2": set(), "VEGF_VEGFR": set()}
    for exp_p in sorted(exp_parent.iterdir()):
        if not exp_p.is_dir():
            continue
        csv_dir = exp_p / "csv_output"
        if not csv_dir.exists():
            continue
        for cf in csv_dir.glob("*.csv"):
            target_name = cf.stem.replace("_re", "")
            matched = next((t for t in existing if t.lower() in target_name.lower()), None)
            if not matched:
                continue
            try:
                with open(cf, "r", encoding="utf-8", errors="ignore") as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for row in reader:
                        smi = row[3].strip() if len(row) >= 4 else (row[1].strip() if len(row) >= 2 else "")
                        if smi and basic_validate(smi):
                            existing[matched].add(smi)
            except Exception:
                pass

    print(f"\n{'='*60}")
    print(f"处理实验: {exp_dir}")
    print(f"已有分子: EGFR={len(existing['EGFR'])}, IDH1_IDH2={len(existing['IDH1_IDH2'])}, VEGF_VEGFR={len(existing['VEGF_VEGFR'])}")
    print(f"{'='*60}")

    all_results = {}

    for raw_file in sorted(raw_dir.glob("*_raw.json")):
        target = raw_file.stem.replace("_raw", "")
        print(f"\n--- {target} ---")

        with open(raw_file, encoding="utf-8") as f:
            responses = json.load(f)

        seen = set()  # 本次提取去重
        new_mols = []
        dup_mols = []
        total_extracted = 0

        for call_data in responses:
            resp = call_data["response"]
            extracted = extract_from_retrosynthesis(resp)
            total_extracted += len(extracted)

            for smi, strategy, props in extracted:
                if smi in existing[target]:
                    if smi not in dup_mols:
                        dup_mols.append(smi)
                else:
                    if smi not in seen:
                        seen.add(smi)
                        new_mols.append({
                            "smiles": smi,
                            "strategy": strategy,
                            **props
                        })

            strategies = {}
            for _, s, _ in extracted:
                strategies[s] = strategies.get(s, 0) + 1
            print(f"  Call {call_data['call']}: 提取={len(extracted)}, "
                  f"新增={len(new_mols)}, 重复={len(dup_mols)}, "
                  f"策略: {strategies}")

        # 保存 CSV
        csv_path = csv_out / f"{target}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["entity_id", "mol_id", "library_name", "smiles", "MW", "TPSA", "LogP", "target", "strategy"])
            for i, mol in enumerate(new_mols):
                w.writerow([
                    f"llamole_retro_{ts}",
                    f"{target}_{i:05d}",
                    "Llamole-Qwen2-7B-Instruct-Adapter",
                    mol["smiles"],
                    f"{mol['mw']:.2f}",
                    f"{mol['tpsa']:.2f}",
                    f"{mol['logp']:.2f}",
                    target,
                    mol["strategy"],
                ])

        # 统计药物相似性
        druglike = sum(1 for m in new_mols if 150 <= m["mw"] <= 700 and 20 <= m["tpsa"] <= 140)
        print(f"  [{target}] 完成: {len(new_mols)} 新分子, "
              f"药物相似性: {druglike}/{len(new_mols)}, "
              f"总提取: {total_extracted}, 新颖率: {len(new_mols)/max(total_extracted,1)*100:.0f}%")

        all_results[target] = {
            "new": len(new_mols),
            "duplicate_vs_prior": len(dup_mols),
            "total_extracted": total_extracted,
            "druglike": druglike,
            "novelty_rate": f"{len(new_mols) / max(total_extracted, 1) * 100:.1f}%",
        }

    # 保存汇总
    with open(out_dir / "extraction_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "llamole_retro_extract",
            "source_dir": str(exp_dir),
            "timestamp": ts,
            "results": all_results,
            "existing_counts": {t: len(s) for t, s in existing.items()},
        }, f, indent=2, ensure_ascii=False)

    print(f"\n提取完成! 输出: {out_dir}")
    return out_dir


def main():
    import argparse
    parser = argparse.ArgumentParser(description="从逆合成原始输出中提取 SMILES")
    parser.add_argument('--exp_dir', type=str, required=True,
                        help='实验目录（含 raw_responses 子目录）')
    parser.add_argument('--output_suffix', type=str, default="",
                        help='输出目录后缀')
    args = parser.parse_args()

    process_experiment(args.exp_dir, args.output_suffix)


if __name__ == "__main__":
    main()
