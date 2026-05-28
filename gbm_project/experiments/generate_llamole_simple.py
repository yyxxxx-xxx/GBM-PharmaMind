#!/usr/bin/env python3
"""
Llamole + Qwen2-7B-Instruct 对比实验
====================================
使用 HuggingFace 国内镜像源 (hf-mirror.com)，空闲 GPU，
输入简单中文 prompt 生成 GBM 药物分子，信任模型输出，只做最基础的格式清理。

模型组合：
  - Base model: Qwen2-7B-Instruct (本地)
  - Adapter: Llamole-Qwen2-7B-Instruct-Adapter (本地)
"""

import os
import sys
import re
import json
import csv
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Set, Optional

import torch
from rdkit import Chem
from rdkit.Chem import Descriptors

PROJECT_ROOT = Path("/root/Llamole-main")
sys.path.insert(0, str(PROJECT_ROOT))

import warnings
warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "0"


# =============================================================================
# 1. 收集已有实验的 SMILES（去重用）
# =============================================================================

def collect_existing() -> dict:
    """收集之前所有实验的 SMILES，避免重复。"""
    exp_dir = PROJECT_ROOT / "gbm_project" / "experiments"
    existing = {"EGFR": set(), "IDH1_IDH2": set(), "VEGF_VEGFR": set()}

    for exp_path in sorted(exp_dir.iterdir()):
        if not exp_path.is_dir():
            continue
        csv_out = exp_path / "csv_output"
        if not csv_out.exists():
            continue
        for cf in csv_out.glob("*.csv"):
            target_name = cf.stem.replace("_re", "")
            matched = next((t for t in existing if t.lower() in target_name.lower()), None)
            if not matched:
                continue
            try:
                with open(cf, "r", encoding="utf-8", errors="ignore") as f:
                    reader = csv.reader(f)
                    next(reader, None)
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
    """RDKit 验证。"""
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
    """
    从模型输出中提取 SMILES。
    策略：逗号/分号/换行切分，去掉前缀，RDKit 验证，去重。
    """
    results = []
    seen = set()

    for chunk in re.split(r"[,;\n]", text):
        chunk = chunk.strip()
        chunk = re.sub(r"^[\-\u2022\u2023\u25e6\*\u2043]\s*", "", chunk)
        chunk = re.sub(r"^[\d]+[\.\)\:]+\s*", "", chunk)
        chunk = re.sub(r"^[Ss][Mm][Ii][Ll][Ee][Ss]\s*[:\-]?\s*", "", chunk)
        chunk = re.sub(r"[\`\`\`]",
 "", chunk).strip()
        if not chunk or len(chunk) < 5:
            continue
        if not re.search(r"[CNOSPFIcnosp]", chunk):
            continue
        if chunk not in seen and _rdkit_ok(chunk):
            seen.add(chunk)
            results.append(chunk)

    return results


# =============================================================================
# 3. 模型加载与生成
# =============================================================================

def load_model(base_model_path: str, lora_adapter_path: str, device: str):
    """加载 Qwen2-7B-Instruct + Llamole LoRA adapter。"""
    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading tokenizer from {base_model_path}")
    tok = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading base model from {base_model_path}")
    bm = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=device,
    )

    print(f"Loading Llamole LoRA adapter from {lora_adapter_path}")
    m = PeftModel.from_pretrained(
        bm,
        lora_adapter_path,
        torch_dtype=torch.float16,
    )
    m.eval()
    print("Model loaded!")
    return m, tok


# =============================================================================
# 4. 中文 Simple Prompt
# =============================================================================

# =============================================================================
# 4. Simple Prompt（与 Exp8 相同的格式，验证有效）
# =============================================================================

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
        "You are a medicinal chemist. Generate exactly 5 valid SMILES strings for drug molecules "
        "targeting VEGF or VEGFR (Vascular Endothelial Growth Factor/Receptor) for Glioblastoma (GBM).\n\n"
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
}


# =============================================================================
# 5. 主流程
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Llamole + Qwen2-7B-Instruct GBM 对比实验")
    parser.add_argument('--targets', type=str, nargs='+',
                        default=["EGFR", "IDH1_IDH2", "VEGF_VEGFR"],
                        help='靶点列表')
    parser.add_argument('--num_molecules', type=int, default=50,
                        help='每个靶点生成的分子数量 (default: 50)')
    parser.add_argument('--gpu_id', type=int, default=1,
                        help='GPU 设备号 (default: 1)')
    parser.add_argument('--max_calls', type=int, default=10,
                        help='每个靶点最大调用次数 (default: 10)')
    parser.add_argument('--molecules_per_call', type=int, default=5,
                        help='每次调用请求的分子数量 (default: 5)')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='采样温度 (default: 0.7)')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Nucleus 采样 top_p (default: 0.9)')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='最大生成长度 (default: 512)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录 (default: auto)')
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = PROJECT_ROOT / "gbm_project" / "experiments" / f"llamole_simple_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_out = out_dir / "csv_output"
    csv_out.mkdir(exist_ok=True)

    device = f"cuda:{args.gpu_id}"
    torch.cuda.set_device(args.gpu_id)
    torch.cuda.empty_cache()

    base_model_path = str(PROJECT_ROOT / "models" / "Qwen2-7B-Instruct")
    lora_adapter_path = str(PROJECT_ROOT / "saves" / "Llamole-Qwen2-7B-Instruct-Adapter")

    print("=" * 60)
    print("Llamole + Qwen2-7B-Instruct 简单 Prompt 实验")
    print(f"输出目录: {out_dir}")
    print(f"GPU: {device}")
    print(f"靶点: {args.targets}")
    print(f"每靶点分子数: {args.num_molecules}")
    print("=" * 60)

    # 收集已有分子
    print("\n[1] 收集已有实验的分子...")
    existing = collect_existing()
    for t, s in existing.items():
        print(f"  {t}: {len(s)} 个已有分子")

    # 加载模型
    print("\n[2] 加载模型...")
    model, tok = load_model(base_model_path, lora_adapter_path, device)

    # 生成
    print("\n[3] 开始生成分子...")
    all_results = {}

    for target in args.targets:
        prompt = SIMPLE_PROMPTS.get(target)
        if not prompt:
            print(f"  [{target}] 无对应 prompt，跳过")
            continue

        print(f"\n--- {target} ---")
        seen = set()
        new_mols = []
        dup_mols = []
        total_extracted = 0
        num_calls = 0

        while len(new_mols) < args.num_molecules and num_calls < args.max_calls:
            num_calls += 1
            try:
                messages = [{"role": "user", "content": prompt}]
                formatted = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tok(formatted, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        do_sample=True,
                        pad_token_id=tok.pad_token_id,
                        eos_token_id=tok.eos_token_id,
                    )

                input_len = inputs["input_ids"].shape[1]
                resp = tok.decode(out[0][input_len:], skip_special_tokens=True)
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
                print(f"  Call {num_calls}: 提取={len(smis)}, 新增={len(new_mols)}, "
                      f"重复={len(dup_mols)}, 进度={len(new_mols)}/{args.num_molecules}, 新颖率={nov:.0f}%")

            except Exception as e:
                print(f"  Call {num_calls} 错误: {e}")
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
                    lp = Descriptors.MolLogP(mol)
                except Exception:
                    mw = tp = lp = 0.0
                w.writerow([
                    f"llamole_{ts}",
                    f"{target}_{i:05d}",
                    "Llamole-Qwen2-7B-Instruct-Adapter",
                    smi,
                    f"{mw:.2f}", f"{tp:.2f}", f"{lp:.2f}",
                    target,
                ])

        print(f"  [{target}] 完成: {len(new_mols)} 个新分子 (共 {num_calls} 次调用)")
        all_results[target] = {
            "new": len(new_mols),
            "duplicate_vs_prior": len(dup_mols),
            "total_extracted": total_extracted,
            "calls": num_calls,
            "novelty_rate": f"{len(new_mols) / total_extracted * 100:.1f}%" if total_extracted else "N/A",
        }

        torch.cuda.empty_cache()

    # 汇总
    print("\n" + "=" * 60)
    print("实验汇总")
    print("=" * 60)
    for t, s in all_results.items():
        print(f"\n{t}:")
        for k, v in s.items():
            print(f"  {k}: {v}")

    exp_cfg = {
        "experiment": "llamole_simple",
        "description": (
            "Llamole + Qwen2-7B-Instruct-Adapter 简单 prompt 对比实验。"
            "使用 HF 镜像源，空闲 GPU，信任模型输出，只做最基础的格式清理。"
        ),
        "base_model": base_model_path,
        "lora_adapter": lora_adapter_path,
        "hf_mirror": "https://hf-mirror.com",
        "gpu_id": args.gpu_id,
        "targets": args.targets,
        "num_molecules": args.num_molecules,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "timestamp": ts,
        "results": all_results,
        "existing_counts": {t: len(s) for t, s in existing.items()},
    }
    with open(out_dir / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(exp_cfg, f, indent=2, ensure_ascii=False)

    print(f"\n完成! 输出目录: {out_dir}")


if __name__ == "__main__":
    main()
