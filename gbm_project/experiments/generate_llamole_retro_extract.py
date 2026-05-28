#!/usr/bin/env python3
"""
从逆合成规划文本中提取 SMILES 分子式
======================================
Llamole LoRA adapter 生成的是逆合成规划文本，其中包含目标分子 SMILES。
本脚本：
1. 重新运行 Llamole 实验，保存原始模型输出
2. 从逆合成规划中提取目标分子 SMILES
"""

import os
import re
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Set, Tuple

import torch
from rdkit import Chem
from rdkit.Chem import Descriptors

PROJECT_ROOT = Path("/root/Llamole-main")
sys_path = str(PROJECT_ROOT)
if sys_path not in os.environ.get("PYTHONPATH", ""):
    os.environ["PYTHONPATH"] = sys_path + ":" + os.environ.get("PYTHONPATH", "")

import warnings
warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "0"


# =============================================================================
# 逆合成文本中提取 SMILES 的策略
# =============================================================================

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


def _basic_validate(smi: str) -> bool:
    """宽松验证（用于初步提取阶段）。"""
    if len(smi) < 5 or len(smi) > 600:
        return False
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        if mol is None:
            return False
        Chem.SanitizeMol(mol, catchErrors=True)
        return True
    except Exception:
        return False


def extract_smiles_from_retrosynthesis(text: str) -> List[Tuple[str, str]]:
    """
    从逆合成规划文本中提取目标分子 SMILES。
    返回: [(smiles, context), ...]
    策略：
      1. "To synthesize XXX" 后面是目标分子，提取 SMILES
      2. "This is step N" 前面是中间体/起始原料，忽略
      3. "(available)" / "(not available)" 前面是试剂/构建块，忽略
      4. "SMILES N:" 格式直接提取
      5. 裸 SMILES 行提取
    """
    results = []
    seen = set()

    text = text.strip()

    # 策略1：查找 "To synthesize <name>" 模式
    # 目标分子通常在 "To synthesize XXX" 或 "For the synthesis of XXX" 后面
    target_patterns = [
        r"To\s+synthesize\s+([^\n]+?)(?:\s*[,;.]|\s*\(|\s*\n|$)",
        r"for\s+the\s+synthesis\s+of\s+([^\n]+?)(?:\s*[,;.]|\s*\(|\s*\n|$)",
        r"target\s+molecule\s+is\s+([^\n]+?)(?:\s*[,;.]|\s*\(|\s*\n|$)",
        r"final\s+product\s+is\s+([^\n]+?)(?:\s*[,;.]|\s*\(|\s*\n|$)",
    ]

    for pattern in target_patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            candidate = m.group(1).strip()
            # 尝试从候选文本中提取 SMILES
            smis = _extract_from_text_segment(candidate)
            for s in smis:
                if s not in seen and _basic_validate(s):
                    seen.add(s)
                    results.append((s, f"from_target_phrase:{candidate[:30]}"))

    # 策略2："SMILES N:" 格式
    for m in re.finditer(r"SMILES\s*\d*\s*:\s*([^\n]+)", text, re.IGNORECASE):
        smi = m.group(1).strip()
        smi = re.sub(r"\s+", "", smi).strip(" ,.;")
        if smi and smi not in seen and _basic_validate(smi):
            seen.add(smi)
            results.append((smi, "from_smiles_label"))

    # 策略3：逆合成步骤中的目标分子（step 1 / step 2 之前的分子）
    # 逆合成通常先给出目标分子，再列出合成步骤
    step_markers = [
        "This is step", "Step", "step", "synthesis process",
        "retrosynthes", " retrosyn",
    ]
    lines = text.split("\n")
    for i, line in enumerate(lines):
        line_clean = line.strip()
        if not line_clean:
            continue
        # 跳过包含 step 的行（这些是中间步骤，不是目标分子）
        if any(marker in line_clean for marker in step_markers):
            continue
        # 跳过包含 available/not available 的行（试剂/构建块）
        if re.search(r"\(available\)|\(not\s+available\)", line_clean, re.IGNORECASE):
            continue
        # 跳过纯文字描述行
        if not re.search(r"[CNOSPFIcnosp]", line_clean):
            continue
        # 提取行中的 SMILES 部分
        smis = _extract_from_text_segment(line_clean)
        for s in smis:
            if s not in seen and _basic_validate(s):
                seen.add(s)
                results.append((s, f"from_step_line:{line_clean[:40]}"))

    # 策略4：提取括号前的完整 SMILES
    # "Nc1ccccc1Cl (available)" -> "Nc1ccccc1Cl"
    for m in re.finditer(r"([CNOSPFI][A-Za-z0-9@+\-\[\]\(\)=#%/\.\\]+)\s*\(", text):
        smi = m.group(1).strip()
        if smi and smi not in seen and _basic_validate(smi):
            seen.add(smi)
            results.append((smi, "from_paren_prefix"))

    # 策略5：贪婪提取所有 SMILES-like 片段
    # 在已知有效 SMILES 之后，查找更多
    pattern_smiles = r"\b([CNOSPFI][A-Za-z0-9@+\-\[\]\(\)=#%/\.\\]{8,})\b"
    for m in re.finditer(pattern_smiles, text):
        smi = m.group(1).strip()
        if smi and smi not in seen and _basic_validate(smi):
            # 排除已知的非目标关键词
            lower = smi.lower()
            skip_words = [
                "chain", "analysis", "reasoning", "scaffold",
                "strategy", "rationale", "synthetic", "procedure",
                "therefore", "suggest", "step", "synthes",
                "available", "example", "reference",
            ]
            if not any(w in lower for w in skip_words):
                seen.add(smi)
                results.append((smi, "from_greedy_pattern"))

    return results


def _extract_from_text_segment(segment: str) -> List[str]:
    """从文本片段中提取 SMILES。"""
    results = []

    # 去掉括号及后面内容
    segment = re.sub(r"\s*\([^)]*\)", "", segment)

    # 去掉 "To synthesize XXX" 或 "for the synthesis of XXX" 开头
    segment = re.sub(r"^(to\s+synthesize|for\s+the\s+synthesis\s+of|target\s+molecule|final\s+product)\s+", "", segment, flags=re.IGNORECASE)

    # 去掉末尾标点
    segment = segment.strip().rstrip(",;.")

    if not segment:
        return results

    # 检查整个片段是否是 SMILES
    if _basic_validate(segment):
        results.append(segment)
        return results

    # 按空格/逗号/分号切分
    for chunk in re.split(r"[\s,;]+", segment):
        chunk = chunk.strip()
        if not chunk:
            continue
        if _basic_validate(chunk):
            results.append(chunk)

    return results


def extract_from_raw_csv_line(smiles_field: str) -> List[str]:
    """
    从已保存的脏 SMILES 字段中重新提取。
    例如: "O=C(Oc1ccc(-c2nc(C(=O)O)c3ccccc23)cc1)c1cccs1 (not available,"
    """
    text = smiles_field
    results = []

    # 去掉括号内容
    text = re.sub(r"\s*\([^)]*\)", "", text)

    # 去掉 "This is step N" 等
    text = re.sub(r"This\s+is\s+step\s+\d+[^,;.]*[,;.]?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"step\s+\d+[^,;.]*[,;.]?", "", text, flags=re.IGNORECASE)

    if _basic_validate(text.strip()):
        results.append(text.strip())
        return results

    for chunk in re.split(r"[,;\n]", text):
        chunk = chunk.strip()
        if _basic_validate(chunk):
            results.append(chunk)

    return results


# =============================================================================
# 模型加载与生成
# =============================================================================

def load_model(base_model_path: str, lora_adapter_path: str, device: str):
    """加载 Qwen2-7B-Instruct + Llamole LoRA adapter。"""
    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading tokenizer from {base_model_path}")
    tok = AutoTokenizer.from_pretrained(
        base_model_path, trust_remote_code=True, padding_side="right"
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading base model from {base_model_path}")
    bm = AutoModelForCausalLM.from_pretrained(
        base_model_path, trust_remote_code=True,
        torch_dtype=torch.float16, device_map=device,
    )

    print(f"Loading Llamole LoRA adapter from {lora_adapter_path}")
    m = PeftModel.from_pretrained(bm, lora_adapter_path, torch_dtype=torch.float16)
    m.eval()
    print("Model loaded!")
    return m, tok


# =============================================================================
# Simple Prompt（与 Exp8 相同）
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
# 收集已有分子
# =============================================================================

def collect_existing() -> dict:
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
                        smi = row[3].strip() if len(row) >= 4 else (row[1].strip() if len(row) >= 2 else "")
                        if smi and _rdkit_ok(smi):
                            existing[matched].add(smi)
            except Exception:
                pass
    return existing


# =============================================================================
# 主流程
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="从逆合成规划中提取 SMILES")
    parser.add_argument('--targets', type=str, nargs='+',
                        default=["EGFR", "IDH1_IDH2", "VEGF_VEGFR"])
    parser.add_argument('--num_molecules', type=int, default=50)
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--max_calls', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--max_new_tokens', type=int, default=800)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = PROJECT_ROOT / "gbm_project" / "experiments" / f"llamole_retro_extract_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_out = out_dir / "csv_output"
    csv_out.mkdir(parents=True, exist_ok=True)

    # 保存原始输出的目录
    raw_out = out_dir / "raw_responses"
    raw_out.mkdir(exist_ok=True)

    device = f"cuda:{args.gpu_id}"
    torch.cuda.set_device(args.gpu_id)
    torch.cuda.empty_cache()

    base_model_path = str(PROJECT_ROOT / "models" / "Qwen2-7B-Instruct")
    lora_adapter_path = str(PROJECT_ROOT / "saves" / "Llamole-Qwen2-7B-Instruct-Adapter")

    print("=" * 60)
    print("Llamole LoRA: 从逆合成规划中提取 SMILES")
    print(f"输出目录: {out_dir}")
    print(f"GPU: {device}")
    print("=" * 60)

    # 收集已有分子
    print("\n[1] 收集已有分子...")
    existing = collect_existing()
    for t, s in existing.items():
        print(f"  {t}: {len(s)} 个已有分子")

    # 加载模型
    print("\n[2] 加载模型...")
    model, tok = load_model(base_model_path, lora_adapter_path, device)

    # 生成 + 提取
    print("\n[3] 生成分子并提取 SMILES...")
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

        # 保存该靶点的原始响应
        raw_responses = []

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

                # 保存原始响应
                raw_responses.append({
                    "call": num_calls,
                    "response": resp,
                })

                # 从原始响应中提取 SMILES
                extracted = extract_smiles_from_retrosynthesis(resp)
                total_extracted += len(extracted)

                for smi, ctx in extracted:
                    if smi in existing[target]:
                        if smi not in dup_mols:
                            dup_mols.append(smi)
                    else:
                        if smi not in seen:
                            seen.add(smi)
                            new_mols.append((smi, ctx))

                nov = len(new_mols) / total_extracted * 100 if total_extracted else 0
                print(f"  Call {num_calls}: 提取={len(extracted)}, 新增={len(new_mols)}, "
                      f"重复={len(dup_mols)}, 进度={len(new_mols)}/{args.num_molecules}, "
                      f"新颖率={nov:.0f}%")

                # 打印前两个响应的摘要
                if num_calls <= 2:
                    preview = resp[:200].replace("\n", " | ")
                    print(f"    原始输出预览: {preview}...")

            except Exception as e:
                print(f"  Call {num_calls} 错误: {e}")
                continue

        # 保存原始响应
        with open(raw_out / f"{target}_raw.json", "w", encoding="utf-8") as f:
            json.dump(raw_responses, f, ensure_ascii=False, indent=2)

        # 保存 CSV（带来源标注）
        csv_path = csv_out / f"{target}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["entity_id", "mol_id", "library_name", "smiles", "MW", "TPSA", "LogP", "target", "source"])
            for i, (smi, ctx) in enumerate(new_mols):
                try:
                    mol = Chem.MolFromSmiles(smi)
                    mw = Descriptors.MolWt(mol)
                    tp = Descriptors.TPSA(mol)
                    lp = Descriptors.MolLogP(mol)
                except Exception:
                    mw = tp = lp = 0.0
                w.writerow([
                    f"llamole_retro_{ts}",
                    f"{target}_{i:05d}",
                    "Llamole-Qwen2-7B-Instruct-Adapter",
                    smi,
                    f"{mw:.2f}", f"{tp:.2f}", f"{lp:.2f}",
                    target,
                    ctx,
                ])

        print(f"  [{target}] 完成: {len(new_mols)} 个新分子 (共 {num_calls} 次调用)")
        print(f"    原始响应已保存至: {raw_out / f'{target}_raw.json'}")
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
        "experiment": "llamole_retro_extract",
        "description": (
            "从 Llamole LoRA 逆合成规划中提取目标分子 SMILES。"
            "使用 HF 镜像源，空闲 GPU，信任模型输出。"
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
    import sys
    main()
