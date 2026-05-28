#!/usr/bin/env python3
"""
LLaMA-MolInst-Molecule-7B 对比实验 (exp10)
==========================================
使用 HuggingFace 国内镜像源 (hf-mirror.com)，空闲 GPU，
输入极简 prompt，信任模型输出，只做最基础的格式清理。

模型组合：
  - Base model: NousResearch/Llama-2-7b-hf (本地缓存)
  - Adapter: zjunlp/llama-molinst-molecule-7b (from hf-mirror)

关键修复：
  1. 使用 Alpaca prompt 格式（与 Mol-Instructions demo 一致）
  2. 设置正确的 token IDs: bos=1, eos=2, pad=0
  3. 使用 greedy 解码（do_sample=False, num_beams=4）和小 temperature
  4. 支持 atom-bond notation [C][C][O] 格式转换
"""

import os
import sys
import re
import json
import csv
import argparse
from pathlib import Path
from datetime import datetime
from typing import List

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
# 1. Alpaca Prompt Format (与 Mol-Instructions demo 一致)
# =============================================================================

ALPACA_PROMPT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

# 短 prompt 格式 - 直接明确要求 SMILES
INSTRUCTION_TEMPLATES = {
    "EGFR": "Generate SMILES of 5 drug molecules targeting EGFR for glioblastoma.",
    "IDH1_IDH2": "Generate SMILES of 5 drug molecules targeting IDH1/IDH2 for glioblastoma.",
    "VEGF_VEGFR": "Generate SMILES of 5 drug molecules targeting VEGF/VEGFR for glioblastoma.",
}

def make_prompt(target: str) -> str:
    """使用简短格式的 prompt"""
    instr = INSTRUCTION_TEMPLATES.get(target, f"Generate SMILES for {target} targeting molecules.")
    return instr


# =============================================================================
# 2. 收集已有实验的 SMILES（去重用）
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
    """严格的 RDKit 验证 - 必须能正常解析为有效分子。"""
    if not smi or len(smi) > 1000:
        return False
    # 包含明显非分子文本的拒绝
    garbage_words = ["in conclusion", "i think", "thank", "however", "therefore", "moreover",
                     "previous", "in my opinion", "in addition", "the molecule is",
                     "instructions", "please", "response", "answer", "helpful", "hope",
                     "you must", "your response", "as a side", "this has nothing",
                     "let me", "i cannot", "i will", "it was the first"]
    lower = smi.lower()
    for word in garbage_words:
        if word in lower[:100]:
            return False
    try:
        mol = Chem.MolFromSmiles(smi)  # 默认 sanitize=True
        if mol is None:
            return False
        # 检查原子组成 - 至少要有一定比例的化学原子
        atom_chars = set("BCNOPSFIbcnosp[]()=#@+-0123456789")
        chem_chars = sum(1 for c in smi if c in atom_chars)
        if chem_chars / len(smi) < 0.5:
            return False
        return True
    except Exception:
        return False


# =============================================================================
# 3. Atom-bond Notation 转 SMILES
# =============================================================================

def _bracket_to_smiles(bracket_text: str) -> str:
    """
    将 [C][C][O] 格式转换为标准 SMILES。
    这是 Mol-Instructions 使用的原子-键记号法。
    """
    result = bracket_text
    replacements = [
        # 原子
        ("[C]", "C"), ("[c]", "c"),
        ("[N]", "N"), ("[n]", "n"),
        ("[O]", "O"), ("[o]", "o"),
        ("[S]", "S"), ("[s]", "s"),
        ("[P]", "P"), ("[p]", "p"),
        ("[F]", "F"), ("[I]", "I"),
        ("[B]", "B"),
        # 二元卤素
        ("[Br]", "Br"), ("[Cl]", "Cl"),
        # 显式价态
        ("[N+1]", "N+"), ("[N-1]", "N-"),
        ("[O+1]", "O+"), ("[O-1]", "O-"),
        # 显式氢
        ("[H]", ""), ("[H2]", ""), ("[H3]", ""),
        # 常见基团
        ("[OH]", "O"), ("[NH2]", "N"), ("[NH]", "N"),
        # 立体化学
        ("[C@]", "C@"), ("[C@@]", "C@@"), ("[C@H]", "C@H"),
        ("[C@@H]", "C@@H"), ("[C@H1]", "C@H"),
        # 键合标记
        ("[=C]", "C"), ("[=N]", "N"), ("[=O]", "O"), ("[=S]", "S"),
        ("[#C]", "C"), ("[#N]", "N"),
        # 特殊标记
        ("[Ring1]", ""), ("[Ring2]", ""),
        ("[Ring3]", ""),
        ("[Branch1]", ""), ("[Branch2]", ""), ("[Branch3]", ""),
        ("[=Branch1]", ""), ("[=Branch2]", ""), ("[#Branch1]", ""),
        ("[#Branch2]", ""),
        ("[se]", "Se"), ("[Se]", "Se"),
        ("[Si]", "Si"),
    ]
    for old, new in replacements:
        result = result.replace(old, new)
    return result


# =============================================================================
# 4. 提取 SMILES（正则精确匹配 + 严格验证）
# =============================================================================

def extract_smiles(text: str) -> List[str]:
    """
    从模型输出中提取 SMILES。
    策略：用正则精确匹配 SMILES 模式，拒绝文本描述。
    支持标准 SMILES 和 atom-bond notation [C][C][O]。
    """
    results = []
    seen = set()

    # 去掉 prompt 部分（如果被 decode 进来）
    if "### Response:" in text:
        text = text.split("### Response:")[-1]
    # 去掉 </s> 等特殊 token
    text = re.sub(r"</?s>", "", text)
    text = re.sub(r"<.*?>", "", text)

    # 方法1: 用正则精确匹配 atom-bond notation 片段
    # 模式: [...] 序列，通常以 ] 结尾
    bracket_pattern = re.compile(r'\[[^\]]{1,6}\]')
    bracket_chunks = bracket_pattern.findall(text)
    bracket_joined = "".join(bracket_chunks)
    if len(bracket_joined) >= 5:
        converted = _bracket_to_smiles(bracket_joined)
        if converted and converted not in seen and _rdkit_ok(converted):
            seen.add(converted)
            results.append(converted)

    # 方法2: 在文本中搜索独立的 SMILES 行/片段
    # 标准 SMILES 通常是独立的文本段，不包含空格或普通单词
    lines = re.split(r'[,;\n]', text)
    for line in lines:
        line = line.strip()
        # 跳过包含明显文本描述的行
        lower = line.lower()
        skip_words = ["the molecule", "this molecule", "i think", "in conclusion",
                      "in my opinion", "in addition", "however", "therefore",
                      "thank", "previous", "response", "instructions", "please",
                      "you must", "your response", "i hope", "i cannot",
                      "let me", "it was", "moreover", "as a side",
                      "first time", "helpful", "hopeful", "answer",
                      "conclusion", "final", "citation", "note", "indeed",
                      "kindness", "person", "meet", "enjoy", "together",
                      "equipped", "bothered", "work out", "glad", "completed",
                      "completed", "inhibitor", "drug candidate"]
        if any(w in lower[:60] for w in skip_words):
            continue

        # 跳过太短的行
        if len(line) < 4:
            continue

        # 跳过纯文本（包含太多普通字母）
        text_ratio = sum(1 for c in line if c.isalpha()) / len(line)
        if text_ratio > 0.6:
            continue

        # 必须是化学字符
        if not re.search(r"[CNOSPFIcnosp\[\]]", line):
            continue

        # 去掉行首前缀
        line = re.sub(r"^[\-\u2022\u2023\u25e6\*\u2043]\s*", "", line)
        line = re.sub(r"^[\d]+[\.\)\:]+\s*", "", line)
        line = re.sub(r"^[Ss][Mm][Ii][Ll][Ee][Ss]\s*[:\-]?\s*", "", line)
        line = re.sub(r"```+", "", line).strip()

        if len(line) < 4 or line in seen:
            continue

        # 严格的 RDKit 验证
        if _rdkit_ok(line):
            seen.add(line)
            results.append(line)
            continue

        # 尝试 atom-bond notation
        if "][" in line:
            converted = _bracket_to_smiles(line)
            if converted and converted not in seen and _rdkit_ok(converted):
                seen.add(converted)
                results.append(converted)

    return results


def post_process_response(text: str) -> str:
    """清理模型输出中的特殊 token 和 prompt 残留。"""
    # 去掉 </s> 等特殊 token
    text = re.sub(r"</?s>", "", text)
    text = re.sub(r"<pad>", "", text)
    text = re.sub(r"<.*?\n", "", text)
    # 去掉末尾的 # 符号
    text = text.replace("#", "")
    return text.strip()


# =============================================================================
# 5. 模型加载与生成
# =============================================================================

def load_model(base_model_path: str, lora_adapter_path: str, gpu_id: int):
    """
    加载 LLaMA v1 (decapoda) + llama-molinst-molecule-7b LoRA adapter。

    关键修复（与 Mol-Instructions demo 一致）：
    1. 使用 device_map={"": 0} 指定 GPU
    2. 设置 bos_token_id=1, eos_token_id=2, pad_token_id=0
    3. 使用 fp16，不使用量化
    4. 不 merge LoRA，保持 PeftModel（demo 方式）
    """
    import gc
    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModelForCausalLM

    gc.collect()
    torch.cuda.empty_cache()

    print(f"Loading tokenizer from {base_model_path}")
    from transformers.models.llama.tokenization_llama import LlamaTokenizer
    tok = LlamaTokenizer.from_pretrained(
        base_model_path,
        tokenizer_file=None,
    )
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    print(f"  Token IDs: bos={tok.bos_token_id}, eos={tok.eos_token_id}, pad={tok.pad_token_id}")

    print(f"Loading base model from {base_model_path} (fp16, GPU {gpu_id})")
    bm = AutoModelForCausalLM.from_pretrained(
        base_model_path, trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map={"": gpu_id},
    )

    gc.collect()
    torch.cuda.empty_cache()

    print(f"Loading llama-molinst-molecule-7b LoRA adapter from {lora_adapter_path}")
    m = PeftModel.from_pretrained(
        bm, lora_adapter_path,
        device_map={"": gpu_id},
    )
    m.eval()

    gc.collect()
    torch.cuda.empty_cache()

    print("Model loaded!")
    return m, tok


# =============================================================================
# 6. 主流程
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LLaMA-MolInst-Molecule-7B GBM 对比实验 (exp10)")
    parser.add_argument('--targets', type=str, nargs='+',
                        default=["EGFR", "IDH1_IDH2", "VEGF_VEGFR"],
                        help='靶点列表')
    parser.add_argument('--num_molecules', type=int, default=50,
                        help='每个靶点生成的分子数量 (default: 50)')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU 设备号 (default: 0)')
    parser.add_argument('--max_calls', type=int, default=20,
                        help='每个靶点最大调用次数 (default: 20)')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='采样温度 (default: 0.1, demo使用0.1)')
    parser.add_argument('--top_p', type=float, default=0.75,
                        help='Nucleus 采样 top_p (default: 0.75)')
    parser.add_argument('--top_k', type=int, default=40,
                        help='Top-k 采样 (default: 40)')
    parser.add_argument('--num_beams', type=int, default=4,
                        help='Beam 数量 (default: 4, demo使用4)')
    parser.add_argument('--repetition_penalty', type=float, default=1.0,
                        help='重复惩罚 (default: 1.0)')
    parser.add_argument('--max_new_tokens', type=int, default=256,
                        help='最大生成长度 (default: 256)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录 (default: auto)')
    parser.add_argument('--base_model', type=str, default=None,
                        help='Llama base model 路径或 HuggingFace ID')
    parser.add_argument('--do_sample', action='store_true',
                        help='使用采样模式 (default: False, 使用beam search)')
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = PROJECT_ROOT / "gbm_project" / "experiments" / f"exp10_llamamolinst_simple_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_out = out_dir / "csv_output"
    csv_out.mkdir(exist_ok=True)

    device = f"cuda:{args.gpu_id}"
    torch.cuda.empty_cache()

    # 模型路径：使用本地缓存的 snapshot 路径
    # 重要：llama-molinst-molecule-7b LoRA 是在 decapoda-research/llama-7b-hf (LLaMA v1) 上训练的
    llama_snapshot_v1 = PROJECT_ROOT / "models" / "models--baffo32--decapoda-research-llama-7B-hf" / "snapshots" / "main"
    llama_snapshot_v2 = PROJECT_ROOT / "models" / "models--NousResearch--Llama-2-7b-hf" / "snapshots" / "8efe6c9b93655b934e27bd9981e3ec13e55aee9d"
    molinst_snapshot = PROJECT_ROOT / "models" / "models--zjunlp--llama-molinst-molecule-7b" / "snapshots" / "b147ec6e64e6e4f70284e6c24ad24bfe3d60f8fb"

    if args.base_model:
        base_model_path = args.base_model
    elif llama_snapshot_v1.exists():
        base_model_path = str(llama_snapshot_v1)
        print("  [INFO] Using LLaMA v1 (decapoda) for LoRA compatibility")
    elif llama_snapshot_v2.exists():
        base_model_path = str(llama_snapshot_v2)
        print("  [WARNING] LLaMA v2 detected - LoRA may not be compatible!")
    else:
        base_model_path = "baffo32/decapoda-research-llama-7B-hf"

    lora_adapter_path = str(molinst_snapshot)

    print("=" * 60)
    print("Experiment 10: LLaMA-MolInst-Molecule-7B")
    print("Output dir: " + str(out_dir))
    print("GPU: " + device)
    print("Targets: " + str(args.targets))
    print("Num molecules per target: " + str(args.num_molecules))
    print("Base model: " + base_model_path)
    print("Adapter: " + lora_adapter_path)
    print("=" * 60)

    # 收集已有分子
    print("\n[1] Collecting existing molecules...")
    existing = collect_existing()
    for t, s in existing.items():
        print(f"  {t}: {len(s)} existing")

    # 加载模型
    print("\n[2] Loading model...")
    model, tok = load_model(base_model_path, lora_adapter_path, args.gpu_id)

    # 生成
    print("\n[3] Starting generation...")
    all_results = {}

    for target in args.targets:
        prompt_text = make_prompt(target)
        print(f"\n--- {target} ---")

        # 编码 prompt
        inputs = tok(prompt_text, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        input_len = input_ids.shape[1]

        seen = set()
        new_mols = []
        dup_mols = []
        total_extracted = 0
        num_calls = 0

        while len(new_mols) < args.num_molecules and num_calls < args.max_calls:
            num_calls += 1
            try:
                with torch.no_grad():
                    generation_output = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        num_beams=args.num_beams,
                        repetition_penalty=args.repetition_penalty,
                        do_sample=args.do_sample,
                        pad_token_id=tok.pad_token_id,
                        bos_token_id=tok.bos_token_id,
                        eos_token_id=tok.eos_token_id,
                    )

                # 解码生成的部分（不包含输入 prompt）
                input_len = input_ids.shape[1]
                generated_ids = generation_output[0][input_len:]
                resp = tok.decode(generated_ids, skip_special_tokens=True)

                if num_calls <= 3:
                    preview = resp[:400].replace('\n', '\\n')
                    print(f"    [DEBUG call {num_calls}] resp={preview}")

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
                      f"dup={len(dup_mols)}, progress={len(new_mols)}/{args.num_molecules}, novelty={nov:.0f}%")

            except Exception as e:
                print(f"  Call {num_calls} error: {e}")
                import traceback
                traceback.print_exc()
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
                    f"llamamolinst_{ts}",
                    f"{target}_{i:05d}",
                    "LLaMA-MolInst-Molecule-7B",
                    smi,
                    f"{mw:.2f}", f"{tp:.2f}", f"{lp:.2f}",
                    target,
                ])

        print(f"  [{target}] Done: {len(new_mols)} new molecules ({num_calls} calls)")
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
    print("Summary")
    print("=" * 60)
    for t, s in all_results.items():
        print(f"\n{t}:")
        for k, v in s.items():
            print(f"  {k}: {v}")

    exp_cfg = {
        "experiment": "exp10_llamamolinst_simple",
        "description": (
            "LLaMA-MolInst-Molecule-7B 对比实验。"
            "使用 Alpaca prompt 格式，atom-bond notation 输出，"
            "与 Mol-Instructions demo 保持一致。"
        ),
        "base_model": base_model_path,
        "lora_adapter": lora_adapter_path,
        "hf_mirror": "https://hf-mirror.com",
        "gpu_id": args.gpu_id,
        "targets": args.targets,
        "num_molecules": args.num_molecules,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "num_beams": args.num_beams,
        "repetition_penalty": args.repetition_penalty,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "timestamp": ts,
        "results": all_results,
        "existing_counts": {t: len(s) for t, s in existing.items()},
    }
    with open(out_dir / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(exp_cfg, f, indent=2, ensure_ascii=False)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "exp10_llamamolinst_simple",
            "model": "LLaMA-MolInst-Molecule-7B (LLaMA-2-7b-hf + zjunlp/llama-molinst-molecule-7b)",
            "timestamp": ts,
            "results": all_results,
            "existing_counts": {t: len(s) for t, s in existing.items()},
        }, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Output dir: {out_dir}")


if __name__ == "__main__":
    main()
