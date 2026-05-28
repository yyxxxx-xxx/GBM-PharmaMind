#!/usr/bin/env python3
"""
快速生成 VEGF_VEGFR + 补充 EGFR/IDH1_IDH2（使用 LoRA adapter）
保存原始输出，事后用 extract_from_retro.py 提取 SMILES
"""

import os, sys, json, re, csv
from pathlib import Path
from datetime import datetime

import torch
from rdkit import Chem
from rdkit.Chem import Descriptors

PROJECT_ROOT = Path("/root/Llamole-main")
sys.path.insert(0, str(PROJECT_ROOT))

import warnings; warnings.filterwarnings("ignore")
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "0"


def load_model(base_model_path, lora_adapter_path, device):
    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, padding_side="right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bm = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map=device)
    m = PeftModel.from_pretrained(bm, lora_adapter_path, torch_dtype=torch.float16)
    m.eval()
    return m, tok


PROMPTS = {
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--targets', type=str, nargs='+', default=["VEGF_VEGFR"])
    parser.add_argument('--max_calls', type=int, default=10)
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = PROJECT_ROOT / "gbm_project" / "experiments" / f"llamole_retro2_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw_responses"
    raw_dir.mkdir(exist_ok=True)

    device = f"cuda:{args.gpu_id}"
    torch.cuda.set_device(args.gpu_id)
    torch.cuda.empty_cache()

    base_model = str(PROJECT_ROOT / "models" / "Qwen2-7B-Instruct")
    lora_adapter = str(PROJECT_ROOT / "saves" / "Llamole-Qwen2-7B-Instruct-Adapter")

    print(f"Loading model on {device}...")
    model, tok = load_model(base_model, lora_adapter, device)
    print("Model ready!")

    print(f"\nTargets: {args.targets}, Max calls: {args.max_calls}")

    for target in args.targets:
        prompt = PROMPTS.get(target)
        if not prompt:
            print(f"  [{target}] No prompt, skip")
            continue

        print(f"\n--- {target} ---")
        responses = []
        for i in range(args.max_calls):
            try:
                messages = [{"role": "user", "content": prompt}]
                formatted = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tok(formatted, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=800, temperature=0.5, top_p=0.9, do_sample=True, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
                input_len = inputs["input_ids"].shape[1]
                resp = tok.decode(out[0][input_len:], skip_special_tokens=True)
                responses.append({"call": i+1, "response": resp})
                print(f"  Call {i+1}: {len(resp)} chars")
            except Exception as e:
                print(f"  Call {i+1} ERROR: {e}")
                continue

        with open(raw_dir / f"{target}_raw.json", "w", encoding="utf-8") as f:
            json.dump(responses, f, ensure_ascii=False, indent=2)

        print(f"  Saved {len(responses)} responses to {raw_dir / f'{target}_raw.json'}")

    # 保存配置
    with open(out_dir / "experiment_config.json", "w") as f:
        json.dump({
            "experiment": "llamole_retro2",
            "targets": args.targets,
            "max_calls": args.max_calls,
            "gpu_id": args.gpu_id,
            "timestamp": ts,
        }, f, indent=2)

    print(f"\nDone! Output: {out_dir}")


if __name__ == "__main__":
    main()
