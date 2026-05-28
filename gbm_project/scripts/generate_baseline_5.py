#!/usr/bin/env python3
"""
Generate 5 molecules using the original (baseline) model with the same concise prompt,
extract SMILES, evaluate with GBMEvaluator, and save results.
"""
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from gbm_project.src.gbm_knowledge_base import GBMKnowledgeBase
from gbm_project.src.gbm_prompt_generator import GBMPromptGenerator
from gbm_project.improved_smiles_extraction import extract_multiple_smiles
from gbm_project.src.gbm_evaluator import GBMEvaluator

def generate_with_baseline(model, tokenizer, instruction: str, domain_prompt: str):
    prompt = f"Instruction: {instruction}\nInput: {domain_prompt}\nResponse:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    targets_path = os.path.join(base_dir, "data/gbm_targets/gbm_targets.json")
    clinical_path = os.path.join(base_dir, "data/gbm_clinical/gbm_clinical_data.json")
    molecules_path = os.path.join(base_dir, "data/gbm_molecules/gbm_molecules.json")

    kb = GBMKnowledgeBase(targets_path, clinical_path, molecules_path)
    prompts_path = os.path.join(base_dir, "configs/english_gbm_prompts.yaml")
    prompt_gen = GBMPromptGenerator(kb, prompts_path, language="english")

    concise_instruction = "Generate 5 GBM novel drug candidates."
    domain_prompt = prompt_gen.generate_domain_prompt("EGFR", constraints=None)

    # Load baseline model (local)
    model_path = os.path.join(os.path.abspath(os.path.join(base_dir, "..")), "models", "Qwen2-7B-Instruct")
    print("Loading baseline model from", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="right")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)

    raw = generate_with_baseline(model, tokenizer, concise_instruction, domain_prompt)
    smiles = extract_multiple_smiles(raw, max_count=5)

    evaluator = GBMEvaluator()
    results = []
    for smi in smiles:
        res = evaluator.evaluate_molecule(smi)
        results.append(res)

    # pad to 5
    while len(results) < 5:
        results.append({'smiles': None, 'valid': False, 'error': 'no_smiles', 'scores': {'composite_score': 0.0}})

    out_dir = os.path.join(base_dir, "experiments", "baseline_generation_5")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "generation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"instruction": concise_instruction, "domain_prompt": domain_prompt, "raw_response": raw, "results": results}, f, ensure_ascii=False, indent=2)

    print("Saved baseline results to", out_path)

if __name__ == "__main__":
    main()




