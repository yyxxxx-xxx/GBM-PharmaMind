#!/usr/bin/env python3
"""
Generate 10 GBM drug candidates using market drug reference CoT
with corrected LoRA adapter and strict SMILES formatting.
"""
import os
import sys
import json
import yaml
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.gbm_lora_finetuner import GBMLoRAFinetuner
from src.gbm_prompt_generator import GBMPromptGenerator
from src.gbm_knowledge_base import GBMKnowledgeBase
from src.gbm_evaluator import GBMEvaluator
from improved_smiles_extraction import extract_multiple_smiles


def extract_numbered_smiles(text: str) -> list:
    """Extract SMILES from numbered list format (1. SMILES, 2. SMILES, etc.)"""
    smiles_list = []

    # Pattern for numbered SMILES: "1. SMILES" or "1) SMILES" or "1: SMILES"
    patterns = [
        r'(\d+)[\.\)\:]\s*([A-Za-z0-9@+\-\[\]\(\)=#]+)',
        r'SMILES\s*\d+[\.\)\:]\s*([A-Za-z0-9@+\-\[\]\(\)=#]+)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple):
                smiles = match[1]  # Second group is the SMILES
            else:
                smiles = match

            # Clean and validate
            smiles = smiles.strip()
            if validate_smiles_quick(smiles):
                smiles_list.append(smiles)

    # Remove duplicates while preserving order
    seen = set()
    unique_smiles = []
    for s in smiles_list:
        if s not in seen:
            seen.add(s)
            unique_smiles.append(s)

    return unique_smiles


def validate_smiles_quick(smiles: str) -> bool:
    """Quick SMILES validation"""
    if not smiles or len(smiles) < 3 or len(smiles) > 200:
        return False
    # Basic checks
    if any(char in smiles for char in [' ', '\n', '\t']):
        return False
    # Must contain some organic atoms
    if not any(atom in smiles.upper() for atom in ['C', 'N', 'O', 'S', 'P']):
        return False
    return True


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Model paths
    base_model_path = os.path.join(os.path.abspath(os.path.join(base_dir, "..")), "models", "Qwen2-7B-Instruct")
    llamole_adapter_path = os.path.join(os.path.abspath(os.path.join(base_dir, "..")), "saves", "Llamole-Qwen2-7B-Instruct-Adapter")
    gbm_adapter_path = os.path.join(os.path.abspath(os.path.join(base_dir, "..")), "saves", "Llamole-Qwen2-7B-Instruct-Adapter-gbm-evaluation-corrected")

    print(f"Base model: {base_model_path}")
    print(f"Llamole adapter: {llamole_adapter_path}")
    print(f"GBM adapter: {gbm_adapter_path}")

    # Load model
    finetuner = GBMLoRAFinetuner(base_model_path, {})
    model, tokenizer = finetuner.load_finetuned_model(gbm_adapter_path, llamole_adapter_path)

    # Load knowledge base
    kb = GBMKnowledgeBase(
        os.path.join(base_dir, "data/gbm_targets/gbm_targets.json"),
        os.path.join(base_dir, "data/gbm_clinical/gbm_clinical_data.json"),
        os.path.join(base_dir, "data/gbm_molecules/gbm_molecules.json")
    )

    # Load prompts
    with open(os.path.join(base_dir, "configs/gbm_prompts.yaml"), 'r', encoding='utf-8') as f:
        prompts = yaml.safe_load(f)
    cot_template = prompts.get('cot_reasoning_templates', {}).get('step_by_step_design', '')

    # Build domain prompt
    prompt_gen = GBMPromptGenerator(kb, os.path.join(base_dir, "configs/gbm_prompts.yaml"))
    domain_prompt = prompt_gen.generate_domain_prompt("EGFR", constraints={
        'molecular_weight_max': 500,
        'logp_min': 2.0,
        'logp_max': 4.0,
        'target_bbb': 'high',
        'target_selectivity': 'EGFRvIII > WT-EGFR >10-fold'
    })

    # Strict instruction for numbered SMILES list
    simple_instruction = """Generate exactly 10 GBM drug candidates based on market drug structural patterns.

Output format: Number each candidate as:
1. SMILES_string
2. SMILES_string
...
10. SMILES_string

Requirements:
- Molecular weight: 300-500 Da (like Erlotinib/Afatinib)
- BBB-permeable scaffolds: quinazoline, pyrimidine, indole
- Selective warheads: acrylamide, cyanoacrylamide
- Only output the numbered SMILES list, no explanations."""

    # Combine prompts
    final_prompt = f"{cot_template}\n\n{domain_prompt}\n\n{simple_instruction}"

    print(f"Prompt length: {len(final_prompt)} characters")
    print("Generating candidates...")

    # Generate response
    raw_response = finetuner.generate_with_finetuned_model(model, tokenizer, simple_instruction, final_prompt)

    print("\n" + "="*80)
    print("MODEL RESPONSE:")
    print("="*80)
    print(raw_response[:2000])
    if len(raw_response) > 2000:
        print("... [truncated]")
    print("="*80)

    # Extract SMILES using multiple methods
    smiles_list = []

    # Method 1: Extract from numbered list
    numbered_smiles = extract_numbered_smiles(raw_response)
    smiles_list.extend(numbered_smiles)

    # Method 2: Fallback to general extraction
    if len(smiles_list) < 5:
        print("Using fallback SMILES extraction...")
        general_smiles = extract_multiple_smiles(raw_response, max_count=15)
        for s in general_smiles:
            if s not in smiles_list:
                smiles_list.append(s)

    # Limit to 10 and validate
    validated_smiles = []
    for s in smiles_list[:10]:
        if validate_smiles_quick(s):
            validated_smiles.append(s)

    print(f"\nExtracted {len(smiles_list)} total, validated {len(validated_smiles)} SMILES")
    print("Validated SMILES:", validated_smiles)

    # Evaluate with GBMEvaluator
    evaluator = GBMEvaluator()
    results = []

    for i, smi in enumerate(validated_smiles):
        print(f"\nEvaluating candidate {i+1}: {smi}")
        try:
            res = evaluator.evaluate_molecule(smi)
            results.append(res)
            print(f"  Valid: {res.get('valid', False)}")
            if res.get('valid'):
                scores = res.get('scores', {})
                print(".3f")
        except Exception as e:
            print(f"  Evaluation failed: {e}")
            results.append({
                'smiles': smi,
                'valid': False,
                'error': str(e),
                'scores': {'composite_score': 0.0}
            })

    # Pad to 10 if needed
    while len(results) < 10:
        results.append({
            'smiles': None,
            'valid': False,
            'error': 'no_smiles_extracted',
            'scores': {'composite_score': 0.0}
        })

    # Format unified results
    unified_results = []
    for r in results:
        if r.get('valid', False):
            unified_results.append({
                'smiles': r['smiles'],
                'valid': True,
                'composite_score_unified': r['scores'].get('composite_score', 0.0),
                'evaluator_scores': r['scores'],
                'evaluator_properties': r.get('properties', {}),
                'evaluator_assessment': r.get('assessment', '')
            })
        else:
            unified_results.append({
                'smiles': r.get('smiles'),
                'valid': False,
                'composite_score_unified': 0.0,
                'error': r.get('error', 'unknown_error')
            })

    # Save results
    out_dir = os.path.join(base_dir, "experiments", "cot_market_reference_generation_10")
    os.makedirs(out_dir, exist_ok=True)

    output_data = {
        "experiment_config": {
            "model": "LoRA-finetuned (gbm-evaluation-corrected)",
            "base_model": base_model_path,
            "adapter": gbm_adapter_path,
            "cot_enabled": True,
            "cot_template": "step_by_step_design_with_market_reference",
            "target_count": 10,
            "target": "EGFR",
            "reference_drugs": ["Erlotinib", "Gefitinib", "Afatinib", "Osimertinib"],
            "constraints": {
                'molecular_weight_range': '300-500 Da',
                'logp_range': '2.0-4.0',
                'target_bbb': 'high',
                'target_selectivity': 'EGFRvIII > WT-EGFR >10-fold'
            }
        },
        "instruction": simple_instruction,
        "full_prompt": final_prompt,
        "raw_response": raw_response,
        "results": unified_results
    }

    out_path = os.path.join(out_dir, "generation_results_unified.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved results to {out_path}")

    # Summary
    valid_count = sum(1 for r in unified_results if r.get('valid'))
    avg_score = sum(r['composite_score_unified'] for r in unified_results if r.get('valid')) / valid_count if valid_count > 0 else 0.0
    max_score = max((r['composite_score_unified'] for r in unified_results if r.get('valid')), default=0.0)

    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Valid molecules: {valid_count}/10")
    print(".3f")
    print(".3f")
    print(".3f")
    print(f"Reference drugs used: Erlotinib, Gefitinib, Afatinib, Osimertinib")
    print(f"CoT strategy: Market drug structural reference")


if __name__ == "__main__":
    main()
