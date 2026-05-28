#!/usr/bin/env python3
"""
使用完整Llamole模型（包含图模型）生成GBM药物候选分子
使用CoT思维链 + 统一评估器打分
"""

import os
import sys
import json
import yaml
import torch

# Add both gbm_project and Llamole-main root to path
sys.path.insert(0, '/root/Llamole-main')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gbm_project.src.gbm_lora_finetuner import GBMLoRAFinetuner
from gbm_project.src.gbm_prompt_generator import GBMPromptGenerator
from gbm_project.src.gbm_knowledge_base import GBMKnowledgeBase
from gbm_project.src.gbm_evaluator import GBMEvaluator
from gbm_project.improved_smiles_extraction import extract_multiple_smiles, validate_smiles, debug_smiles_extraction


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # 模型路径 - 使用完整集成图模型的LoRA适配器
    base_model_path = os.path.join(os.path.abspath(os.path.join(base_dir, "..")), "models", "Qwen2-7B-Instruct")
    llamole_adapter_path = os.path.join(os.path.abspath(os.path.join(base_dir, "..")), "saves", "Llamole-Qwen2-7B-Instruct-Adapter")
    gbm_adapter_path = os.path.join(os.path.abspath(os.path.join(base_dir, "..")), "saves", "Llamole-Qwen2-7B-Instruct-Adapter-gbm-with-graph-models")

    # 图模型路径
    graph_decoder_path = os.path.join(os.path.abspath(os.path.join(base_dir, "..")), "saves", "graph_decoder")
    graph_encoder_path = os.path.join(os.path.abspath(os.path.join(base_dir, "..")), "saves", "graph_encoder")
    graph_predictor_path = os.path.join(os.path.abspath(os.path.join(base_dir, "..")), "saves", "graph_predictor")
    graph_lm_connector_path = os.path.join(os.path.abspath(os.path.join(base_dir, "..")), "saves", "Llamole-Qwen2-7B-Instruct-Adapter", "connector")

    print("=" * 80)
    print("完整Llamole模型GBM药物生成实验")
    print("=" * 80)
    print(f"基础模型: {base_model_path}")
    print(f"Llamole适配器: {llamole_adapter_path}")
    print(f"GBM适配器: {gbm_adapter_path}")
    print(f"图模型: ✓ GraphDiT + GraphCLIP + GraphPredictor")

    # 创建finetuner实例，启用图模型支持
    finetuner = GBMLoRAFinetuner(
        base_model_path=base_model_path,
        lora_config={},
        use_llamole_graph_models=True,
        graph_decoder_path=graph_decoder_path,
        graph_encoder_path=graph_encoder_path,
        graph_predictor_path=graph_predictor_path,
        graph_lm_connector_path=graph_lm_connector_path
    )

    # 加载完整模型
    print("\n🔄 加载完整Llamole模型...")
    model, tokenizer = finetuner.load_finetuned_model(gbm_adapter_path, llamole_adapter_path)
    print("✅ 模型加载完成")
    # 加载知识库和提示生成器
    kb = GBMKnowledgeBase(
        os.path.join(base_dir, "data/gbm_targets/gbm_targets.json"),
        os.path.join(base_dir, "data/gbm_clinical/gbm_clinical_data.json"),
        os.path.join(base_dir, "data/gbm_molecules/gbm_molecules.json")
    )

    prompt_gen = GBMPromptGenerator(kb, os.path.join(base_dir, "configs/gbm_prompts.yaml"))

    # 加载CoT模板
    with open(os.path.join(base_dir, "configs/gbm_prompts.yaml"), 'r', encoding='utf-8') as f:
        prompts = yaml.safe_load(f)
    cot_template = prompts.get('cot_reasoning_templates', {}).get('step_by_step_design', '')

    # 构建领域提示
    domain_prompt = prompt_gen.generate_domain_prompt("EGFR", constraints={
        'molecular_weight_max': 500,
        'logp_min': 2.0,
        'logp_max': 4.0,
        'target_bbb': 'high',
        'target_selectivity': 'EGFRvIII > WT-EGFR >10-fold'
    })

    # 简单生成指令
    simple_instruction = "Generate 50 GBM novel drug candidates. Output each candidate as 'SMILES: <smiles>' on a separate line."

    # 强制编号输出格式
    output_format = """
Output format: Number each candidate as:
1. SMILES_string
2. SMILES_string
...
50. SMILES_string

Requirements:
- Molecular weight: 300-500 Da (like Erlotinib/Afatinib)
- BBB-permeable scaffolds: quinazoline, pyrimidine, indole
- Selective warheads: acrylamide, cyanoacrylamide
- Only output the numbered SMILES list, no explanations."""

    # 组合完整提示
    final_prompt = cot_template + "\n\n" + domain_prompt + "\n\n" + simple_instruction + output_format

    print(f"\n📝 提示长度: {len(final_prompt)} 字符")
    print("🎯 使用CoT思维链: ✓")
    print("🔬 图模型集成: ✓")
    print("🎨 强制编号输出: ✓")
    # 生成分子
    print("\n🚀 开始生成...")
    raw_response = finetuner.generate_with_finetuned_model(model, tokenizer, simple_instruction, final_prompt)

    print("\n" + "="*60)
    print("🤖 模型原始输出:")
    print("="*60)
    print(raw_response[:2000])
    if len(raw_response) > 2000:
        print("... [已截断]")
    print("="*60)

    # 提取并记录 CoT 思维链（模型推理依据）
    import re
    cot_text = ""
    # 尝试通过数字列表起始位置切分 CoT 与 SMILES 列表
    list_start_match = re.search(r'\n\s*1[\.\)\:]\s', raw_response)
    if list_start_match:
        cot_text = raw_response[:list_start_match.start()].strip()
    else:
        # 如果没有明确编号，尝试以 "Chain-of-Thought" 或 "Chain of Thought" 为切分点
        cot_marker = re.search(r'Chain[- ]of[- ]Thought[:：]?', raw_response, re.IGNORECASE)
        if cot_marker:
            # 取从 marker 到列表或结尾的文本作为 CoT（过长时保留前4000字符）
            cot_text = raw_response[cot_marker.start():].strip()[:4000]
        else:
            cot_text = ""

    # 使用改进的 SMILES 提取器提取候选（优先编号列表，再回退到通用提取）
    smiles_list = []
    # 优先尝试编号列表提取
    numbered_pattern = r'(\d+)[\.\)\:]\s*([A-Za-z0-9@+\-\[\]\(\)=#\/\\+]+)'
    numbered_matches = re.findall(numbered_pattern, raw_response)
    for match in numbered_matches:
        cand = match[1].strip()
        if validate_smiles(cand):
            smiles_list.append(cand)

    # 若编号提取不足，回退到改进提取器（最多提取15个候选）
    if len(smiles_list) < 5:
        general_smiles = extract_multiple_smiles(raw_response, max_count=50)
        for s in general_smiles:
            if s not in smiles_list and validate_smiles(s):
                smiles_list.append(s)

    # 去重并限制为50个（由prompt控制）
    smiles_list = list(dict.fromkeys(smiles_list))[:50]

    print(f"\n🔍 从模型输出提取到 {len(smiles_list)} 个 RDKit 验证通过的 SMILES:")
    for i, s in enumerate(smiles_list, 1):
        print(f"  {i}. {s}")

    # 评估分子
    evaluator = GBMEvaluator()
    results = []

    print("\n🧪 仅对 RDKit 验证通过的分子进行评估...")
    for i, smi in enumerate(smiles_list):
        print(f"  评估分子 {i+1}: {smi}")
        try:
            res = evaluator.evaluate_molecule(smi)
            results.append(res)
            scores = res.get('scores', {})
            print(f"    评分: {scores.get('composite_score', 0.0):.3f}")
        except Exception as e:
            print(f"    ❌ 评估失败: {e}")
            results.append({
                'smiles': smi,
                'valid': False,
                'error': str(e),
                'scores': {'composite_score': 0.0}
            })

    # 对未通过RDKit验证或未生成的编号位置，补充失败记录以保持结果长度为50
    while len(results) < 50:
        results.append({
            'smiles': None,
            'valid': False,
            'error': 'no_smiles_generated_or_invalid',
            'scores': {'composite_score': 0.0}
        })

    # 格式化结果
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

    # 保存结果
    out_dir = os.path.join(base_dir, "experiments", "full_llamole_cot_generation_50")
    os.makedirs(out_dir, exist_ok=True)

    output_data = {
        "experiment_config": {
            "model": "完整Llamole模型 + GBM LoRA微调",
            "base_model": base_model_path,
            "llamole_adapter": llamole_adapter_path,
            "gbm_adapter": gbm_adapter_path,
            "graph_models": {
                "decoder": graph_decoder_path,
                "encoder": graph_encoder_path,
                "predictor": graph_predictor_path,
                "connectors": graph_lm_connector_path
            },
            "cot_enabled": True,
            "cot_template": "step_by_step_design_with_market_reference",
            "target_count": 50,
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
        "cot_chain": cot_text,
        "results": unified_results
    }

    out_path = os.path.join(out_dir, "generation_results_unified.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n💾 结果已保存至: {out_path}")

    # 统计和总结
    valid_count = sum(1 for r in unified_results if r.get('valid'))
    avg_score = sum(r['composite_score_unified'] for r in unified_results if r.get('valid')) / valid_count if valid_count > 0 else 0.0
    max_score = max((r['composite_score_unified'] for r in unified_results if r.get('valid')), default=0.0)

    print(f"\n" + "="*60)
    print("📊 实验总结")
    print("="*60)
    print(f"🎯 目标分子数: 50")
    print(f"✅ 有效分子数: {valid_count}/50")
    print(".3f")
    print(".3f")
    print(".3f")
    print("🔬 模型特性:")
    print("  • 完整Llamole架构 (语言模型 + 图模型)")
    print("  • GBM专用LoRA微调")
    print("  • 基于市面药物参考的CoT思维链")
    print("  • 强制编号SMILES输出格式")
    print("🚀 关键改进:")
    print("  • 集成了GraphDiT、GraphCLIP、GraphPredictor")
    print("  • 在预训练Llamole适配器基础上微调")
    print("  • 使用市面成功药物结构知识")
    print("  • 优化的分子生成提示和格式")


def validate_smiles_quick(smiles: str) -> bool:
    """快速SMILES验证"""
    if not smiles or len(smiles) < 3 or len(smiles) > 200:
        return False
    if any(char in smiles for char in [' ', '\n', '\t']):
        return False
    if not any(atom in smiles.upper() for atom in ['C', 'N', 'O', 'S', 'P']):
        return False
    return True


if __name__ == "__main__":
    main()
