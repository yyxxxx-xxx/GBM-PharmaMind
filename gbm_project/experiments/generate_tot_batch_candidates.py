#!/usr/bin/env python3
"""
GBM ToT批量候选分子生成实验脚本
==================================
功能特性：
1. 领域知识prompt注入：通过GBMPromptGenerator自动注入GBM靶点专业知识
2. Tree-of-Thoughts (ToT) 三层搜索：
   - Level 0: Scaffold Proposal（骨架提议）
   - Level 1: Assembly Strategy（组装策略）
   - Level 2: SMILES Generation（SMILES生成）
3. 使用微调模型：
   - 基座模型：models/Qwen2-7B-Instruct
   - Llamole适配器：saves/Llamole-Qwen2-7B-Instruct-Adapter
   - GBM领域适配器：saves/Llamole-Qwen2-7B-Instruct-Adapter-gbm-with-graph-models

使用方法：
    python generate_tot_batch_candidates.py --targets EGFR IDH1_IDH2 --num_molecules 50 --gpu_id 3
    python generate_tot_batch_candidates.py  # 生成所有靶点，每个50个分子
"""

import os
import sys
import json
import argparse
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import logging

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入ToT生成器
sys.path.insert(0, str(PROJECT_ROOT / "gbm_project" / "scripts"))
from generate_tot_molecules import TreeOfThoughtsGenerator, ToTNode


def load_targets() -> List[str]:
    """加载所有可用的靶点"""
    targets_file = PROJECT_ROOT / "gbm_project" / "data" / "gbm_targets" / "gbm_targets.json"
    with open(targets_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    targets = []
    for target_data in data['gbm_targets']:
        target_name = target_data['name']
        # 标准化靶点名称（处理斜杠和空格）
        if '/' in target_name:
            target_name = target_name.replace('/', '_')
        if ' ' in target_name:
            target_name = target_name.replace(' ', '_')
        targets.append(target_name)
    
    return targets


def generate_molecules_for_target(
    generator: TreeOfThoughtsGenerator,
    target_name: str,
    num_molecules: int = 50,
    k: int = 3,
    b: int = 2,
    max_attempts: int = 200,
    max_no_new_streak: int = 50,
    checkpoint_callback=None  # 新增：每轮尝试后调用，用于实时保存进度
) -> List[Dict[str, Any]]:
    """
    为单个靶点生成指定数量的分子
    
    Args:
        generator: ToT生成器实例
        target_name: 靶点名称
        num_molecules: 需要生成的分子数量
        k: 每层生成的候选数
        b: BFS保留的分支数
        checkpoint_callback: 每轮尝试后的回调，签名: callback(all_molecules, target_name, attempts)
    
    Returns:
        生成的分子列表
    """
    logger.info(f"开始为靶点 {target_name} 生成 {num_molecules} 个分子...")
    
    all_molecules = []
    attempts = 0
    no_new_streak = 0
    
    # 更新生成器的k和b参数
    generator.k = k
    generator.b = b

    # 关键优化：预加载并缓存 prompt_generator（避免每次attempt重复加载KB/YAML）
    generator.load_prompt_generator(target_name)
    
    while len(all_molecules) < num_molecules and attempts < max_attempts:
        try:
            # 执行ToT搜索
            molecules = generator.generate_molecules(target_name)
            
            # 去重（基于SMILES）
            existing_smiles = {mol['smiles'] for mol in all_molecules}
            new_molecules = [
                mol for mol in molecules 
                if mol['smiles'] not in existing_smiles
            ]
            
            all_molecules.extend(new_molecules)
            attempts += 1

            if len(new_molecules) == 0:
                no_new_streak += 1
            else:
                no_new_streak = 0
            
            logger.info(
                f"  尝试 {attempts}: 生成 {len(molecules)} 个分子, "
                f"新增 {len(new_molecules)} 个, 总计 {len(all_molecules)}/{num_molecules}"
            )

            # 每轮尝试后写入 checkpoint（防止进程中途终止丢数据）
            if checkpoint_callback is not None:
                checkpoint_callback(list(all_molecules), target_name, attempts)

            # 早停：连续多次无新增，说明ToT在该target上解析/验证/去重后产出接近0
            # 注意：为了满足“每靶点 num_molecules”的 batch 需求，这个阈值默认调大；
            # 如需更省算力，可通过 CLI 降低 max_no_new_streak（<=0 表示禁用早停）。
            if max_no_new_streak > 0 and no_new_streak >= max_no_new_streak:
                logger.error(
                    f"  Early stop: {no_new_streak} consecutive attempts produced 0 new SMILES for {target_name}. "
                    "Likely causes: prompt format mismatch, overly strict validation, or model degeneration to repeated outputs."
                )
                break
            
            # 如果已经达到目标数量，提前退出
            if len(all_molecules) >= num_molecules:
                break
                
        except Exception as e:
            logger.error(f"  尝试 {attempts + 1} 时出错: {e}")
            attempts += 1
            continue
    
    # 只返回前num_molecules个
    result = all_molecules[:num_molecules]
    logger.info(f"靶点 {target_name} 完成: 生成 {len(result)} 个分子")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='使用ToT方法批量生成GBM候选分子'
    )
    
    parser.add_argument(
        '--targets',
        type=str,
        nargs='+',
        default=None,
        help='指定靶点列表，如果不指定则生成所有靶点'
    )
    parser.add_argument(
        '--num_molecules',
        type=int,
        default=50,
        help='每个靶点生成的分子数量（默认: 50）'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=3,
        help='ToT每层生成的候选数（默认: 3）'
    )
    parser.add_argument(
        '--b',
        type=int,
        default=2,
        help='ToT BFS保留的分支数（默认: 2）'
    )
    parser.add_argument(
        '--gpu_id',
        type=int,
        default=3,
        help='GPU设备ID（默认: 3）'
    )
    parser.add_argument(
        '--use_8bit',
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否使用 8-bit 量化加载模型以节省显存（默认: False）"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='输出目录，默认使用时间戳'
    )
    parser.add_argument(
        '--prompts_config_path',
        type=str,
        default=str(PROJECT_ROOT / "gbm_project" / "configs" / "english_gbm_prompts.yaml"),
        help="Prompt 配置 YAML 路径（默认: english_gbm_prompts.yaml）"
    )

        # 护栏回退开关：按当前需求默认开启（可通过 --no-enable_guardrail_fallback 关闭）
    parser.add_argument(
        '--enable_guardrail_fallback',
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to enable guardrail fallback when all branches are 'impossible' (default: False — disabled for production)"
    )
    parser.add_argument(
        '--enable_default_strategy_fallback',
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to enable default strategy fallback when parsing fails (default: False — disabled for production)"
    )

    # 批量生成预算：为了尽量跑满 num_molecules，可调大尝试次数并弱化早停
    parser.add_argument(
        '--max_attempts',
        type=int,
        default=200,
        help="单个靶点最多尝试多少次 ToT search（默认: 200）"
    )
    parser.add_argument(
        '--max_no_new_streak',
        type=int,
        default=0,
        help="连续多少次无新增 SMILES 触发早停（<=0 表示禁用早停；默认: 0，即不禁用）"
    )
    
    args = parser.parse_args()
    
    # 确定要生成的靶点列表
    if args.targets:
        targets = args.targets
    else:
        targets = load_targets()
        logger.info(f"加载到 {len(targets)} 个靶点: {targets}")
    
    # 设置输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "gbm_project" / "experiments" / f"tot_batch_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")
    
    # 配置ToT生成器
    # 注意：此配置已集成以下功能：
    # 1. 领域知识prompt注入：通过GBMPromptGenerator.generate_domain_prompt()实现
    # 2. Tree-of-Thoughts功能：通过TreeOfThoughtsGenerator.bfs_tot_search()实现三层BFS搜索
    # 3. 微调模型：使用Llamole适配器 + GBM领域适配器
    config = {
        'gpu_id': args.gpu_id,
        'base_model_path': str(PROJECT_ROOT / "models" / "Qwen2-7B-Instruct"),  # 本地基座模型
        'llamole_adapter_path': str(PROJECT_ROOT / "saves" / "Llamole-Qwen2-7B-Instruct-Adapter"),
        'gbm_adapter_path': str(PROJECT_ROOT / "saves" / "Llamole-Qwen2-7B-Instruct-Adapter-gbm-with-graph-models"),  # 微调后的GBM模型
        'prompts_config_path': str(args.prompts_config_path),
        # 护栏回退（由 TreeOfThoughtsGenerator 消费）
        'enable_guardrail_fallback': bool(args.enable_guardrail_fallback),
        'enable_default_strategy_fallback': bool(args.enable_default_strategy_fallback),
        'generation': {
            'max_new_tokens': 400,
            'temperature': 0.5,  # 降低temperature以提高格式一致性
            'top_p': 0.9,
            'do_sample': True
        },
        # ToT 搜索预算（避免“长时间无产出”的极端浪费）
        # 参考 tree-of-thought-llm-master：评估应短、可批量；并对单次搜索设置预算上限
            'tot_step_max_new_tokens': {
            'scaffold': 200,
            'assembly': 280,
            'smiles': 450,
            'evaluate': 60,
            'vote': 100
        },
        'tot_max_llm_calls_per_search': 50,
        'tot_max_search_seconds': 300,
        'tot_depth': 3,  # ToT搜索深度：Scaffold -> Assembly -> SMILES
        'tot_k': args.k,  # 每层生成的候选数
        'tot_b': args.b,  # BFS保留的分支数
        'constraints': {
            'mw_range': '300-500',
            'bbb_requirement': 'high',
            'logp_range': '2.0-4.0',
            'tpsa_range': '40-120'  # Topological Polar Surface Area (Å²), must be > 0 for BBB permeability
        }
    }
    
    # 初始化生成器
    logger.info("初始化ToT生成器...")
    generator = TreeOfThoughtsGenerator(config)
    
    # 加载模型（只加载一次）
    logger.info("加载模型...")
    # 先清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"GPU memory before loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved() / 1024**3:.2f} GB reserved")
    # 暂时禁用8-bit量化，因为与PeftModel适配器加载存在兼容性问题
    # 使用GPU 0/1/2（有24GB空闲内存）应该足够加载FP16模型
    generator.load_models(use_8bit=bool(args.use_8bit))
    
    # 保存实验配置
    experiment_config = {
        'targets': targets,
        'num_molecules_per_target': args.num_molecules,
        'tot_k': args.k,
        'tot_b': args.b,
        'gpu_id': args.gpu_id,
        'prompts_config_path': str(args.prompts_config_path),
        'enable_guardrail_fallback': bool(args.enable_guardrail_fallback),
        'enable_default_strategy_fallback': bool(args.enable_default_strategy_fallback),
        'max_attempts': int(args.max_attempts),
        'max_no_new_streak': int(args.max_no_new_streak),
        'timestamp': datetime.now().isoformat()
    }
    
    config_file = output_dir / "experiment_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(experiment_config, f, indent=2, ensure_ascii=False)
    
    # 准备 checkpoint 回调：每轮尝试后立即写入，防止中途终止丢数据
    def checkpoint_write_fn(molecules_so_far, target_name, attempts):
        """每轮尝试后写入 checkpoint（追加到 all_results）"""
        all_results[target_name] = molecules_so_far
        checkpoint_file = output_dir / "all_results_checkpoint.json"
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        # 同时保存 csv
        _save_csv_output(molecules_so_far, target_name, output_dir)
    
    def _save_csv_output(molecules, target_name, out_dir):
        """将当前已有分子写入 CSV（追加模式）"""
        csv_dir = out_dir / "csv_output"
        csv_dir.mkdir(exist_ok=True)
        csv_file = csv_dir / f"{target_name}.csv"
        existing = set()
        if csv_file.exists():
            with open(csv_file) as f:
                for line in f.readlines()[1:]:  # 跳过 header
                    if line.strip():
                        parts = line.strip().split(',', 2)
                        if len(parts) >= 2:
                            existing.add(parts[1])
        lines = []
        for mol in molecules:
            if mol.get('smiles', '') not in existing:
                smiles = mol.get('smiles', '')
                tpsa = mol.get('tpsa', 0)
                mw = mol.get('mw', 0)
                lines.append(f"{target_name}_{mol.get('id', 0):05d},{smiles},{tpsa:.2f},{mw:.2f},{target_name}\n")
                existing.add(smiles)
        if lines:
            with open(csv_file, 'a') as f:
                f.writelines(lines)
    
    # 为每个靶点生成分子
    all_results = {}
    summary_stats = {}
    
    for target in targets:
        logger.info("=" * 80)
        logger.info(f"处理靶点: {target}")
        logger.info("=" * 80)
        
        try:
            molecules = generate_molecules_for_target(
                generator=generator,
                target_name=target,
                num_molecules=args.num_molecules,
                k=args.k,
                b=args.b,
                max_attempts=args.max_attempts,
                max_no_new_streak=args.max_no_new_streak,
                checkpoint_callback=checkpoint_write_fn
            )

            all_results[target] = molecules

            # 统计信息
            summary_stats[target] = {
                'total_generated': len(molecules),
                'unique_smiles': len(set(m['smiles'] for m in molecules)),
                'avg_path_length': sum(len(m.get('tot_path', [])) for m in molecules) / len(molecules) if molecules else 0
            }

            # 保存单个靶点的结果（靶点完成后立即保存）
            target_output_dir = output_dir / target
            target_output_dir.mkdir(exist_ok=True)
            generator.save_results(molecules, target, target_output_dir)

            # 同时将当前进度追加写入总文件（防止进程中途崩溃丢数据）
            checkpoint_file = output_dir / "all_results_checkpoint.json"
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

            logger.info(f"✓ 靶点 {target} 完成: {len(molecules)} 个分子")
            
        except Exception as e:
            logger.error(f"✗ 靶点 {target} 生成失败: {e}")
            import traceback
            traceback.print_exc()
            all_results[target] = []
            summary_stats[target] = {
                'total_generated': 0,
                'error': str(e)
            }
            continue
    
    # 保存汇总结果
    summary_file = output_dir / "summary.json"
    summary_data = {
        'experiment_config': experiment_config,
        'summary_stats': summary_stats,
        'total_targets': len(targets),
        'total_molecules': sum(len(mols) for mols in all_results.values()),
        'targets_breakdown': {
            target: len(molecules) 
            for target, molecules in all_results.items()
        }
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    # 保存所有结果到一个文件
    all_results_file = output_dir / "all_results.json"
    with open(all_results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 打印最终总结
    logger.info("\n" + "=" * 80)
    logger.info("实验完成！")
    logger.info("=" * 80)
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"处理的靶点数: {len(targets)}")
    logger.info(f"总生成分子数: {summary_data['total_molecules']}")
    logger.info("\n各靶点统计:")
    for target, stats in summary_stats.items():
        logger.info(f"  {target}: {stats.get('total_generated', 0)} 个分子")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

