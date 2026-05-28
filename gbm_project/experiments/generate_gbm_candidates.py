#!/usr/bin/env python3
"""
GBM候选分子生成实验脚本
使用Llamole模型生成GBM靶向候选分子并进行评估
支持 Chain-of-Thought (CoT) 和 Tree-of-Thoughts (ToT) 两种推理模式
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# 确保可以导入主项目和 GBM 子项目代码
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 添加 gbm_project 路径
gbm_project_path = PROJECT_ROOT / "gbm_project"
if str(gbm_project_path) not in sys.path:
    sys.path.insert(0, str(gbm_project_path))

from gbm_project.src.gbm_generator import GBMGenerator


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='GBM候选分子生成实验')

    # 基本参数
    parser.add_argument('--config', type=str, default='../configs/gbm_generation.yaml',
                       help='GBM配置文件路径')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='输出目录')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='实验名称，默认使用时间戳')

    # 生成参数
    parser.add_argument('--num_candidates', type=int, default=20,
                       help='生成候选分子数量')
    parser.add_argument('--target', type=str, default=None,
                       choices=['EGFR', 'VEGF_VEGFR', 'IDH1_IDH2', 'MGMT', 'GBM_Stem_Cells',
                               'PD1_PDL1', 'PI3K_AKT_mTOR', 'p53_MDM2'],
                       help='指定靶点，如果不指定则自动选择')
    parser.add_argument('--use_cot', action='store_true', default=True,
                       help='使用Chain-of-Thought推理（默认模式）')
    parser.add_argument('--no_cot', action='store_false', dest='use_cot',
                       help='不使用Chain-of-Thought推理')
    
    # ToT 参数
    parser.add_argument('--use_tot', action='store_true', default=False,
                       help='使用Tree-of-Thoughts推理（覆盖CoT模式）')
    parser.add_argument('--tot_k', type=int, default=3,
                       help='ToT每层生成的候选数量（默认：3）')
    parser.add_argument('--tot_b', type=int, default=2,
                       help='ToT BFS保留的分支数量（默认：2）')
    parser.add_argument('--tot_depth', type=int, default=3,
                       help='ToT搜索深度（默认：3）')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='使用的GPU设备ID（默认：0）')

    # 约束条件参数
    parser.add_argument('--max_mw', type=float, default=600,
                       help='最大分子量')
    parser.add_argument('--min_logp', type=float, default=1.0,
                       help='最小logP值')
    parser.add_argument('--max_logp', type=float, default=4.0,
                       help='最大logP值')
    parser.add_argument('--max_tpsa', type=float, default=120,
                       help='最大TPSA值')

    # 筛选参数
    parser.add_argument('--top_n', type=int, default=10,
                       help='保留的顶级候选分子数量')
    parser.add_argument('--min_score', type=float, default=0.6,
                       help='最低综合得分阈值')

    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='显示详细输出')

    return parser.parse_args()


def setup_experiment_directory(args):
    """设置实验目录"""
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"gbm_generation_{timestamp}"

    experiment_dir = Path(args.output_dir) / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # 保存实验配置
    config_file = experiment_dir / "experiment_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    return experiment_dir


def main():
    """主函数"""
    args = parse_args()

    if args.verbose:
        print("=" * 60)
        print("GBM候选分子生成实验")
        print("=" * 60)
        print(f"配置文件: {args.config}")
        print(f"生成数量: {args.num_candidates}")
        print(f"指定靶点: {args.target if args.target else '自动选择'}")
        if args.use_tot:
            print(f"推理模式: Tree-of-Thoughts (ToT)")
            print(f"  - ToT深度: {args.tot_depth}")
            print(f"  - 每层候选数(k): {args.tot_k}")
            print(f"  - 保留分支数(b): {args.tot_b}")
        else:
            print(f"推理模式: Chain-of-Thought (CoT)")
            print(f"  - 使用CoT推理: {args.use_cot}")
        print(f"GPU设备: {args.gpu_id}")
        print(f"随机种子: {args.seed}")
        print("-" * 60)

    # 检查配置文件存在
    if not os.path.exists(args.config):
        print(f"错误：配置文件不存在: {args.config}")
        sys.exit(1)

    # 设置实验目录
    experiment_dir = setup_experiment_directory(args)

    try:
        # 设置约束条件
        constraints = {
            'molecular_weight_max': args.max_mw,
            'molecular_weight_min': 300,  # 默认最小分子量
            'logp_min': args.min_logp,
            'logp_max': args.max_logp,
            'tpsa_max': args.max_tpsa
        }
        
        # 根据模式选择生成器
        if args.use_tot:
            # 使用 Tree-of-Thoughts 模式
            if args.verbose:
                print("初始化Tree-of-Thoughts生成器...")
            
            # 导入 ToT 生成器
            from gbm_project.scripts.generate_tot_molecules import TreeOfThoughtsGenerator
            
            # 构建 ToT 配置
            tot_config = {
                'gpu_id': args.gpu_id,
                'base_model_path': str(PROJECT_ROOT / "models" / "Qwen2-7B-Instruct"),
                'llamole_adapter_path': str(PROJECT_ROOT / "saves" / "Llamole-Qwen2-7B-Instruct-Adapter"),
                'gbm_adapter_path': str(PROJECT_ROOT / "saves" / "Llamole-Qwen2-7B-Instruct-Adapter-gbm-with-graph-models"),
                'prompts_config_path': str(PROJECT_ROOT / "gbm_project" / "configs" / "gbm_prompts.yaml"),
                'generation': {
                    'max_new_tokens': 512,
                    'temperature': 0.8,
                    'top_p': 0.95,
                    'do_sample': True
                },
                'tot_depth': args.tot_depth,
                'tot_k': args.tot_k,
                'tot_b': args.tot_b,
                'constraints': {
                    'mw_range': f"{int(constraints.get('molecular_weight_min', 300))}-{int(constraints.get('molecular_weight_max', 600))}",
                    'bbb_requirement': 'high',
                    'logp_range': f"{constraints.get('logp_min', 1.0)}-{constraints.get('logp_max', 4.0)}"
                }
            }
            
            # 初始化 ToT 生成器
            generator = TreeOfThoughtsGenerator(tot_config)
            
            # 加载模型
            if args.verbose:
                print("加载模型...")
            generator.load_models(use_8bit=True)
            
            # 确定靶点
            target_name = args.target if args.target else 'EGFR'
            
            # 生成GBM候选分子（ToT模式）
            if args.verbose:
                print(f"开始使用ToT生成 {args.num_candidates} 个GBM候选分子...")
                print(f"目标靶点: {target_name}")
                print(f"约束条件: {tot_config['constraints']}")
            
            # 加载评估器（如果需要评估）
            evaluator = None
            try:
                from gbm_project.src.gbm_evaluator import GBMEvaluator
                from gbm_project.src.gbm_knowledge_base import GBMKnowledgeBase
                
                kb = GBMKnowledgeBase(
                    targets_path=str(PROJECT_ROOT / "gbm_project" / "data" / "gbm_targets" / "gbm_targets.json"),
                    clinical_path=str(PROJECT_ROOT / "gbm_project" / "data" / "gbm_clinical" / "gbm_clinical_data.json"),
                    molecules_path=str(PROJECT_ROOT / "gbm_project" / "data" / "gbm_molecules" / "gbm_molecules.json")
                )
                reference_molecules = []
                for mol in kb.molecules.values():
                    if mol.status in ['Approved', 'Phase 3', 'Phase 2']:
                        reference_molecules.append(mol.smiles)
                evaluator = GBMEvaluator(reference_molecules[:10])
                if args.verbose:
                    print("评估器加载成功")
            except Exception as e:
                if args.verbose:
                    print(f"警告：无法加载评估器: {e}")
            
            # 转换 ToT 结果为标准格式
            results = []
            seen_smiles = set()  # 避免重复
            
            # 多次运行 ToT 搜索直到获得足够的分子
            max_rounds = max(1, (args.num_candidates // (args.tot_k * args.tot_b)) + 1)
            if args.verbose:
                print(f"将运行最多 {max_rounds} 轮 ToT 搜索以生成足够的分子...")
            
            for round_num in range(max_rounds):
                if len(results) >= args.num_candidates:
                    break
                    
                if args.verbose:
                    print(f"\n--- ToT 搜索轮次 {round_num + 1}/{max_rounds} ---")
                
                # 运行 ToT 搜索
                tot_results = generator.generate_molecules(target_name)
                
                # 转换并评估结果
                for tot_mol in tot_results:
                    smiles = tot_mol.get('smiles', '').strip()
                    if not smiles or smiles in seen_smiles:
                        continue
                    
                    seen_smiles.add(smiles)
                    
                    # 评估分子
                    evaluation = {}
                    if evaluator:
                        try:
                            evaluation = evaluator.evaluate_molecule(smiles)
                        except Exception as e:
                            if args.verbose:
                                print(f"评估分子时出错: {e}")
                    
                    result = {
                        'id': len(results) + 1,
                        'target': target_name,
                        'smiles': smiles,
                        'generation_method': 'tot_bfs',
                        'tot_path': tot_mol.get('tot_path', []),
                        'evaluation': evaluation,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    results.append(result)
                    
                    if len(results) >= args.num_candidates:
                        break
                
                if args.verbose:
                    print(f"当前已生成 {len(results)}/{args.num_candidates} 个分子")
            
            if args.verbose:
                if len(results) < args.num_candidates:
                    print(f"警告：ToT只生成了 {len(results)} 个分子，少于请求的 {args.num_candidates} 个")
                else:
                    print(f"成功生成 {len(results)} 个分子")
        
        else:
            # 使用传统的 CoT 模式
            if args.verbose:
                print("初始化GBM生成器（CoT模式）...")
            generator = GBMGenerator(args.config)

            # 生成GBM候选分子
            if args.verbose:
                print(f"开始生成 {args.num_candidates} 个GBM候选分子...")
                print(f"约束条件: {constraints}")

            results = generator.generate_gbm_molecules(
                num_candidates=args.num_candidates,
                target_name=args.target,
                constraints=constraints,
                use_cot=args.use_cot
            )

        # 保存完整结果
        results_file = experiment_dir / "generated_molecules.json"
        if args.use_tot:
            # ToT 模式：直接保存 JSON
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            if args.verbose:
                print(f"结果已保存到: {results_file}")
        else:
            # CoT 模式：使用生成器的保存方法
            generator.save_results(results, str(results_file))

        # 生成评估报告
        if args.verbose:
            print("生成评估报告...")
        
        if args.use_tot:
            # ToT 模式：手动生成报告
            valid_results = [r for r in results if r.get('evaluation', {}).get('valid', False)]
            scores = [r['evaluation']['scores']['composite_score'] for r in valid_results if 'evaluation' in r and 'scores' in r['evaluation']]
            
            report = {
                'total_generated': len(results),
                'valid_molecules': len(valid_results),
                'success_rate': len(valid_results) / len(results) if results else 0,
                'average_score': float(sum(scores) / len(scores)) if scores else 0,
                'max_score': float(max(scores)) if scores else 0,
                'min_score': float(min(scores)) if scores else 0,
                'high_potential_count': len([s for s in scores if s > 0.7]),
                'target_distribution': {args.target if args.target else 'EGFR': len(results)}
            }
        else:
            # CoT 模式：使用生成器的报告方法
            report = generator.generate_evaluation_report(results)

        # 保存评估报告
        report_file = experiment_dir / "evaluation_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 筛选顶级候选分子
        if args.verbose:
            print(f"筛选顶级候选分子 (top {args.top_n}, min_score {args.min_score})...")

        if args.use_tot:
            # ToT 模式：手动筛选
            valid_results = [r for r in results if r.get('evaluation', {}).get('valid', False)]
            sorted_results = sorted(
                valid_results,
                key=lambda x: x.get('evaluation', {}).get('scores', {}).get('composite_score', 0),
                reverse=True
            )
            top_candidates = [
                r for r in sorted_results
                if r.get('evaluation', {}).get('scores', {}).get('composite_score', 0) >= args.min_score
            ][:args.top_n]
        else:
            # CoT 模式：使用生成器的筛选方法
            top_candidates = generator.filter_top_candidates(
                results, top_n=args.top_n, min_score=args.min_score
            )

        # 保存顶级候选分子
        top_candidates_file = experiment_dir / "top_candidates.json"
        with open(top_candidates_file, 'w', encoding='utf-8') as f:
            json.dump(top_candidates, f, indent=2, ensure_ascii=False)

        # 打印实验总结
        if args.verbose:
            print("\n" + "=" * 60)
            print("实验完成！")
            print("=" * 60)
            print(f"实验目录: {experiment_dir}")
            print(f"生成分子总数: {report['total_generated']}")
            print(f"有效分子数: {report['valid_molecules']}")
            print(f"成功率: {report['success_rate']:.2%}")
            print(f"平均综合得分: {report['average_score']:.3f}")
            print(f"最高得分: {report['max_score']:.3f}")
            print(f"最低得分: {report['min_score']:.3f}")
            print(f"高潜力分子数 (score>0.7): {report['high_potential_count']}")
            print(f"顶级候选分子数: {len(top_candidates)}")
            print("-" * 60)
            print("输出文件:")
            print(f"  - 完整结果: {results_file}")
            print(f"  - 评估报告: {report_file}")
            print(f"  - 顶级候选: {top_candidates_file}")
            print(f"  - 实验配置: {experiment_dir}/experiment_config.json")

            # 显示顶级候选分子
            if top_candidates:
                print("\n顶级候选分子:")
                for i, candidate in enumerate(top_candidates[:5], 1):  # 显示前5个
                    eval_data = candidate.get('evaluation', {})
                    scores = eval_data.get('scores', {}) if isinstance(eval_data, dict) else {}
                    eval_score = scores.get('composite_score', 0.0) if isinstance(scores, dict) else 0.0
                    target = candidate.get('target', 'Unknown')
                    smiles = candidate.get('smiles', 'N/A')
                    method = candidate.get('generation_method', 'cot')
                    print(f"{i}. 目标: {target} | 方法: {method} | 分数: {eval_score:.3f} | SMILES: {smiles[:60]}...")

    except Exception as e:
        print(f"实验过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
