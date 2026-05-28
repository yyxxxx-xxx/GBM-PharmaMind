#!/usr/bin/env python3
"""
GBM候选分子评估分析脚本
分析和可视化GBM分子生成结果
"""

import os
import sys
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

# 添加项目路径到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from src.gbm_evaluator import GBMEvaluator


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='GBM候选分子评估分析')

    parser.add_argument('--results_file', type=str, required=True,
                       help='生成的分子结果文件路径')
    parser.add_argument('--output_dir', type=str, default='./analysis',
                       help='分析输出目录')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='实验名称，默认从结果文件推断')

    # 分析参数
    parser.add_argument('--generate_plots', action='store_true', default=True,
                       help='生成可视化图表')
    parser.add_argument('--detailed_analysis', action='store_true', default=True,
                       help='进行详细分析')

    return parser.parse_args()


def load_results(results_file: str) -> List[Dict[str, Any]]:
    """加载结果文件"""
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results


def create_analysis_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """创建分析用的DataFrame"""
    data = []

    for result in results:
        if not result.get('valid', False):
            continue

        row = {
            'id': result.get('id', 'unknown'),
            'target': result.get('target', 'EGFR'),
            'smiles': result.get('smiles'),
            'composite_score': result['evaluator_scores']['composite_score'],
            'bbb_permeability': result['evaluator_scores']['bbb_permeability'],
            'gbm_activity': result['evaluator_scores']['gbm_activity'],
            'normal_cell_toxicity': result['evaluator_scores']['normal_cell_toxicity'],
            'selectivity_index': result['evaluator_scores']['selectivity_index'],
            'synthetic_accessibility': result['evaluator_scores']['synthetic_accessibility'],
            'clinical_similarity': result['evaluator_scores']['clinical_similarity'],
            'bbb_classification': result['evaluator_scores']['bbb_classification']
        }

        # 添加分子性质
        if 'evaluator_properties' in result:
            props = result['evaluator_properties']
            row.update({
                'molecular_weight': props.get('molecular_weight'),
                'logp': props.get('logp'),
                'tpsa': props.get('tpsa'),
                'hbd': props.get('hbd'),
                'hba': props.get('hba'),
                'rotatable_bonds': props.get('rotatable_bonds'),
                'ring_count': props.get('ring_count'),
                'qed': props.get('qed')
            })

        data.append(row)

    return pd.DataFrame(data)


def generate_summary_report(df: pd.DataFrame, output_dir: Path) -> Dict[str, Any]:
    """生成汇总报告"""
    report = {
        'total_molecules': len(df),
        'targets_covered': df['target'].nunique(),
        'target_distribution': df['target'].value_counts().to_dict(),
        'score_statistics': {
            'composite_score': {
                'mean': df['composite_score'].mean(),
                'std': df['composite_score'].std(),
                'min': df['composite_score'].min(),
                'max': df['composite_score'].max(),
                'median': df['composite_score'].median()
            }
        },
        'high_potential_molecules': len(df[df['composite_score'] > 0.7]),
        'good_bbb_molecules': len(df[df['bbb_classification'] == 'high']),
        'top_candidates': []
    }

    # 添加各项得分统计
    score_columns = ['bbb_permeability', 'gbm_activity', 'normal_cell_toxicity',
                    'selectivity_index', 'synthetic_accessibility', 'clinical_similarity']

    for col in score_columns:
        if col in df.columns:
            report['score_statistics'][col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max()
            }

    # 找出顶级候选分子
    top_candidates = df.nlargest(10, 'composite_score')
    for _, row in top_candidates.iterrows():
        candidate_info = {
            'id': row['id'],
            'target': row['target'],
            'smiles': row['smiles'],
            'composite_score': row['composite_score'],
            'bbb_classification': row['bbb_classification'],
            'selectivity_index': row['selectivity_index']
        }
        report['top_candidates'].append(candidate_info)

    # 保存报告
    report_file = output_dir / "analysis_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def generate_visualizations(df: pd.DataFrame, output_dir: Path):
    """生成可视化图表"""
    # 设置matplotlib参数
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    sns.set_style("whitegrid")

    # 1. Composite Score Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['composite_score'], bins=20, kde=True)
    plt.title('GBM Candidate Molecules Composite Score Distribution')
    plt.xlabel('Composite Score')
    plt.ylabel('Number of Molecules')
    plt.axvline(df['composite_score'].mean(), color='red', linestyle='--',
               label='.2f')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "composite_score_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Target Distribution
    plt.figure(figsize=(12, 6))
    target_counts = df['target'].value_counts()
    sns.barplot(x=target_counts.index, y=target_counts.values)
    plt.title('Target Distribution of Candidate Molecules')
    plt.xlabel('Target')
    plt.ylabel('Number of Molecules')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "target_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. BBB Permeability vs Composite Score Scatter Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='bbb_permeability', y='composite_score',
                   hue='bbb_classification', palette='viridis')
    plt.title('BBB Permeability vs Composite Score')
    plt.xlabel('BBB Permeability (logBB)')
    plt.ylabel('Composite Score')
    plt.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='High Potential Threshold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "bbb_vs_score_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 4. 各项得分相关性热力图
    score_cols = ['composite_score', 'bbb_permeability', 'gbm_activity',
                 'selectivity_index', 'synthetic_accessibility', 'clinical_similarity']
    available_cols = [col for col in score_cols if col in df.columns]

    if len(available_cols) > 1:
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[available_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   fmt='.2f', square=True)
        plt.title('Score Correlation Analysis')
        plt.tight_layout()
        plt.savefig(output_dir / "score_correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 5. 分子性质分布图
    property_cols = ['molecular_weight', 'logp', 'tpsa', 'hbd', 'hba']
    available_props = [col for col in property_cols if col in df.columns and not df[col].isnull().all()]

    if available_props:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()

        for i, prop in enumerate(available_props[:6]):
            if i < len(axes):
                sns.histplot(df[prop], bins=20, kde=True, ax=axes[i])
                axes[i].set_title(f'{prop.replace("_", " ").title()} Distribution')
                axes[i].set_xlabel(prop.replace("_", " ").title())
                axes[i].set_ylabel('Count')

        # 隐藏多余的子图
        for i in range(len(available_props), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_dir / "molecular_properties_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 6. 顶级候选分子雷达图
    top_5 = df.nlargest(5, 'composite_score')
    if len(top_5) >= 3:
        # 雷达图需要的指标
        radar_metrics = ['bbb_permeability', 'gbm_activity', 'selectivity_index',
                        'synthetic_accessibility', 'clinical_similarity']

        # 归一化数据
        radar_data = top_5[radar_metrics].copy()
        for col in radar_metrics:
            if col in radar_data.columns:
                radar_data[col] = (radar_data[col] - radar_data[col].min()) / (radar_data[col].max() - radar_data[col].min())

        # 创建雷达图
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

        # 计算角度
        angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形

        for idx, (_, row) in enumerate(radar_data.iterrows()):
            values = row.values.tolist()
            values += values[:1]  # 闭合图形

            ax.plot(angles, values, 'o-', linewidth=2, label=f'Molecule {idx+1}')
            ax.fill(angles, values, alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in radar_metrics])
        ax.set_ylim(0, 1)
        ax.set_title('Top Candidate Molecules Multi-dimensional Comparison', size=16, fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(output_dir / "top_candidates_radar.png", dpi=300, bbox_inches='tight')
        plt.close()


def perform_detailed_analysis(df: pd.DataFrame, output_dir: Path):
    """进行详细分析"""
    analysis_results = {}

    # 1. 靶点特异性分析
    target_analysis = {}
    for target in df['target'].unique():
        target_data = df[df['target'] == target]
        target_analysis[target] = {
            'count': len(target_data),
            'avg_score': target_data['composite_score'].mean(),
            'best_score': target_data['composite_score'].max(),
            'bbb_success_rate': (target_data['bbb_classification'] == 'high').mean(),
            'high_potential_count': len(target_data[target_data['composite_score'] > 0.7])
        }

    analysis_results['target_analysis'] = target_analysis

    # 2. 性质-活性关系分析
    property_activity_corr = {}
    score_cols = ['composite_score', 'gbm_activity', 'bbb_permeability']
    property_cols = ['molecular_weight', 'logp', 'tpsa', 'hbd', 'hba', 'qed']

    for score_col in score_cols:
        if score_col in df.columns:
            correlations = {}
            for prop_col in property_cols:
                if prop_col in df.columns:
                    corr = df[score_col].corr(df[prop_col])
                    if not pd.isna(corr):
                        correlations[prop_col] = corr
            property_activity_corr[score_col] = correlations

    analysis_results['property_activity_correlations'] = property_activity_corr

    # 3. 成功案例识别
    success_criteria = (
        (df['composite_score'] > 0.8) &
        (df['bbb_classification'] == 'high') &
        (df['selectivity_index'] > 5)
    )

    success_cases = df[success_criteria]
    analysis_results['success_cases'] = {
        'count': len(success_cases),
        'details': success_cases[['id', 'target', 'composite_score', 'selectivity_index']].to_dict('records')
    }

    # 保存详细分析结果
    analysis_file = output_dir / "detailed_analysis.json"
    with open(analysis_file, 'w', encoding='utf-8') as f:
        # 将numpy类型转换为Python类型以便JSON序列化
        json_compatible_results = {}
        for key, value in analysis_results.items():
            if isinstance(value, dict):
                json_compatible_results[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, dict):
                        json_compatible_results[key][sub_key] = {}
                        for k, v in sub_value.items():
                            if isinstance(v, (np.float64, np.float32)):
                                json_compatible_results[key][sub_key][k] = float(v)
                            elif isinstance(v, np.int64):
                                json_compatible_results[key][sub_key][k] = int(v)
                            else:
                                json_compatible_results[key][sub_key][k] = v
                    else:
                        if isinstance(sub_value, (np.float64, np.float32)):
                            json_compatible_results[key][sub_key] = float(sub_value)
                        elif isinstance(sub_value, np.int64):
                            json_compatible_results[key][sub_key] = int(sub_value)
                        else:
                            json_compatible_results[key][sub_key] = sub_value
            else:
                json_compatible_results[key] = value

        json.dump(json_compatible_results, f, indent=2, ensure_ascii=False)


def print_summary_report(report: Dict[str, Any]):
    """打印汇总报告"""
    print("\n" + "=" * 80)
    print("GBM候选分子评估分析报告")
    print("=" * 80)

    print(f"\n总分子数: {report['total_molecules']}")
    print(f"覆盖靶点数: {report['targets_covered']}")
    print(f"高潜力分子数 (得分>0.7): {report['high_potential_molecules']}")
    print(f"良好BBB穿透分子数: {report['good_bbb_molecules']}")

    print(f"\n综合得分统计:")
    stats = report['score_statistics']['composite_score']
    print(f"  平均得分: {stats['mean']:.3f}")
    print(f"  标准差: {stats['std']:.3f}")
    print(f"  最低得分: {stats['min']:.3f}")
    print(f"  最高得分: {stats['max']:.3f}")
    print(f"  中位数得分: {stats['median']:.3f}")
    print(f"\n靶点分布:")
    for target, count in report['target_distribution'].items():
        print(f"  {target}: {count} 个分子")

    if report['top_candidates']:
        print(f"\n顶级候选分子 (前5个):")
        for i, candidate in enumerate(report['top_candidates'][:5], 1):
            print(f"  {i}. ID:{candidate['id']} | 靶点:{candidate['target']} | "
                  ".3f"                  f"BBB:{candidate['bbb_classification']} | "
                  ".2f")

    print("\n" + "=" * 80)


def main():
    """主函数"""
    args = parse_args()

    # 检查结果文件存在
    if not os.path.exists(args.results_file):
        print(f"错误：结果文件不存在: {args.results_file}")
        sys.exit(1)

    # 设置输出目录
    if args.experiment_name is None:
        # 从结果文件路径推断实验名称
        results_path = Path(args.results_file)
        args.experiment_name = results_path.parent.name

    output_dir = Path(args.output_dir) / f"{args.experiment_name}_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"开始分析实验结果: {args.experiment_name}")
    print(f"结果文件: {args.results_file}")
    print(f"输出目录: {output_dir}")

    try:
        # 加载结果
        print("加载结果数据...")
        full_data = load_results(args.results_file)
        results = full_data.get('results', [])

        # 创建分析DataFrame
        print("处理数据...")
        df = create_analysis_dataframe(results)

        if len(df) == 0:
            print("错误：没有有效的评估数据")
            sys.exit(1)

        # 生成汇总报告
        print("生成汇总报告...")
        report = generate_summary_report(df, output_dir)

        # 生成可视化图表
        if args.generate_plots:
            print("生成可视化图表...")
            generate_visualizations(df, output_dir)

        # 进行详细分析
        if args.detailed_analysis:
            print("进行详细分析...")
            perform_detailed_analysis(df, output_dir)

        # 打印汇总报告
        print_summary_report(report)

        print(f"\n分析完成！结果保存在: {output_dir}")

    except Exception as e:
        print(f"分析过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
