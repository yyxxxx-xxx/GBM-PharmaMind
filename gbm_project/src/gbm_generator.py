"""
GBM分子生成器
集成GBM知识库、prompt生成和评估器的完整生成流程
"""

import os
import json
import yaml
from pathlib import Path
import torch
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
import random
import numpy as np
from transformers import GenerationConfig

# 导入GBM相关模块
from .gbm_knowledge_base import GBMKnowledgeBase
from .gbm_prompt_generator import GBMPromptGenerator
from .gbm_evaluator import GBMEvaluator

# Llamole相关模块将在运行时动态导入


class GBMGenerator:
    """GBM分子生成器"""

    def __init__(self, config_path_or_dict: str or dict):
        """
        初始化GBM生成器

        Args:
            config_path_or_dict: GBM配置文件路径或配置字典
        """
        # 确定项目根目录（Llamole-main）
        self.project_root = Path(__file__).resolve().parents[2]
        
        # 加载配置
        if isinstance(config_path_or_dict, str):
            # 配置文件路径
            config_path = Path(config_path_or_dict)
            if not config_path.is_absolute():
                # 如果是相对路径，先尝试相对于当前工作目录解析
                working_dir_path = Path.cwd() / config_path
                if working_dir_path.exists():
                    config_path = working_dir_path.resolve()
                else:
                    # 如果不存在，尝试从项目根目录解析
                    config_path = (self.project_root / config_path).resolve()
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        elif isinstance(config_path_or_dict, dict):
            # 配置字典
            self.config = config_path_or_dict
        else:
            raise TypeError("config_path_or_dict必须是字符串路径或字典")

        # 初始化GBM专用组件
        self._init_gbm_components()

        # 初始化Llamole模型 (如果可用)
        self.llamole_model = None
        self.tokenizer = None
        self._init_llamole_model()

        # 设置随机种子
        self._set_seed(self.config.get('seed', 42))

    def _init_gbm_components(self):
        """初始化GBM专用组件"""
        # 解析路径（如果是相对路径，从项目根目录解析）
        def resolve_path(path_str):
            path = Path(path_str)
            if path.is_absolute():
                return str(path)
            return str(self.project_root / path)
        
        # 初始化知识库
        targets_path = resolve_path(self.config['gbm_targets_path'])
        clinical_path = resolve_path(self.config['gbm_clinical_path'])
        molecules_path = resolve_path(self.config['gbm_molecules_path'])

        self.knowledge_base = GBMKnowledgeBase(targets_path, clinical_path, molecules_path)

        # 初始化prompt生成器
        prompts_config_path = resolve_path(self.config['gbm_prompt_templates_path'])
        self.prompt_generator = GBMPromptGenerator(self.knowledge_base, prompts_config_path)

        # 初始化评估器
        reference_molecules = self._get_reference_molecules()
        self.evaluator = GBMEvaluator(reference_molecules)

    def _init_llamole_model(self):
        """初始化Llamole模型"""
        try:
            # 添加必要的路径
            import sys
            llamole_root = Path(__file__).parent.parent.parent
            if str(llamole_root) not in sys.path:
                sys.path.insert(0, str(llamole_root))

            # 导入Llamole模块
            from src.model.modeling_llamole import GraphLLMForCausalMLM
            from src.model.loader import load_tokenizer
            # 这里直接使用内部的 _parse_train_args，避免对训练用的严格检查，
            # 同时只传入与 Llamole 相关的配置键，保证“环境用配置好的 Llamole”
            from src.hparams.parser import _parse_train_args

            # 过滤掉 GBM 专用、不会被 HfArgumentParser 使用的配置项
            gbm_specific_keys = {
                "gbm_targets_path",
                "gbm_clinical_path",
                "gbm_molecules_path",
                "gbm_prompt_templates_path",
                "gbm_generation_mode",
                "use_gbm_domain_knowledge",
                "use_cot_reasoning",
                "target_selection_strategy",
                "molecular_constraints",
                "gbm_target_weights",
                "evaluate_gbm_properties",
                "gbm_evaluation_metrics",
                "num_candidates_per_target",
                "filter_candidates",
                "ranking_method",
                "save_intermediate_results",
                "intermediate_save_path",
                "num_return_sequences",
            }

            llamole_config = {
                k: v for k, v in self.config.items() if k not in gbm_specific_keys
            }

            # 只解析 Llamole 相关配置，得到 model/data/training/finetuning/generating 五类参数
            model_args, data_args, training_args, finetuning_args, generating_args = _parse_train_args(
                llamole_config
            )

            # 加载 tokenizer（与官方评测 workflow 保持一致）
            tokenizer_module = load_tokenizer(model_args, generate_mode=True)
            tokenizer = tokenizer_module["tokenizer"]

            # 使用预训练的 Llamole 图文模型（仅推理，不训练）
            self.llamole_model = GraphLLMForCausalMLM.from_pretrained(
                tokenizer, model_args, data_args, training_args, finetuning_args, load_adapter=True
            )

            # 保存 tokenizer 以供后续生成使用
            self.tokenizer = tokenizer

            print("✓ Llamole模型加载成功")
            print(f"  - 模型路径: {self.config.get('model_name_or_path')}")
            print(f"  - 适配器路径: {self.config.get('adapter_name_or_path')}")

        except Exception as e:
            print(f"❌ Llamole模型加载失败: {e}")
            print("请检查模型路径和依赖是否正确")
            import traceback
            traceback.print_exc()
            raise

    def _get_reference_molecules(self) -> List[str]:
        """获取参考分子SMILES列表"""
        reference_molecules = []
        for mol in self.knowledge_base.molecules.values():
            if mol.status in ['Approved', 'Phase 3', 'Phase 2']:
                reference_molecules.append(mol.smiles)
        return reference_molecules[:10]  # 限制数量

    def _set_seed(self, seed: int):
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def generate_gbm_molecules(self, num_candidates: int = 20, target_name: Optional[str] = None,
                              constraints: Optional[Dict[str, Any]] = None,
                              use_cot: bool = True) -> List[Dict[str, Any]]:
        """
        生成GBM候选分子

        Args:
            num_candidates: 生成候选分子数量
            target_name: 指定靶点名称，如果为None则自动选择
            constraints: 分子约束条件
            use_cot: 是否使用Chain-of-Thought推理

        Returns:
            生成的分子列表，包含SMILES和评估结果
        """
        generated_molecules = []
        successful_generations = 0

        print(f"开始生成 {num_candidates} 个GBM候选分子...")

        for i in tqdm(range(num_candidates), desc="生成GBM候选分子"):
            try:
                # 生成单个分子
                molecule_data = self._generate_single_molecule(
                    target_name=target_name,
                    constraints=constraints,
                    use_cot=use_cot,
                    candidate_id=i+1
                )

                if molecule_data:
                    generated_molecules.append(molecule_data)
                    successful_generations += 1

            except Exception as e:
                print(f"生成分子 {i+1} 时出错: {e}")
                continue

        print(f"成功生成 {successful_generations}/{num_candidates} 个GBM候选分子")

        # 对结果进行排序
        generated_molecules.sort(key=lambda x: x.get('evaluation', {}).get('composite_score', 0),
                               reverse=True)

        return generated_molecules

    def _generate_single_molecule(self, target_name: Optional[str], constraints: Optional[Dict[str, Any]],
                                use_cot: bool, candidate_id: int) -> Optional[Dict[str, Any]]:
        """生成单个GBM候选分子"""

        # 选择靶点
        if target_name is None:
            target_weights = self.config.get('gbm_target_weights', {})
            target = self.knowledge_base.get_random_target(target_weights)
            target_name = target.name

        # 生成GBM专业prompt
        full_prompt = self.prompt_generator.generate_full_prompt(
            target_name=target_name,
            constraints=constraints,
            use_cot=use_cot
        )

        # 使用Llamole生成分子
        generated_text = self._generate_with_llamole(full_prompt)

        # 提取SMILES字符串
        smiles = self._extract_smiles_from_text(generated_text)

        if not smiles:
            return None

        # 评估生成的分子
        evaluation = self.evaluator.evaluate_molecule(smiles)

        return {
            'id': candidate_id,
            'target': target_name,
            'smiles': smiles,
            'prompt': full_prompt,
            'generated_text': generated_text,
            'evaluation': evaluation,
            'timestamp': self._get_timestamp()
        }

    def _generate_with_llamole(self, prompt: str) -> str:
        """使用Llamole模型生成文本"""
        if self.llamole_model is None or self.tokenizer is None:
            raise RuntimeError("Llamole模型或tokenizer未正确加载")

        try:
            # 编码输入，并将张量移动到与模型相同的设备上（保持与已配置的 Llamole 环境一致）
            inputs = self.tokenizer(prompt, return_tensors="pt")
            device = next(self.llamole_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # 生成配置
            generation_config = GenerationConfig(
                max_length=self.config.get('max_length', 1024),
                max_new_tokens=self.config.get('max_new_tokens', 256),
                temperature=self.config.get('temperature', 0.7),
                top_p=self.config.get('top_p', 0.9),
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

            # 生成
            with torch.no_grad():
                outputs = self.llamole_model.generate(
                    **inputs,
                    generation_config=generation_config
                )

            # 解码输出
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 移除输入prompt
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()

            return generated_text

        except Exception as e:
            print(f"Llamole生成出错: {e}")
            raise

    def _extract_smiles_from_text(self, text: str) -> Optional[str]:
        """从生成的文本中提取SMILES字符串"""
        import re

        # 常见的SMILES提取模式
        patterns = [
            r'SMILES:\s*([^\s\n]+)',
            r'smiles:\s*([^\s\n]+)',
            r'分子式:\s*([^\s\n]+)',
            r'C[0-9A-Za-z@+\-\[\]\(\)=#]+'  # 直接匹配SMILES模式
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # 验证SMILES有效性
                for smiles in matches:
                    if self._is_valid_smiles(smiles.strip()):
                        return smiles.strip()

        return None

    def _is_valid_smiles(self, smiles: str) -> bool:
        """验证SMILES字符串的有效性"""
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False

    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """保存生成结果"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 转换为可序列化格式
        serializable_results = []
        for result in results:
            serializable_result = result.copy()
            # 将不可序列化的对象移除或转换
            if 'evaluation' in serializable_result:
                eval_data = serializable_result['evaluation']
                if isinstance(eval_data, dict) and 'properties' in eval_data:
                    # 确保properties可序列化
                    props = eval_data['properties']
                    for key, value in props.items():
                        if isinstance(value, (np.float32, np.float64)):
                            props[key] = float(value)
                        elif isinstance(value, (np.int32, np.int64)):
                            props[key] = int(value)
            serializable_results.append(serializable_result)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        print(f"结果已保存到: {output_path}")

    def generate_evaluation_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成评估报告"""
        if not results:
            return {'error': '没有结果可报告'}

        # 基本统计
        valid_results = [r for r in results if r.get('evaluation', {}).get('valid', False)]
        scores = [r['evaluation']['scores']['composite_score'] for r in valid_results]

        report = {
            'total_generated': len(results),
            'valid_molecules': len(valid_results),
            'success_rate': len(valid_results) / len(results) if results else 0,
            'average_score': float(np.mean(scores)) if scores else 0,
            'max_score': float(np.max(scores)) if scores else 0,
            'min_score': float(np.min(scores)) if scores else 0,
            'high_potential_count': len([s for s in scores if s > 0.7]),
            'target_distribution': self._analyze_target_distribution(results)
        }

        return report

    def _analyze_target_distribution(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """分析靶点分布"""
        target_counts = {}
        for result in results:
            target = result.get('target', 'unknown')
            target_counts[target] = target_counts.get(target, 0) + 1

        return target_counts

    def filter_top_candidates(self, results: List[Dict[str, Any]], top_n: int = 10,
                            min_score: float = 0.6) -> List[Dict[str, Any]]:
        """筛选顶级候选分子"""
        valid_results = [r for r in results if r.get('evaluation', {}).get('valid', False)]

        # 按综合得分排序
        sorted_results = sorted(valid_results,
                              key=lambda x: x['evaluation']['scores']['composite_score'],
                              reverse=True)

        # 应用筛选条件
        filtered_results = [r for r in sorted_results
                          if r['evaluation']['scores']['composite_score'] >= min_score]

        return filtered_results[:top_n]
