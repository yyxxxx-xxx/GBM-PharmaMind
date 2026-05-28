"""
GBM Knowledge Base
包含GBM相关的靶点信息、临床数据和分子特征
"""

import json
import yaml
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import random


@dataclass
class GBMTarget:
    """GBM靶点信息"""
    name: str
    description: str
    mutation_types: List[str]
    current_drugs: List[str]
    challenges: List[str]
    structural_requirements: Dict[str, Any]


@dataclass
class GBMClinicalData:
    """GBM临床数据"""
    standard_treatment: Dict[str, str]
    treatment_challenges: List[str]
    failed_trials_insights: List[str]
    successful_patterns: List[str]
    molecular_subtypes: Dict[str, Dict]
    biomarkers: Dict[str, Dict]
    resistance_mechanisms: Dict[str, List]


@dataclass
class GBMMolecule:
    """GBM相关分子信息"""
    name: str
    smiles: str
    target: str
    status: str
    clinical_phase: str
    mechanism: str
    limitations: str
    structural_features: Dict[str, Any]
    failure_reason: Optional[str] = None


class GBMKnowledgeBase:
    """GBM知识库管理类"""

    def __init__(self, targets_path: str, clinical_path: str, molecules_path: str):
        self.targets_path = targets_path
        self.clinical_path = clinical_path
        self.molecules_path = molecules_path

        # 加载数据
        self.targets = self._load_targets()
        self.clinical_data = self._load_clinical_data()
        self.molecules = self._load_molecules()

        # 创建索引
        self._create_indexes()

    def _load_targets(self) -> Dict[str, GBMTarget]:
        """加载GBM靶点数据"""
        with open(self.targets_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        targets = {}
        for target_data in data['gbm_targets']:
            try:
                # 处理不同的靶点数据结构
                mutation_types = target_data.get('mutation_types', [])
                if not mutation_types:
                    # 对于没有mutation_types的靶点，使用其他字段
                    if 'key_markers' in target_data:
                        mutation_types = [f"Key markers: {', '.join(target_data['key_markers'])}"]
                    elif 'expression_pattern' in target_data:
                        mutation_types = [f"Expression: {', '.join(target_data['expression_pattern'])}"]
                    else:
                        mutation_types = ["Not applicable"]

                target = GBMTarget(
                    name=target_data['name'],
                    description=target_data['description'],
                    mutation_types=mutation_types,
                    current_drugs=target_data.get('current_drugs', []),
                    challenges=target_data.get('challenges', []),
                    structural_requirements=target_data.get('structural_requirements', {})
                )
                targets[target.name] = target
            except Exception as e:
                print(f"警告 - 创建靶点 {target_data.get('name', 'unknown')} 时出错: {e}")
                continue

        return targets

    def _load_clinical_data(self) -> GBMClinicalData:
        """加载GBM临床数据"""
        with open(self.clinical_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 处理嵌套结构
        clinical_data = data.get('clinical_data', {})

        return GBMClinicalData(
            standard_treatment=clinical_data.get('standard_treatment', {}),
            treatment_challenges=clinical_data.get('treatment_challenges', []),
            failed_trials_insights=clinical_data.get('failed_trials_insights', []),
            successful_patterns=clinical_data.get('successful_patterns', []),
            molecular_subtypes=data.get('molecular_subtypes', {}),
            biomarkers=data.get('biomarkers', {}),
            resistance_mechanisms=data.get('resistance_mechanisms', {})
        )

    def _load_molecules(self) -> Dict[str, GBMMolecule]:
        """加载GBM相关分子数据"""
        with open(self.molecules_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        molecules = {}

        # 处理分子数据，确保所有必需字段都存在
        def create_molecule(mol_data):
            return GBMMolecule(
                name=mol_data['name'],
                smiles=mol_data['smiles'],
                target=mol_data['target'],
                status=mol_data['status'],
                clinical_phase=mol_data.get('clinical_phase', 'Unknown'),
                mechanism=mol_data['mechanism'],
                limitations=mol_data.get('limitations', 'Not specified'),
                structural_features=mol_data.get('structural_features', {}),
                failure_reason=mol_data.get('failure_reason')
            )

        # 加载已批准药物
        for mol_data in data['gbm_drugs']:
            try:
                mol = create_molecule(mol_data)
                molecules[mol.name] = mol
            except Exception as e:
                print(f"警告 - 加载药物 {mol_data.get('name', 'unknown')} 时出错: {e}")
                continue

        # 加载临床候选物
        for mol_data in data['clinical_candidates']:
            try:
                mol = create_molecule(mol_data)
                molecules[mol.name] = mol
            except Exception as e:
                print(f"警告 - 加载候选物 {mol_data.get('name', 'unknown')} 时出错: {e}")
                continue

        return molecules

    def _create_indexes(self):
        """创建数据索引以便快速查询"""
        # 按靶点索引分子
        self.molecules_by_target = {}
        for mol in self.molecules.values():
            if mol.target not in self.molecules_by_target:
                self.molecules_by_target[mol.target] = []
            self.molecules_by_target[mol.target].append(mol)

        # 按状态索引分子
        self.molecules_by_status = {}
        for mol in self.molecules.values():
            if mol.status not in self.molecules_by_status:
                self.molecules_by_status[mol.status] = []
            self.molecules_by_status[mol.status].append(mol)

    def get_target_info(self, target_name: str) -> Optional[GBMTarget]:
        """获取特定靶点信息"""
        return self.targets.get(target_name)

    def get_random_target(self, weights: Optional[Dict[str, float]] = None) -> GBMTarget:
        """根据权重随机选择靶点"""
        if weights is None:
            return random.choice(list(self.targets.values()))

        targets = list(weights.keys())
        probabilities = [weights[t] for t in targets]
        selected_target = random.choices(targets, weights=probabilities, k=1)[0]
        return self.targets[selected_target]

    def get_similar_molecules(self, target: str, limit: int = 5) -> List[GBMMolecule]:
        """获取特定靶点的相似分子"""
        if target not in self.molecules_by_target:
            return []

        molecules = self.molecules_by_target[target]
        return molecules[:limit] if len(molecules) > limit else molecules

    def get_clinical_insights(self) -> Dict[str, Any]:
        """获取临床治疗洞察"""
        return {
            'challenges': self.clinical_data.treatment_challenges,
            'failed_insights': self.clinical_data.failed_trials_insights,
            'successful_patterns': self.clinical_data.successful_patterns
        }

    def get_design_principles(self) -> Dict[str, Any]:
        """获取分子设计原则"""
        # 从分子数据中提取设计原则
        with open(self.molecules_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data.get('design_principles', {})

    def get_structural_motifs(self) -> Dict[str, List[str]]:
        """获取结构基序"""
        with open(self.molecules_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data.get('structural_motifs', {})

    def get_subtype_info(self, subtype: str) -> Optional[Dict]:
        """获取分子亚型信息"""
        return self.clinical_data.molecular_subtypes.get(subtype)

    def get_resistance_mechanisms(self, target_type: str) -> List[str]:
        """获取耐药机制"""
        return self.clinical_data.resistance_mechanisms.get(target_type, [])
