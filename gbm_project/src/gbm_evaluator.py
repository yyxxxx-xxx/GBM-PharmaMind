"""
GBM专项评估器
评估GBM候选分子的多维度性质，包括BBB穿透、活性、毒性等
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, QED
from . import sascorer
import rdkit
import math
import warnings
warnings.filterwarnings('ignore')


class BBBPermeabilityPredictor:
    """
    血脑屏障穿透性预测器（Clark 公式 + 启发式惩罚）。

    策略：
    1. Clark 公式（Clark & Delany, 2000）用于 logBB 粗估：
           logBB_est = 0.152 * LogP - 0.0148 * TPSA + 0.139
       仅使用脂水分配系数和拓扑极性表面积。
    2. 启发式惩罚项（不属于 Clark 原始公式）：
           MW > 450:   -0.20
           HBD > 2:    -0.15
           HBA > 6:    -0.10
       最终 logBB = logBB_est + penalty
    3. 分类阈值基于 logBB 值（与 Clark 原始阈值一致）
    """

    def predict(self, mol) -> float:
        """
        预测 BBB 穿透性 (logBB)。
        返回 Clark logBB + 惩罚项的和，供外部 normalize 使用。
        """
        if mol is None:
            return -2.0

        try:
            logp = Crippen.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            mw   = Descriptors.ExactMolWt(mol)
            hbd  = Descriptors.NumHDonors(mol)
            hba  = Descriptors.NumHAcceptors(mol)
        except Exception:
            return -2.0

        # --- Clark 公式：logBB 粗估（仅 LogP + TPSA）---
        logBB_est = 0.152 * logp - 0.0148 * tpsa + 0.139

        # --- 启发式惩罚项（不属于 Clark 原始公式）---
        penalty = 0.0
        if mw > 450:
            penalty -= 0.20
        if hbd > 2:
            penalty -= 0.15
        if hba > 6:
            penalty -= 0.10

        return logBB_est + penalty

    def classify_permeability(self, logbb: float) -> str:
        """分类 BBB 穿透性 — 基于 Clark logBB 阈值"""
        if logbb > -0.1:
            return "high"
        elif logbb > -0.3:
            return "medium"
        elif logbb > -0.5:
            return "low"
        else:
            return "very_low"


class GBMActivityPredictor:
    """GBM细胞活性预测器"""

    def __init__(self):
        # 基于分子描述符的简化GBM活性预测
        self.gbm_selective_features = {
            'high_logp': lambda x: 1 if x > 3 else 0,
            'moderate_mw': lambda x: 1 if 300 < x < 600 else 0,
            'low_tpsa': lambda x: 1 if x < 100 else 0,
            'kinase_like': lambda mol: 1 if self._has_kinase_motif(mol) else 0
        }

    def predict_activity(self, mol) -> float:
        """预测GBM细胞活性 (pIC50)"""
        if mol is None:
            return 4.0

        try:
            mw = Descriptors.ExactMolWt(mol)
            logp = Crippen.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)

            # 计算GBM选择性得分
            score = 0
            score += self.gbm_selective_features['high_logp'](logp) * 0.3
            score += self.gbm_selective_features['moderate_mw'](mw) * 0.25
            score += self.gbm_selective_features['low_tpsa'](tpsa) * 0.25
            score += self.gbm_selective_features['kinase_like'](mol) * 0.2

            # 转换为pIC50 (假设活性范围5-9)
            pic50 = 5.0 + score * 2.0

            return min(pic50, 9.0)
        except:
            return 4.0

    def _has_kinase_motif(self, mol) -> bool:
        """检测激酶抑制剂结构基序"""
        # 简化检测：检查是否有氮杂芳环
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 7:  # 氮原子
                # 检查是否在芳环中
                if atom.GetIsAromatic():
                    return True
        return False


class ToxicityPredictor:
    """毒性预测器"""

    def __init__(self):
        self.toxicity_rules = {
            'high_logp': lambda x: 0.8 if x > 4.5 else 0.1,
            'reactive_groups': lambda mol: self._count_reactive_groups(mol) * 0.2,
            'high_mw': lambda x: 0.3 if x > 600 else 0.05
        }

    def predict_normal_cell_toxicity(self, mol) -> float:
        """预测正常细胞毒性 (pIC50)"""
        if mol is None:
            return 5.0

        try:
            mw = Descriptors.ExactMolWt(mol)
            logp = Crippen.MolLogP(mol)

            toxicity_score = 0
            toxicity_score += self.toxicity_rules['high_logp'](logp)
            toxicity_score += self.toxicity_rules['reactive_groups'](mol)
            toxicity_score += self.toxicity_rules['high_mw'](mw)

            # 转换为pIC50 (毒性范围4-8)
            pic50 = 4.0 + toxicity_score * 2.0

            return min(pic50, 8.0)
        except:
            return 5.0

    def _count_reactive_groups(self, mol) -> int:
        """计数潜在反应性基团"""
        reactive_smarts = [
            '[C,c]=[N,n]',  # 亚胺
            '[N,n]-[N,n]',  # 偶氮
            '[C,c]-[Cl,Br,I]',  # 卤代烃
        ]

        count = 0
        for smarts in reactive_smarts:
            try:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern:
                    matches = mol.GetSubstructMatches(pattern)
                    count += len(matches)
            except:
                continue

        return count


class SyntheticAccessibilityScorer:
    """合成可行性评分器"""

    def __init__(self):
        # SA分数计算参数 (基于Ertl et al.方法简化版)
        self.fragment_penalties = {
            'ring_fusion': 0.1,
            'stereocenter': 0.1,
            'large_ring': 0.2,
            'complex_heterocycle': 0.15
        }

    def calculate_sa_score(self, mol) -> float:
        """计算合成可行性得分 (1-10, 10最易合成)"""
        if mol is None:
            return 3.0

        try:
            # Prefer using more robust sascorer implementation if available
            try:
                return float(sascorer.calculateScore(mol))
            except Exception:
                # fallback to simple heuristic (legacy)
                base_score = 8.0

                # 分子量惩罚
                mw = Descriptors.ExactMolWt(mol)
                if mw > 500:
                    base_score -= (mw - 500) / 100

                # 环复杂度惩罚
                ring_count = Descriptors.RingCount(mol)
                if ring_count > 3:
                    base_score -= (ring_count - 3) * 0.5

                # 手性中心惩罚
                chiral_centers = len(Chem.FindMolChiralCenters(mol))
                base_score -= chiral_centers * 0.3

                # 杂原子比例奖励
                hetero_ratio = len([a for a in mol.GetAtoms() if a.GetAtomicNum() not in [1,6]]) / mol.GetNumAtoms()
                if hetero_ratio > 0.3:
                    base_score += 0.5

                return max(1.0, min(10.0, base_score))
        except:
            return 3.0


class ClinicalSimilarityCalculator:
    """临床相似度计算器"""

    def __init__(self, reference_molecules: List[str]):
        self.reference_molecules = reference_molecules
        self.reference_mols = [Chem.MolFromSmiles(smiles) for smiles in reference_molecules if smiles]
        self.reference_mols = [mol for mol in self.reference_mols if mol is not None]

    def calculate_similarity(self, mol) -> float:
        """计算与临床分子的相似度 (0-1)"""
        if mol is None or not self.reference_mols:
            return 0.0

        try:
            max_similarity = 0
            for ref_mol in self.reference_mols:
                # 使用Tanimoto相似度基于Morgan指纹
                fp1 = Chem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                fp2 = Chem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=1024)
                similarity = Chem.DataStructs.TanimotoSimilarity(fp1, fp2)
                max_similarity = max(max_similarity, similarity)

            return max_similarity
        except:
            return 0.0


class DrugLikenessCalculator:
    """类药性计算器：结合Lipinski规则和QED，返回0-1分数"""
    def __init__(self):
        pass

    def calculate_druglikeness(self, mol) -> float:
        """返回 0-1 的类药性分数（越高越接近药物样）"""
        if mol is None:
            return 0.0
        try:
            # Lipinski规则判定 (4项：MW, LogP, HBD, HBA)
            mw = Descriptors.ExactMolWt(mol)
            logp = Crippen.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)

            lipinski_passes = 0
            lipinski_passes += 1 if mw <= 500 else 0
            lipinski_passes += 1 if logp <= 5 else 0
            lipinski_passes += 1 if hbd <= 5 else 0
            lipinski_passes += 1 if hba <= 10 else 0
            lipinski_score = lipinski_passes / 4.0

            # QED 值（已经在 properties 中使用）
            try:
                qed_value = QED.qed(mol)
            except Exception:
                qed_value = 0.0

            # 组合：权重可调，这里先平均
            combined = 0.5 * lipinski_score + 0.5 * qed_value
            return max(0.0, min(1.0, combined))
        except:
            return 0.0


class GBMEvaluator:
    """GBM专项评估器"""

    def __init__(self, reference_molecules: Optional[List[str]] = None):
        self.bbb_predictor = BBBPermeabilityPredictor()
        self.activity_predictor = GBMActivityPredictor()
        self.toxicity_predictor = ToxicityPredictor()
        self.sa_scorer = SyntheticAccessibilityScorer()
        # 用类药性计算替代原来的临床相似度计算器
        self.druglikeness_calculator = DrugLikenessCalculator()
        # 保留原始参考分子作为元数据（如果需要）
        self._reference_molecules = reference_molecules or self._get_default_references()

        # 评估权重 - 基于GBM药物评估策略优化
        # BBB穿透性对GBM至关重要，合成可及性反映可行性
        self.weights = {
            'bbb_permeability': 0.35,      # 提高到35% - GBM关键指标
            'gbm_activity': 0.25,          # 保持25% - 治疗效果核心
            'toxicity': 0.15,              # 降低到15% - 已上市药物毒性已验证
            'synthetic_accessibility': 0.20, # 提高到20% - 药物开发可行性
            'druglikeness': 0.05            # 用类药性取代临床相似度
        }

    def _get_default_references(self) -> List[str]:
        """获取默认参考分子SMILES"""
        return [
            'Cn1cnc2N(C)C(=O)N(C)C(=O)c12',  # Temozolomide
            'CC1CCC2CC(C(=CC=CC=CC(CC(C(=O)C(C(C(=CC(C(=O)CC(OC(=O)C3CCCCN3C(=O)C(=O)C1(O2)O)C(C)CC4CCC(C(C4)OC)O)C)C)O)OC)C)C)C)OC',  # Everolimus
            'C1CC2=C(C=C(C=C2)Cl)C(=O)N1C(=O)N.Cl',  # Carmustine
            'CC1CCC(CC1)C(=O)N2C(=O)N(C(=O)N2)Cl'  # Lomustine
        ]

    def evaluate_molecule(self, smiles: str) -> Dict[str, Any]:
        """全面评估GBM候选分子"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return self._create_failed_evaluation(smiles)

            # 基本分子过滤：过滤明显不适合作为药物的分子
            if not self._passes_basic_drug_filters(mol):
                return self._create_failed_evaluation(smiles, "分子不符合基本药物标准")

            # 计算各项指标
            bbb_score = self.bbb_predictor.predict(mol)
            activity_score = self.activity_predictor.predict_activity(mol)
            toxicity_score = self.toxicity_predictor.predict_normal_cell_toxicity(mol)
            sa_score = self.sa_scorer.calculate_sa_score(mol)
            druglikeness = self.druglikeness_calculator.calculate_druglikeness(mol)

            # 计算选择性指数 (GBM活性/正常细胞毒性)
            selectivity_index = activity_score / toxicity_score if toxicity_score > 0 else 0

            # 计算综合得分
            composite_score = self._calculate_composite_score({
                'bbb_permeability': self._normalize_bbb_score(bbb_score),
                'gbm_activity': self._normalize_activity_score(activity_score),
                'toxicity': self._normalize_toxicity_score(toxicity_score),
                'synthetic_accessibility': sa_score / 10.0,  # 归一化到0-1
                'druglikeness': druglikeness
            })

            # 分类评估结果
            bbb_class = self.bbb_predictor.classify_permeability(bbb_score)

            return {
                'smiles': smiles,
                'valid': True,
                'scores': {
                    'bbb_permeability': bbb_score,
                    'bbb_classification': bbb_class,
                    'gbm_activity': activity_score,
                    'normal_cell_toxicity': toxicity_score,
                    'selectivity_index': selectivity_index,
                    'synthetic_accessibility': sa_score,
                    'druglikeness': druglikeness,
                    'composite_score': composite_score
                },
                'properties': self._calculate_basic_properties(mol),
                'assessment': self._generate_assessment(composite_score, selectivity_index, bbb_class)
            }

        except Exception as e:
            return self._create_failed_evaluation(smiles, str(e))

    def _passes_basic_drug_filters(self, mol) -> bool:
        """检查分子是否通过基本药物过滤"""
        try:
            mw = Descriptors.ExactMolWt(mol)
            num_atoms = mol.GetNumAtoms()
            num_heavy_atoms = mol.GetNumHeavyAtoms()
            num_rings = Chem.rdMolDescriptors.CalcNumRings(mol)

            # 基本药物标准过滤
            # 1. 分子量：药物通常在150-800 Da之间，但GBM药物更倾向于200-600 Da
            if mw < 150 or mw > 800:
                return False

            # 2. 重原子数：太少的原子无法形成有效的药效团
            if num_heavy_atoms < 8:  # 苯有6个重原子，不够
                return False

            # 3. 避免单环简单芳烃（像苯这样的分子）
            if num_rings == 1 and num_heavy_atoms <= 6 and num_atoms <= 6:
                # 检查是否是简单的芳香烃（苯、甲苯等）
                if all(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms()):
                    return False

            # 4. 必须包含至少一个杂原子或功能团（N, O, S, F, Cl, Br, I等）
            has_heteroatom = any(atom.GetAtomicNum() in [7, 8, 9, 15, 16, 17, 35, 53]
                                for atom in mol.GetAtoms())
            if not has_heteroatom:
                return False

            # 5. 避免极端疏水分子（LogP太高）
            logp = Crippen.MolLogP(mol)
            if logp > 6.0:  # 过于疏水
                return False

            return True

        except:
            return False

    def evaluate_batch(self, smiles_list: List[str]) -> List[Dict[str, Any]]:
        """批量评估分子"""
        results = []
        for smiles in smiles_list:
            result = self.evaluate_molecule(smiles)
            results.append(result)

        return results

    def _normalize_bbb_score(self, logbb: float) -> float:
        """归一化 BBB 得分 (0-1, 越高越好)。

        基于 Clark 公式：logBB > 0.5 = 1.0, logBB < -1.0 = 0.0
        """
        return max(0.0, min(1.0, (logbb + 1.0) / 1.5))

    def _normalize_activity_score(self, activity_score: float) -> float:
        """归一化活性得分 (0-1, 越高越好)"""
        # pIC50 > 7 为优秀，< 5 为较差
        return max(0, min(1, (activity_score - 5) / 3))

    def _normalize_toxicity_score(self, toxicity_score: float) -> float:
        """归一化毒性得分 (0-1, 越低越好)"""
        # pIC50 < 5 为低毒性，> 7 为高毒性
        return max(0, min(1, 1 - (toxicity_score - 4) / 4))

    def _calculate_composite_score(self, normalized_scores: Dict[str, float]) -> float:
        """计算综合得分"""
        composite = 0
        for metric, score in normalized_scores.items():
            composite += score * self.weights[metric]

        return composite

    def _calculate_basic_properties(self, mol) -> Dict[str, Any]:
        """计算基本分子性质"""
        try:
            return {
                'molecular_weight': Descriptors.ExactMolWt(mol),
                'logp': Crippen.MolLogP(mol),
                'tpsa': Descriptors.TPSA(mol),
                'hbd': Descriptors.NumHDonors(mol),
                'hba': Descriptors.NumHAcceptors(mol),
                'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'ring_count': Descriptors.RingCount(mol),
                'qed': QED.qed(mol)
            }
        except:
            return {}

    def _generate_assessment(self, composite_score: float, selectivity_index: float, bbb_class: str) -> str:
        """生成评估总结"""
        assessment = f"综合得分: {composite_score:.2f}/1.00"

        if composite_score > 0.8:
            assessment += " - 优秀GBM候选物"
        elif composite_score > 0.6:
            assessment += " - 良好GBM候选物"
        elif composite_score > 0.4:
            assessment += " - 一般GBM候选物"
        else:
            assessment += " - 需要进一步优化"

        assessment += f"\n选择性指数: {selectivity_index:.2f}"
        assessment += f"\nBBB穿透性: {bbb_class}"

        return assessment

    def _create_failed_evaluation(self, smiles: str, error: str = "分子解析失败") -> Dict[str, Any]:
        """创建失败评估结果"""
        return {
            'smiles': smiles,
            'valid': False,
            'error': error,
            'scores': {
                'composite_score': 0.0
            }
        }

    def get_evaluation_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成评估结果汇总"""
        valid_results = [r for r in results if r['valid']]

        if not valid_results:
            return {'error': '没有有效的评估结果'}

        scores = [r['scores']['composite_score'] for r in valid_results]

        return {
            'total_molecules': len(results),
            'valid_molecules': len(valid_results),
            'average_score': np.mean(scores),
            'max_score': np.max(scores),
            'min_score': np.min(scores),
            'std_score': np.std(scores),
            'high_potential_count': len([s for s in scores if s > 0.7])
        }
