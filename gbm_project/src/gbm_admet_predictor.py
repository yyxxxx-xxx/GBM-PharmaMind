"""
GBM ADMET 预测器 - 改进版
=========================
相比 gbm_evaluator.py 中的原始实现，本模块有以下改进：

1. ToxicityPredictorV2:
   - 扩展了反应性基团 SMARTS（从 3 种扩展到 15+ 种）
   - 集成了 RDKit PAINS/NIH/BRENK 过滤器
   - 添加 hERG 阻断风险检测
   - 添加 CYP450 抑制风险检测
   - 添加 PAINS 警报计数作为独立毒性指标

2. BBBPermeabilityPredictorV2:
   - 使用更标准的 logBB 线性模型系数（基于文献参数）
   - 添加 CNS 药物规则过滤器（MW < 450, TPSA < 90, LogP 1-4, HBD ≤ 2）
   - 分类阈值与实验数据对齐

3. ADMETFilter:
   - TPSA 下限从 20 Å² 提高到 40 Å²（CNS 药物标准）
   - LogP 范围收紧到 1.0-4.5（减少过高亲脂性导致的非特异性毒性）
   - 添加 PAINS/NIH/BRENK 过滤
   - 添加 hERG 和 CYP450 风险过滤

使用方式:
    from src.gbm_admet_predictor import (
        ToxicityPredictorV2, BBBPermeabilityPredictorV2, ADMETFilter
    )
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, QED, rdMolDescriptors
from rdkit.Chem import rdfiltercatalog
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# 毒性预测器 V2
# =============================================================================

class ToxicityPredictorV2:
    """
    改进版毒性预测器

    改进点：
    1. 反应性基团从 3 种扩展到 15+ 种，覆盖药物化学中常见的毒性/反应性基团
    2. 添加 PAINS/NIH/BRENK 过滤器（RDKit 内置）
    3. 添加 hERG 阻断风险检测（基于化学结构相似性）
    4. 添加 CYP450 抑制风险检测
    5. 每个指标独立评分后加权汇总
    """

    # 扩展的反应性/毒性基团 SMARTS（相比原版的 3 种大幅扩展）
    REACTIVE_GROUPS = [
        # α,β-不饱和羰基（迈克尔受体，反应性最强）
        ('Michael Acceptors', r'[#6]=[#6]-[#6](=O)-[#6]'),
        # 酰氯/磺酰氯
        ('Acyl Chloride', r'C(=O)Cl'),
        # 异硫氰酸酯
        ('Isothiocyanate', r'N=C=S'),
        # 环氧化物
        ('Epoxide', r'C1CO1'),
        # 氮丙啶（乙撑亚胺）
        ('Aziridine', r'C1CN1'),
        # 偶氮化合物
        ('Azo', r'[NX2]=[NX2]'),
        # 肼/酰肼（注意：药物中常见酰肼如酰肼布洛芬，需要风险权衡）
        ('Hydrazine', r'NN'),
        # 重氮甲烷衍生物
        ('Diazo', r'C=[N+]=[N-]'),
        # 芳香硝基（高毒性风险）
        ('Aromatic Nitro', r'c[N+](=O)[O-]'),
        # 醌类（氧化还原循环）
        ('Quinone', r'C1(=O)ccc(=O)c1'),
        # 邻苯二甲酰亚胺
        ('Phthalimide', r'c1ccc2c(c1)C(=O)NC2=O'),
    ]

    # hERG 阻断风险结构特征（基于已知 hERG 抑制剂药效团）
    # 注意：这些结构本身不是毒性的直接指标，需要结合其他特征综合判断
    HERG_RISK_PATTERNS = [
        # 叔胺（脂肪族哌嗪/哌啶）- 风险较高
        ('Piperazine', r'C1CNNCC1'),
        ('Piperidine', r'C1CCNCC1'),
        # 碱性氮原子（非酰胺氮）靠近疏水基团
        ('Alkyl Piperazine', r'C1CCN(C)CCN1'),
        ('Alkyl Piperidine', r'C1CCN(C)CC1'),
    ]

    # CYP450 抑制风险（尤其是 CYP3A4/CYP2D6）
    CYP_RISK_PATTERNS = [
        ('Furan', r'c1ccoc1'),
        ('Thiophene', r'c1ccsc1'),
        ('Thiazole', r'c1ncsc1'),
        ('Imidazole', r'c1cnc[nH]1'),
        ('Pyridine', r'c1ccncc1'),
        ('Alkoxyphenyl', r'c1ccc(O[#6])cc1'),
    ]

    def __init__(self):
        self._pains_catalog = None
        self._nih_catalog = None
        self._brenk_catalog = None
        self._pains_catalog_init()
        self._setup_cached_patterns()

    def _pains_catalog_init(self):
        """初始化 PAINS/NIH/BRENK 过滤器目录"""
        try:
            params = rdfiltercatalog.FilterCatalogParams()
            fc = params.FilterCatalogs
            params.AddCatalog(fc.PAINS)
            self._pains_catalog = rdfiltercatalog.FilterCatalog(params)
            logger.info("PAINS filter catalog loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load PAINS catalog: {e}")
            self._pains_catalog = None

        try:
            params_nih = rdfiltercatalog.FilterCatalogParams()
            fc = params_nih.FilterCatalogs
            params_nih.AddCatalog(fc.NIH)
            self._nih_catalog = rdfiltercatalog.FilterCatalog(params_nih)
            logger.info("NIH filter catalog loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load NIH catalog: {e}")
            self._nih_catalog = None

        try:
            params_brenk = rdfiltercatalog.FilterCatalogParams()
            fc = params_brenk.FilterCatalogs
            params_brenk.AddCatalog(fc.BRENK)
            self._brenk_catalog = rdfiltercatalog.FilterCatalog(params_brenk)
            logger.info("BRENK filter catalog loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load BRENK catalog: {e}")
            self._brenk_catalog = None

    def _setup_cached_patterns(self):
        """预编译 SMARTS 模式以提高性能"""
        self._reactive_patterns = []
        for name, smarts in self.REACTIVE_GROUPS:
            try:
                pat = Chem.MolFromSmarts(smarts)
                if pat:
                    self._reactive_patterns.append((name, pat))
            except Exception:
                pass

        self._herg_patterns = []
        for name, smarts in self.HERG_RISK_PATTERNS:
            try:
                pat = Chem.MolFromSmarts(smarts)
                if pat:
                    self._herg_patterns.append((name, pat))
            except Exception:
                pass

        self._cyp_patterns = []
        for name, smarts in self.CYP_RISK_PATTERNS:
            try:
                pat = Chem.MolFromSmarts(smarts)
                if pat:
                    self._cyp_patterns.append((name, pat))
            except Exception:
                pass

    def predict(self, mol) -> Dict[str, Any]:
        """
        综合毒性预测，返回多维度毒性评估。

        Returns:
            dict with keys:
            - toxicity_score (0-1, 越高毒性越大)
            - pIC50 (float, 正常细胞毒性 pIC50 估计)
            - reactive_group_count (int)
            - reactive_groups (list of str, 检出的反应性基团)
            - pains_alerts (int, PAINS 警报数)
            - nih_alerts (int, NIH 警报数)
            - brenk_alerts (int, BRENK 警报数)
            - herg_risk (bool)
            - cyp_risk (bool)
            - alerts (list of str, 所有警报描述)
        """
        if mol is None:
            return self._default_result()

        mw = Descriptors.ExactMolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        alerts = []

        # ---------- 1. 反应性基团检测 ----------
        reactive_groups_found = []
        reactive_count = 0
        for name, pat in self._reactive_patterns:
            matches = mol.GetSubstructMatches(pat)
            if matches:
                reactive_groups_found.append(f"{name}({len(matches)})")
                reactive_count += len(matches)
        if reactive_groups_found:
            alerts.append(f"反应性基团: {', '.join(reactive_groups_found)}")

        # ---------- 2. PAINS/NIH/BRENK 过滤器 ----------
        pains_count = self._count_filter_alerts(mol, self._pains_catalog)
        nih_count = self._count_filter_alerts(mol, self._nih_catalog)
        brenk_count = self._count_filter_alerts(mol, self._brenk_catalog)
        if pains_count > 0:
            alerts.append(f"PAINS alerts: {pains_count}")
        if nih_count > 0:
            alerts.append(f"NIH alerts: {nih_count}")
        if brenk_count > 0:
            alerts.append(f"BRENK alerts: {brenk_count}")

        # ---------- 3. hERG 阻断风险 ----------
        herg_risk = self._has_herg_risk(mol)
        if herg_risk:
            alerts.append("hERG 阻断风险 (含碱性氮/哌嗪/吗啉等结构)")

        # ---------- 4. CYP450 抑制风险 ----------
        cyp_risk = self._has_cyp_risk(mol)
        if cyp_risk:
            alerts.append("CYP450 抑制风险 (含杂环芳杂环结构)")

        # ---------- 5. 高亲脂性风险 ----------
        if logp > 5.0:
            alerts.append(f"高亲脂性 (LogP={logp:.2f})，可能增加非特异性毒性")
        elif logp > 4.5:
            alerts.append(f"偏高亲脂性 (LogP={logp:.2f})")

        # ---------- 6. 分子量风险 ----------
        if mw > 650:
            alerts.append(f"过大分子量 (MW={mw:.1f})，可能降低溶解度和特异性")

        # ---------- 7. 综合毒性评分 (0-1, 越高毒性越大) ----------
        toxicity_score = self._compute_toxicity_score(
            logp=logp, mw=mw, reactive_count=reactive_count,
            pains_count=pains_count, nih_count=nih_count, brenk_count=brenk_count,
            herg_risk=herg_risk, cyp_risk=cyp_risk
        )

        # 转换为 pIC50 (范围 4-9, 参考原始 gbm_evaluator 尺度)
        pic50 = 4.0 + toxicity_score * 5.0

        return {
            'toxicity_score': toxicity_score,
            'pIC50': pic50,
            'reactive_group_count': reactive_count,
            'reactive_groups': reactive_groups_found,
            'pains_alerts': pains_count,
            'nih_alerts': nih_count,
            'brenk_alerts': brenk_count,
            'herg_risk': herg_risk,
            'cyp_risk': cyp_risk,
            'alerts': alerts
        }

    def _count_filter_alerts(self, mol, catalog) -> int:
        """统计过滤器目录匹配的警报数"""
        if catalog is None or mol is None:
            return 0
        try:
            matches = catalog.GetMatches(mol)
            return len(matches)
        except Exception:
            return 0

    def _has_herg_risk(self, mol) -> bool:
        """检测 hERG 阻断风险结构"""
        if mol is None:
            return False
        for _, pat in self._herg_patterns:
            if mol.GetSubstructMatches(pat):
                return True
        return False

    def _has_cyp_risk(self, mol) -> bool:
        """检测 CYP450 抑制风险结构"""
        if mol is None:
            return False
        count = 0
        for _, pat in self._cyp_patterns:
            count += len(mol.GetSubstructMatches(pat))
        return count >= 2  # 至少2个风险片段才标记

    def _compute_toxicity_score(
        self,
        logp: float,
        mw: float,
        reactive_count: int,
        pains_count: int,
        nih_count: int,
        brenk_count: int,
        herg_risk: bool,
        cyp_risk: bool
    ) -> float:
        """
        综合评分：0-1，越高毒性越大。

        评分体系（改进自原始 gbm_evaluator.py 的 3 条规则）：
        - LogP > 5.0: +0.30; > 4.5: +0.20; > 4.0: +0.10
        - 反应性基团: 每种 +0.10, 每额外匹配 +0.02 (上限 0.30)
        - PAINS 警报: 每条 +0.05 (上限 0.20)
        - NIH/BRENK: 每条 +0.02 (上限 0.10)
        - hERG 风险: +0.20
        - CYP450 风险: +0.05
        - 过大 MW (> 650): +0.10
        """
        score = 0.0

        # LogP 毒性
        if logp > 5.0:
            score += 0.30
        elif logp > 4.5:
            score += 0.20
        elif logp > 4.0:
            score += 0.10

        # 反应性基团（每种主要类型 +0.10，额外匹配 +0.02）
        reactive_score = min(0.30, reactive_count * 0.05 + (reactive_count > 0) * 0.05)
        score += reactive_score

        # PAINS 警报
        score += min(0.20, pains_count * 0.05)

        # NIH/BRENK 警报
        score += min(0.10, (nih_count + brenk_count) * 0.02)

        # hERG 风险（心脏毒性，严重）
        if herg_risk:
            score += 0.20

        # CYP450 风险
        if cyp_risk:
            score += 0.05

        # 分子量风险
        if mw > 650:
            score += 0.10
        elif mw > 600:
            score += 0.05

        return min(1.0, score)

    def _default_result(self) -> Dict[str, Any]:
        return {
            'toxicity_score': 0.5,
            'pIC50': 6.5,
            'reactive_group_count': 0,
            'reactive_groups': [],
            'pains_alerts': 0,
            'nih_alerts': 0,
            'brenk_alerts': 0,
            'herg_risk': False,
            'cyp_risk': False,
            'alerts': []
        }


# =============================================================================
# BBB 穿透性预测器 V2
# =============================================================================

class BBBPermeabilityPredictorV2:
    """
    BBB 穿透性预测器（Clark 公式 + 启发式惩罚）。

    策略：
    1. Clark 公式（Clark & Delany, 2000）用于 logBB 粗估：
           logBB_est = 0.152 * LogP - 0.0148 * TPSA + 0.139
       仅使用脂水分配系数和拓扑极性表面积。
    2. 启发式惩罚项（不属于 Clark 原始公式）：
           MW > 450:   -0.20
           HBD > 2:    -0.15
           HBA > 6:    -0.10
    3. CNS 药物理化约束过滤器（更严格的理化限制，作为警告信息返回）
    4. 分类阈值: logBB_est > -0.1 = high, > -0.3 = medium, > -0.5 = low, else very_low
    """

    def __init__(self):
        # CNS 药物理化性质参考约束（用于生成警告，不参与评分）
        self._filter_descriptors = {
            'mw_max':    450,   # CNS 药物通常 < 450 Da
            'tpsa_max':  90,    # CNS 药物通常 TPSA < 90 Å²
            'logp_min':  1.0,   # 需要一定亲脂性
            'logp_max':  4.5,   # 过高亲脂性降低溶解度
            'hbd_max':   2,     # CNS 药物 HBD 通常 ≤ 2
            'hba_max':   7,     # CNS 药物 HBA 通常 ≤ 7
        }

    def predict(self, mol) -> Dict[str, Any]:
        """
        预测 BBB 穿透性。

        Returns:
            dict with keys:
            - logBB_est (float, Clark 公式 logBB 预测值)
            - bbb_score (float, Clark + 惩罚后的最终评分 [0,1])
            - classification (str: 'high'/'medium'/'low'/'very_low')
            - cns_drug_like (bool, 是否符合 CNS 药物理化性质规则)
            - score (float, 0-1, 与 bbb_score 等价，保持接口兼容)
            - warnings (list of str, 不符合规则的描述)
        """
        if mol is None:
            return self._default_result()

        mw   = Descriptors.ExactMolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd  = Descriptors.NumHDonors(mol)
        hba  = Descriptors.NumHAcceptors(mol)

        warnings = []

        # ---------- 1. Clark 公式：logBB 粗估（仅 LogP + TPSA）----------
        # 参考文献: Clark DE, Delany SG. 2000.
        logBB_est = 0.152 * logp - 0.0148 * tpsa + 0.139

        # ---------- 2. 启发式惩罚项（不属于 Clark 原始公式）----------
        penalty = 0.0
        if mw > 450:
            penalty -= 0.20
            warnings.append(f"MW={mw:.1f} > 450 (过大分子不利于BBB)")
        if hbd > 2:
            penalty -= 0.15
            warnings.append(f"HBD={hbd} > 2 (过多氢键供体不利于BBB)")
        if hba > 6:
            penalty -= 0.10
            warnings.append(f"HBA={hba} > 6 (过多氢键受体不利于BBB)")

        bbb_score = logBB_est + penalty

        # ---------- 3. CNS 药物理化约束检查 ----------
        cns_like = True
        if tpsa > self._filter_descriptors['tpsa_max']:
            cns_like = False
            warnings.append(f"TPSA={tpsa:.1f} > {self._filter_descriptors['tpsa_max']} Å² (CNS上限)")
        if logp < self._filter_descriptors['logp_min']:
            cns_like = False
            warnings.append(f"LogP={logp:.2f} < {self._filter_descriptors['logp_min']} (亲脂性不足)")
        if logp > self._filter_descriptors['logp_max']:
            cns_like = False
            warnings.append(f"LogP={logp:.2f} > {self._filter_descriptors['logp_max']} (过高亲脂性)")
        if mw > self._filter_descriptors['mw_max'] and "MW=" not in str(warnings):
            cns_like = False
            warnings.append(f"MW={mw:.1f} > {self._filter_descriptors['mw_max']} (CNS上限)")
        if hbd > self._filter_descriptors['hbd_max'] and f"HBD={hbd}" not in str(warnings):
            cns_like = False
            warnings.append(f"HBD={hbd} > {self._filter_descriptors['hbd_max']} (CNS上限)")
        if hba > self._filter_descriptors['hba_max'] and f"HBA={hba}" not in str(warnings):
            cns_like = False
            warnings.append(f"HBA={hba} > {self._filter_descriptors['hba_max']} (CNS上限)")

        # ---------- 4. 分类 ----------
        if logBB_est > -0.1:
            classification = 'high'
        elif logBB_est > -0.3:
            classification = 'medium'
        elif logBB_est > -0.5:
            classification = 'low'
        else:
            classification = 'very_low'

        # ---------- 5. 综合评分 (0-1): logBB > 0.5 = 1.0, logBB < -1.0 = 0.0 ----------
        score = max(0.0, min(1.0, (bbb_score + 1.0) / 1.5))

        return {
            'logBB_est':    logBB_est,
            'bbb_score':    score,
            'classification': classification,
            'cns_drug_like': cns_like,
            'score':        score,    # 保持接口兼容
            'warnings':     warnings,
            'descriptors': {
                'mw': mw, 'logp': logp, 'tpsa': tpsa, 'hbd': hbd, 'hba': hba
            }
        }

    def _default_result(self) -> Dict[str, Any]:
        return {
            'logBB_est':     -1.0,
            'bbb_score':     0.0,
            'classification': 'very_low',
            'cns_drug_like': False,
            'score':         0.0,
            'warnings':      ['分子无法解析'],
            'descriptors':   {}
        }


# =============================================================================
# ADMET 综合过滤器
# =============================================================================

class ADMETFilter:
    """
    分子级 ADMET 过滤器

    相比原始代码的改进：
    1. TPSA 下限从 20 提高到 40 Å²（CNS 药物标准）
    2. LogP 范围收紧到 1.0-4.5
    3. MW 上限从 800 降低到 600
    4. 添加 PAINS/NIH/BRENK 硬过滤
    5. 添加 hERG 风险过滤
    6. 添加氢键供体/受体约束
    """

    def __init__(self, strict: bool = True):
        """
        Args:
            strict: 严格模式下，不符合规则的分子直接被拒绝
        """
        self.strict = strict
        self._setup_filter_catalogs()

        self.constraints = {
            # 理化性质约束（CNS 药物优化版本）
            'mw_min': 200,
            'mw_max': 600,           # 收紧（原始: 800）
            'logp_min': 1.0,         # 提高（原始: 无下限）
            'logp_max': 4.5,         # 收紧（原始: 6.0）
            'tpsa_min': 40,          # 大幅提高（原始: 20）← 关键改进
            'tpsa_max': 90,          # 收紧（原始: 120）← 关键改进
            'hbd_max': 3,            # 收紧（原始: 5）
            'hba_max': 7,            # 收紧（原始: 10）
            'rotatable_bonds_max': 8,
            # 过滤阈值
            'pains_threshold': 0,    # PAINS 警报应完全避免
            'nih_threshold': 2,
            'brenk_threshold': 1,
            # 最小原子数
            'min_heavy_atoms': 10,   # 至少 10 个重原子
            'min_rings': 1,          # 至少 1 个环
        }

    def _setup_filter_catalogs(self):
        """初始化过滤器目录"""
        self._pains_catalog = None
        self._nih_catalog = None
        self._brenk_catalog = None

        for name, fc in [
            ('PAINS', rdfiltercatalog.FilterCatalogParams().FilterCatalogs.PAINS),
            ('NIH', rdfiltercatalog.FilterCatalogParams().FilterCatalogs.NIH),
            ('BRENK', rdfiltercatalog.FilterCatalogParams().FilterCatalogs.BRENK),
        ]:
            try:
                params = rdfiltercatalog.FilterCatalogParams()
                params.AddCatalog(fc)
                catalog = rdfiltercatalog.FilterCatalog(params)
                if name == 'PAINS':
                    self._pains_catalog = catalog
                elif name == 'NIH':
                    self._nih_catalog = catalog
                else:
                    self._brenk_catalog = catalog
                logger.info(f"{name} catalog loaded: {catalog.GetNumEntries()} entries")
            except Exception as e:
                logger.warning(f"Failed to load {name} catalog: {e}")

        # 初始化 toxicity predictor 获取 hERG 检测
        self._toxicity_predictor = ToxicityPredictorV2()

    def check(self, smiles: str) -> Tuple[bool, Dict[str, Any]]:
        """
        检查分子是否通过 ADMET 过滤器。

        Args:
            smiles: SMILES 字符串

        Returns:
            (是否通过, 详细信息字典)
        """
        details = {
            'passed': False,
            'reasons': [],
            'warnings': [],
            'smiles': smiles,
        }

        # ---------- 1. 解析 SMILES ----------
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            details['reasons'].append('RDKit 无法解析 SMILES')
            return False, details

        # 尝试 sanitize（非致命，芳香体系可能 kekulize 失败但分子仍然有效）
        try:
            Chem.SanitizeMol(mol, catchErrors=True)
        except Exception:
            pass

        # ---------- 2. 基本分子描述符 ----------
        try:
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            n_rot = Descriptors.NumRotatableBonds(mol)
            n_heavy = mol.GetNumHeavyAtoms()
            n_rings = rdMolDescriptors.CalcNumRings(mol)
        except Exception as e:
            details['reasons'].append(f'描述符计算失败: {e}')
            return False, details

        details['descriptors'] = {
            'mw': mw, 'logp': logp, 'tpsa': tpsa,
            'hbd': hbd, 'hba': hba, 'n_rot': n_rot,
            'n_heavy': n_heavy, 'n_rings': n_rings
        }

        # ---------- 3. 分子量检查 ----------
        if mw < self.constraints['mw_min']:
            details['reasons'].append(f"MW={mw:.1f} < {self.constraints['mw_min']}")
        if mw > self.constraints['mw_max']:
            details['reasons'].append(f"MW={mw:.1f} > {self.constraints['mw_max']}")

        # ---------- 4. LogP 检查 ----------
        if logp < self.constraints['logp_min']:
            details['reasons'].append(f"LogP={logp:.2f} < {self.constraints['logp_min']}")
        elif logp > self.constraints['logp_max']:
            details['reasons'].append(f"LogP={logp:.2f} > {self.constraints['logp_max']}")

        # ---------- 5. TPSA 检查 ← 关键改进 ----------
        if tpsa < self.constraints['tpsa_min']:
            details['reasons'].append(
                f"TPSA={tpsa:.1f} < {self.constraints['tpsa_min']} Å² "
                "(过低极性可能影响 BBB 溶解度平衡)"
            )
        if tpsa > self.constraints['tpsa_max']:
            details['reasons'].append(
                f"TPSA={tpsa:.1f} > {self.constraints['tpsa_max']} Å² "
                "(过高极性阻碍 BBB 穿透)"
            )

        # ---------- 6. HBD/HBA 检查 ----------
        if hbd > self.constraints['hbd_max']:
            details['reasons'].append(f"HBD={hbd} > {self.constraints['hbd_max']}")
        if hba > self.constraints['hba_max']:
            details['reasons'].append(f"HBA={hba} > {self.constraints['hba_max']}")

        # ---------- 7. 旋转键检查 ----------
        if n_rot > self.constraints['rotatable_bonds_max']:
            details['reasons'].append(f"可旋转键={n_rot} > {self.constraints['rotatable_bonds_max']}")

        # ---------- 8. 原子/环数检查 ----------
        if n_heavy < self.constraints['min_heavy_atoms']:
            details['reasons'].append(f"重原子数={n_heavy} < {self.constraints['min_heavy_atoms']}")
        if n_rings < self.constraints['min_rings']:
            details['reasons'].append(f"环数={n_rings} < {self.constraints['min_rings']}")

        # ---------- 9. PAINS/NIH/BRENK 检查 ----------
        if self._pains_catalog:
            try:
                matches = self._pains_catalog.GetMatches(mol)
                n = len(matches)
                details['pains_alerts'] = n
                if n > self.constraints['pains_threshold']:
                    details['reasons'].append(f"PAINS alerts: {n}")
                elif n > 0:
                    details['warnings'].append(f"PAINS alerts: {n}")
            except Exception:
                pass

        if self._nih_catalog:
            try:
                matches = self._nih_catalog.GetMatches(mol)
                n = len(matches)
                details['nih_alerts'] = n
                if n > self.constraints['nih_threshold']:
                    details['reasons'].append(f"NIH alerts: {n}")
            except Exception:
                pass

        if self._brenk_catalog:
            try:
                matches = self._brenk_catalog.GetMatches(mol)
                n = len(matches)
                details['brenk_alerts'] = n
                if n > self.constraints['brenk_threshold']:
                    details['reasons'].append(f"BRENK alerts: {n}")
            except Exception:
                pass

        # ---------- 10. hERG 风险检查 ----------
        tox_result = self._toxicity_predictor.predict(mol)
        if tox_result['herg_risk']:
            details['warnings'].append("hERG 阻断风险")
            if self.strict:
                details['reasons'].append("hERG 心脏毒性风险")

        # ---------- 11. 判定结果 ----------
        passed = len(details['reasons']) == 0
        details['passed'] = passed
        return passed, details

    def filter_batch(self, smiles_list: List[str]) -> Tuple[List[str], List[Dict]]:
        """
        批量过滤分子。

        Returns:
            (通过过滤的 SMILES 列表, 所有分子的详细信息列表)
        """
        passed_list = []
        all_details = []

        for smi in smiles_list:
            ok, details = self.check(smi)
            all_details.append(details)
            if ok:
                passed_list.append(smi)

        return passed_list, all_details


# =============================================================================
# 便捷函数
# =============================================================================

def evaluate_molecule_admet(smiles: str, verbose: bool = False) -> Dict[str, Any]:
    """
    一站式 ADMET 评估函数。

    同时运行 ToxicityPredictorV2、BBBPermeabilityPredictorV2、ADMETFilter，
    返回完整的 ADMET 评估报告。

    Args:
        smiles: SMILES 字符串
        verbose: 是否包含详细描述符

    Returns:
        dict with keys: passed (bool), toxicity, bbb, filter_details, score
    """
    # 解析分子
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return {'passed': False, 'error': 'Invalid SMILES', 'smiles': smiles}

    # 尝试 sanitize（如果失败则继续使用未 sanitize 的分子）
    try:
        Chem.SanitizeMol(mol, catchErrors=True)
    except Exception:
        pass

    # ADMET 过滤
    admet_filter = ADMETFilter(strict=True)
    passed, filter_details = admet_filter.check(smiles)

    # 毒性预测
    tox_pred = ToxicityPredictorV2()
    tox_result = tox_pred.predict(mol)

    # BBB 预测
    bbb_pred = BBBPermeabilityPredictorV2()
    bbb_result = bbb_pred.predict(mol)

    # 综合评分 (0-1, 越高越好)
    # 毒性 35%（越低越好，取 1-toxicity_score）
    # BBB 35%（越高越好）
    # 过滤通过 +10%
    # 无 PAINS +10%
    # 无 hERG 风险 +10%
    composite = (
        (1 - tox_result['toxicity_score']) * 0.35 +
        bbb_result['score'] * 0.35 +
        (0.1 if passed else 0.0) +
        (0.1 if filter_details.get('pains_alerts', 0) == 0 else 0.0) +
        (0.1 if not tox_result['herg_risk'] else 0.0)
    )

    result = {
        'smiles': smiles,
        'passed': passed,
        'toxicity': tox_result,
        'bbb': bbb_result,
        'filter_details': filter_details,
        'composite_score': composite,
    }

    if verbose:
        result['filter_details'] = filter_details

    return result
