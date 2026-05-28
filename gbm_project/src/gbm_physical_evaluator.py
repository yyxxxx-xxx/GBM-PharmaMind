"""
GBM External Physical Evaluator
===============================
A "外科手术式" replacement for LLM-based molecule evaluation (the "toxic tumor").

Design principles (参考 SyntheMol + chemcrow + REINVENT4):
- SyntheMol pattern: SMILES saved to file -> subprocess.run() -> parse float score
- chemcrow pattern: _run() receives SMILES -> returns natural-language string feedback
- REINVENT4 pattern: 每个属性有独立打分组件 -> 聚合函数(加权几何平均) -> MPO Reward

ABSOLUTE PRINCIPLE: Do NOT touch any LLM generation or knowledge-base logic.
This module is purely computational, callable from any evaluation point.
"""

from __future__ import annotations

import os
import sys
import json
import time
import logging
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Literal
from enum import Enum

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, AllChem

logger = logging.getLogger(__name__)


# =============================================================================
# Section 1: 数据结构 — 物理评估结果
# =============================================================================

class EvaluationVerdict(Enum):
    """ToT 树节点的评估结论，对应原来的 sure/likely/impossible"""
    SURE = "sure"        # 物理指标全部合格，强烈推荐继续
    LIKELY = "likely"    # 有轻微问题但值得探索
    IMPOSSIBLE = "impossible"  # 有硬截断问题，该分支应被剪枝


@dataclass
class PhysicalEvaluationResult:
    """
    单个分子的完整物理评估结果。
    包含原始数值、归一化分数、Reward、以及自然语言反馈。
    """
    smiles: str

    # ---- 原始物理量（从外部计算脚本获取）----
    vina_score: Optional[float] = None       # QVina/Wina 对接分数 (kcal/mol)
    dili_prob: Optional[float] = None        # DILI 风险概率 [0, 1]
    herg_prob: Optional[float] = None        # hERG 阻断风险概率 [0, 1]
    bbb_score: Optional[float] = None         # BBB 评分 [0, 1]，Clark logBB + 启发式惩罚

    # ---- RDKit 描述符（本地计算）----
    rd_tpsa: float = 0.0
    rd_mw: float = 0.0
    rd_logp: float = 0.0
    rd_hbd: int = 0
    rd_hba: int = 0
    rd_n_rotatable: int = 0
    rd_n_rings: int = 0

    # ---- 归一化分数 [0, 1]（越高越好）----
    vina_norm: float = 0.0
    dili_norm: float = 1.0      # 1 - dili_prob（越低毒性越好）
    herg_norm: float = 1.0      # 1 - herg_prob
    bbb_norm: float = 0.0
    tpsa_norm: float = 0.0
    mw_norm: float = 0.0

    # ---- 结构警报命中 ----
    dili_alert_matches: List[str] = field(default_factory=list)
    herg_alert_matches: List[str] = field(default_factory=list)

    # ---- 基线毒性（用于 Delta Scoring）----
    baseline_dili: Optional[float] = None
    baseline_herg: Optional[float] = None

    # ---- MPO Reward ----
    reward: float = 0.0        # 最终 Reward
    is_pruned: bool = False    # 是否被硬截断剪枝

    # ---- ToT verdict ----
    verdict: EvaluationVerdict = EvaluationVerdict.LIKELY
    prune_reason: str = ""      # 剪枝原因（如果有的话）

    # ---- 内部元数据 ----
    vina_timeout: bool = False
    vina_error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'smiles': self.smiles,
            'vina_score': self.vina_score,
            'dili_prob': self.dili_prob,
            'herg_prob': self.herg_prob,
            'bbb_score': self.bbb_score,
            'rd_tpsa': round(self.rd_tpsa, 2),
            'rd_mw': round(self.rd_mw, 2),
            'rd_logp': round(self.rd_logp, 2),
            'vina_norm': round(self.vina_norm, 4),
            'dili_norm': round(self.dili_norm, 4),
            'herg_norm': round(self.herg_norm, 4),
            'bbb_norm': round(self.bbb_norm, 4),
            'reward': round(self.reward, 4),
            'is_pruned': self.is_pruned,
            'prune_reason': self.prune_reason,
            'verdict': self.verdict.value,
            'dili_alert_matches': self.dili_alert_matches,
            'herg_alert_matches': self.herg_alert_matches,
        }

    def build_feedback_text(self) -> str:
        """
        将物理评估结果组装成自然语言反馈 Prompt（注入给 LLM）。

        参考 chemcrow/tools/rdkit.py 的 _run() 返回格式：
        返回一段描述性文本，LLM 可据此继续优化。
        """
        parts = []

        # Vina 得分
        if self.vina_score is not None:
            vina_str = f"Vina得分={self.vina_score:.2f} kcal/mol"
            if self.vina_score > -7.0:
                vina_str += " (偏弱，需增强结合)"
            elif self.vina_score < -10.0:
                vina_str += " (强结合)"
            else:
                vina_str += " (中等结合)"
            parts.append(vina_str)
        elif self.vina_error:
            parts.append(f"Vina计算失败: {self.vina_error}")
        else:
            parts.append("Vina得分: N/A")

        # DILI
        if self.dili_prob is not None:
            dili_str = f"肝毒性(DILI)={self.dili_prob:.2f}"
            if self.dili_prob > 0.5:
                dili_str += " (警告: 高风险)"
            elif self.dili_prob > 0.3:
                dili_str += " (注意: 中等风险)"
            else:
                dili_str += " (低风险)"
            parts.append(dili_str)

        # BBB
        if self.bbb_score is not None:
            bbb_str = f"BBB评分={self.bbb_score:.2f}"
            if self.bbb_score > 0.7:
                bbb_str += " (高CNS渗透)"
            elif self.bbb_score > 0.4:
                bbb_str += " (中等CNS渗透)"
            else:
                bbb_str += " (低CNS渗透，不适合GBM)"
            parts.append(bbb_str)

        # 理化性质摘要
        parts.append(
            f"理化性质: MW={self.rd_mw:.1f}, LogP={self.rd_logp:.2f}, "
            f"TPSA={self.rd_tpsa:.1f}Å²"
        )

        # Reward
        parts.append(f"Reward={self.reward:.4f}")

        # 剪枝提示
        if self.is_pruned:
            parts.append(f"[警告: 此分支已被硬截断剪枝: {self.prune_reason}]")

        return " | ".join(parts)


# =============================================================================
# Section 2: Vina 对接工具 (参考 SyntheMol generate/scorer.py 的外部工具模式)
# =============================================================================

class VinaDockingTool:
    """
    Vina/QVina 对接工具。

    参考 SyntheMol 的外部评分器模式：
    1. 将 SMILES 写入临时 PDBQT 文件（或使用预生成 PDBQT）
    2. subprocess.run() 调用 QVina / Smina
    3. 解析 stdout 获取 float 对接分数

    使用方式:
        tool = VinaDockingTool(
            vina_executable="/path/to/qvina",
            receptor_pdbqt="/path/to/receptor.pdbqt",
            center=[x, y, z],
            size=[20, 20, 20]
        )
        score = tool.dock("CCO")  # 返回对接分数
    """

    def __init__(
        self,
        vina_executable: Optional[str] = None,
        receptor_pdbqt: Optional[str] = None,
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        size: Tuple[float, float, float] = (20.0, 20.0, 20.0),
        timeout: int = 30,
        exhaustiveness: int = 8,
        n_poses: int = 1,
    ):
        """
        Args:
            vina_executable: QVina/Wina 可执行文件路径。如果为 None，使用伪计算模式。
            receptor_pdbqt: 受体 PDBQT 文件路径（预准备好的）
            center: 对接盒子中心坐标 (x, y, z)
            size: 对接盒子尺寸 (width, height, depth)
            timeout: 单次对接超时（秒）
            exhaustiveness: Vina exhaustiveness 参数
            n_poses: 返回的 poses 数量
        """
        self.vina_executable = vina_executable
        self.receptor_pdbqt = receptor_pdbqt
        self.center = center
        self.size = size
        self.timeout = timeout
        self.exhaustiveness = exhaustiveness
        self.n_poses = n_poses

        self._enabled = (
            vina_executable is not None
            and os.path.isfile(vina_executable)
            and receptor_pdbqt is not None
            and os.path.isfile(receptor_pdbqt)
        )
        if not self._enabled:
            logger.warning(
                "[VinaDockingTool] Vina not configured or files missing. "
                "Using pseudo-score mode based on MW/LogP. "
                "Set vina_executable and receptor_pdbqt for real docking."
            )

    def _smiles_to_pdbqt(self, smiles: str, output_path: str) -> bool:
        """
        将 SMILES 转换为 PDBQT 文件。

        使用 RDKit 生成 3D 构象，然后导出为 PDBQT 格式。
        如果没有外部工具（如 obabel），使用 RDKit 内置方法。

        Returns:
            True if conversion successful, False otherwise.
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False

            # 生成 3D 构象
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.UFFOptimizeMolecule(mol)

            # 写 PDBQT（需要 obabel 或手动转换）
            # 方案A: 尝试使用 obabel
            try:
                proc = subprocess.run(
                    [
                        "obabel", "-ipdb",
                        self._write_pdb(mol),
                        "-opdbqt",
                        "-O", output_path,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                return proc.returncode == 0
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

            # 方案B: 直接写简化 PDBQT（RDKit 不直接支持 PDBQT，用近似方法）
            # 此处为占位：实际需要调用 prepare_ligand.py 或 Meeko
            pdb_content = self._rdkit_to_pdb(mol)
            with open(output_path, 'w') as f:
                f.write(pdb_content)
            return True

        except Exception as e:
            logger.debug(f"[Vina] SMILES->PDBQT failed: {e}")
            return False

    def _write_pdb(self, mol) -> str:
        """将 RDKit mol 写入临时 PDB 文件"""
        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False, mode='w') as f:
            block = Chem.MolToPDBBlock(mol)
            f.write(block)
            return f.name

    def _rdkit_to_pdb(self, mol) -> str:
        """RDKit Mol -> PDB 字符串（简化版，原子类型不全）"""
        try:
            return Chem.MolToPDBBlock(mol)
        except Exception:
            return ""

    def dock(self, smiles: str) -> Tuple[Optional[float], str]:
        """
        对接单个分子并返回分数。

        Returns:
            (vina_score, error_message)
            vina_score: kcal/mol，越负越好（通常 -12 ~ -4）
            error_message: 如果出错，返回错误描述
        """
        if not self._enabled:
            return self._pseudo_dock(smiles), ""

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                ligand_path = os.path.join(tmpdir, "ligand.pdbqt")

                # Step 1: SMILES -> PDBQT
                if not self._smiles_to_pdbqt(smiles, ligand_path):
                    return None, "SMILES to PDBQT conversion failed"

                # Step 2: 构建 Vina 命令
                cmd = [
                    self.vina_executable,
                    "--receptor", self.receptor_pdbqt,
                    "--ligand", ligand_path,
                    "--center_x", str(self.center[0]),
                    "--center_y", str(self.center[1]),
                    "--center_z", str(self.center[2]),
                    "--size_x", str(self.size[0]),
                    "--size_y", str(self.size[1]),
                    "--size_z", str(self.size[2]),
                    "--exhaustiveness", str(self.exhaustiveness),
                    "--num_modes", str(self.n_poses),
                    "--out", os.path.join(tmpdir, "out.pdbqt"),
                ]

                # Step 3: subprocess.run() 调用 Vina（参考 SyntheMol scorer.py 模式）
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    check=False,
                )

                if result.returncode != 0:
                    return None, f"Vina exit code {result.returncode}: {result.stderr[:200]}"

                # Step 4: 解析输出中的 affinity (kcal/mol)
                return self._parse_vina_output(result.stdout + result.stderr), ""

        except subprocess.TimeoutExpired:
            return None, "Vina timeout"
        except Exception as e:
            return None, str(e)

    def _parse_vina_output(self, output: str) -> Optional[float]:
        """从 Vina stdout/stderr 中解析最优结合分数"""
        import re
        # Vina 输出格式: "RESULT:  ...   -9.2   0.000  2.345\n"
        # 或 "1       -8.7  0.000  2.1 ..." 行
        for line in output.split('\n'):
            line = line.strip()
            # 找第一列是数字、第二列也是数字的模式
            m = re.search(r'^\s*\d+\s+([-+]?\d+\.\d+)', line)
            if m:
                score = float(m.group(1))
                logger.debug(f"[Vina] Parsed score: {score}")
                return score
        # 备选: 直接搜索 "-9.2" 这样的模式
        m = re.search(r'([-+]?\d+\.\d+)\s+(?:kcal|/mol)', output)
        if m:
            return float(m.group(1))
        return None

    def _pseudo_dock(self, smiles: str) -> Optional[float]:
        """
        伪对接模式：当没有真实 Vina 时使用。

        基于分子描述符估算 Vina 分数（占位实现）。
        参考：Vina ~ -0.1*MW + 0.5*LogP - 0.01*TPSA - 6.0 (经验公式，GBM靶点需调参)

        此伪函数确保整个流程在没有 Vina 的环境下也能端到端运行。
        实际使用时替换为真实对接。
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)

            # 经验公式（需要根据具体靶点/PDB 结构标定）
            pseudo_vina = (
                -0.08 * mw
                + 0.4 * logp
                - 0.008 * tpsa
                - 5.5
            )
            # 限制在合理范围
            return max(-12.0, min(-4.0, pseudo_vina))
        except Exception:
            return None


# =============================================================================
# Section 3: ADMET 计算器 (DILI + BBB 概率)
# =============================================================================

class ADMETCalculator:
    """
    ADMET 属性计算器。

    提供 DILI_prob 和 BBB_prob 两个关键概率。

    实现参考：
    - DILI: 基于分子描述符的启发式评分（结合 ToxicityPredictorV2 的反应性基团检测）
    - BBB: 使用 BBBPermeabilityPredictorV2 的 logBB 模型

    真实部署时可替换为：
    - DILI: pkADMET、ADMETlab、PreADMET 等 API/模型
    - BBB: 真实 logBB 预测模型（如 CNN-based）
    """

    # DILI high-risk SMARTS patterns (genuine toxicophores only)
    # NOTE: Acrylamide/cyanoacrylamide warheads are excluded here because they are
    # intentional covalent fragments in EGFR inhibitors (e.g., Afatinib, Osimertinib).
    # They are handled separately via EGFR_WARHEAD_SMARTS for exemption logic.
    DILI_ALERT_SMARTS = [
        r'C(=O)Cl',            # acyl chloride (high reactivity)
        r'N=C=S',              # isothiocyanate
        r'C1CO1',              # epoxide (unsubstituted, highly reactive)
        r'[cx]1([N+](=O)[O-])ccc1',  # dinitroaryl (multiple nitro groups)
        r'C1(=O)ccc(=O)c1',   # quinone
        r'OP(=O)(O)O',        # phosphate ester (fully hydrolyzable)
        r'[N-]=[N+]=C',      # diazonium salt (valid: connects to carbon)
    ]

    # DILI alert metadata: human-readable name + recommended fix
    DILI_ALERT_META: Dict[str, Dict[str, str]] = {
        r'C(=O)Cl': {
            'name': '酰氯基团 (Acyl Chloride)',
            'fix': '用酰胺（-C(=O)NH-）或酯（-C(=O)O-）替换酰氯，以降低反应活性。',
        },
        r'N=C=S': {
            'name': '异硫氰酸酯 (Isothiocyanate)',
            'fix': '异硫氰酸酯具有强亲电性。建议使用碳酰胺（-NH-C(=O)-）替代，或完全去除该基团。',
        },
        r'C1CO1': {
            'name': '环氧烷 (Unsubstituted Epoxide)',
            'fix': '未取代的环氧烷是高风险基因毒性警示结构。建议使用内酰胺环或饱和杂环替代。',
        },
        r'[cx]1([N+](=O)[O-])ccc1': {
            'name': '二硝基芳烃 (Dinitroaryl)',
            'fix': '多硝基芳香族化合物具有强氧化性和潜在基因毒性。建议减少硝基数量或用磺酰胺替代。',
        },
        r'C1(=O)ccc(=O)c1': {
            'name': '醌类 (Quinone)',
            'fix': '醌类代谢物可产生活性氧(ROS)，引发氧化应激损伤。建议将苯醌替换为氢醌二甲醚或饱和环己酮结构。',
        },
        r'OP(=O)(O)O': {
            'name': '磷酸酯 (Phosphate Ester)',
            'fix': '完全可水解的磷酸酯具有急性毒性风险。建议使用膦酸酯或完全去除该基团。',
        },
        r'[N-]=[N+]=C': {
            'name': '重氮盐 (Diazonium Salt)',
            'fix': '重氮盐是强亲电试剂，可与DNA发生反应。请用稳定的氨基（-NH2）替代。',
        },
    }

    # EGFR covalent inhibitor warhead SMARTS — intentionally used, exempt from DILI penalty
    # These are NOT toxicophores in the context of GBM/EGFR covalent drug design.
    EGFR_WARHEAD_SMARTS = [
        r'C=CC(=O)N',           # acrylamide (primary EGFR warhead, e.g., Afatinib)
        r'C=CC(N)=O',           # substituted acrylamide
        r'C=CC#N',              # acrylonitrile variant
        r'C=CC(=O)C=C',        # diacrylamide / bismaleimide
        r'c1ccc(NC(=O)C=C)cc1',  # aryl acrylamide (common EGFR scaffold)
        r'NC(=O)C=C',           # any acrylamide fragment
        r'N#CC=C',              # cyanoacrylamide warhead (e.g., WZ8040)
    ]

    # hERG channel blocking SMARTS patterns
    # Based on: Aronov & Goldman "Predictive models for hERG blockade",
    # Cavalli et al. SAR analysis of known hERG blockers, and medicinal chemistry literature.
    # Key structural features: lipophilic bases, aryls with heteroatoms, high logP regions.
    hERG_ALERT_SMARTS = [
        (r'[NX3;H2,H1;!$(NC=O)]-[CX4H2]-[CX4H2]-[cR1]', '脂溶性三级胺侧链'),  # lipophilic tertiary amine
        (r'[cR1]1[cR1][cR1]([N+!0;D1])[cR1][cR1]1', '氮杂芳环上的取代基'),  # heteroaryl with basic N
        (r'c1ccc2c(c1)cccc2', '萘环 (Naphthalene)'),  # naphthalene scaffold
        (r'c1ccc2ccccc2c1', '萘骨架 (Naphthalene skeleton)'),  # naphthalene (alternative)
        (r'[cR1]1[cR1][cR1][cR1]([N+](=O)[O-])[cR1][cR1]1', '硝基芳香族 (Nitroaryl)'),  # nitroaryl
        (r'[cR1]1[cR1][cR1][cR1][cR1][cR1]1', '未取代苯环 (bare phenyl, high LogP risk)'),  # unsubstituted phenyl (high LogP)
        (r'[CX4H3]-[CX4H2]-[NX3;H2,H1]', '烷基胺侧链'),  # alkyl amine
        (r'[cR1]1[cR1][cR1]([OX2H1])[cR1][cR1][cR1]1', '苯酚/羟基芳香族'),  # phenols (lipophilic)
        (r'c1ccc(C(F)(F)F)cc1', '三氟甲基苯 (Trifluoromethylphenyl)'),  # CF3-aryl
        (r'c1ccc(Cl)cc1', '氯苯 (Chlorophenyl)'),  # chlorophenyl
        (r'c1ccc(Br)cc1', '溴苯 (Bromophenyl)'),  # bromophenyl
    ]

    # hERG alert metadata
    hERG_ALERT_META: Dict[str, Dict[str, str]] = {
        r'[cR1]1[cR1][cR1]([N+!0;D1])[cR1][cR1]1': {
            'name': '含氮杂芳环取代基',
            'fix': '杂芳环上的氮原子增强了对 hERG 通道的亲和力。建议用氧原子（-O-）或碳原子（-CH2-）替换该位置的氮。',
        },
        r'c1ccc2c(c1)cccc2': {
            'name': '萘环结构',
            'fix': '萘环是经典的高 hERG 风险骨架。请将萘替换为吡啶环、哌嗪或带极性侧链的苯环以降低 LogP。',
        },
        r'[cR1]1[cR1][cR1]([N+](=O)[O-])[cR1][cR1]1': {
            'name': '硝基芳香族',
            'fix': '硝基芳香族同时贡献 hERG 风险和 DILI 风险。建议用磺酰胺（-SO2NH-）或酰胺（-C(=O)NH-）替换硝基。',
        },
        r'[cR1]1[cR1][cR1][cR1][cR1][cR1]1': {
            'name': '未取代苯环 (高 LogP 风险)',
            'fix': '未取代苯环的高 LogP 贡献是 hERG 阻断的重要诱因。请在苯环上引入极性取代基（如 -OH、-NH2、-COOH、-SO2NH2）。',
        },
        r'[CX4H3]-[CX4H2]-[NX3;H2,H1]': {
            'name': '烷基胺侧链',
            'fix': '烷基胺的碱性氮原子是 hERG 结合的常见药效团。建议将末端胺基替换为酰胺或磺酰胺，以降低 pKa。',
        },
        r'c1ccc(C(F)(F)F)cc1': {
            'name': '三氟甲基苯 (CF3-Phenyl)',
            'fix': 'CF3 基团的强吸电子性和高 LogP 贡献共同加剧 hERG 阻断风险。请考虑将 -CF3 替换为 -CHF2 或 -CONH2。',
        },
        r'c1ccc(Cl)cc1': {
            'name': '氯苯基团',
            'fix': '氯苯是常见的 hERG 警示结构。建议将氯原子替换为极性基团如 -OH、-NH2，或移动至分子极性端。',
        },
    }

    def __init__(self):
        self._dili_patterns = []
        for smarts in self.DILI_ALERT_SMARTS:
            try:
                pat = Chem.MolFromSmarts(smarts)
                if pat:
                    self._dili_patterns.append(pat)
            except Exception:
                pass

        self._egfr_warhead_patterns = []
        for smarts in self.EGFR_WARHEAD_SMARTS:
            try:
                pat = Chem.MolFromSmarts(smarts)
                if pat:
                    self._egfr_warhead_patterns.append(pat)
            except Exception:
                pass

        # ---- hERG SMARTS patterns ----
        self._herg_patterns = []
        for smarts, _ in self.hERG_ALERT_SMARTS:
            try:
                pat = Chem.MolFromSmarts(smarts)
                if pat:
                    self._herg_patterns.append((pat, smarts))
            except Exception:
                pass

        # Clark 公式参数已移除；现在使用 Clark 公式直接在 _compute_bbb 中计算

    def compute(self, smiles: str) -> Dict[str, Any]:
        """
        计算 DILI_prob, hERG_prob 和 BBB_prob。

        Returns:
            {'dili_prob': float [0,1],
             'herg_prob': float [0,1],
             'bbb_score': float [0,1],
             'dili_alert_matches': List[str],
             'herg_alert_matches': List[str]}
        """
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return {
                'dili_prob': 0.5, 'herg_prob': 0.5, 'bbb_score': 0.3,
                'dili_alert_matches': [], 'herg_alert_matches': [],
            }

        try:
            Chem.SanitizeMol(mol, catchErrors=True)
        except Exception:
            pass

        # --- DILI ---
        dili_result = self._compute_dili(mol)

        # --- hERG ---
        herg_result = self._compute_herg(mol)

        # --- BBB ---
        bbb_score = self._compute_bbb(mol)

        return {
            'dili_prob': float(dili_result['score']),
            'herg_prob': float(herg_result['score']),
            'bbb_score': float(bbb_score),
            'dili_alert_matches': dili_result['alert_names'],
            'herg_alert_matches': herg_result['alert_names'],
        }

    def _compute_herg(self, mol) -> Dict[str, Any]:
        """
        计算 hERG 阻断风险概率 [0, 1]，并返回命中的警报名称。

        Returns:
            {'score': float, 'alert_names': List[str]}
        """
        score = 0.0
        alert_names: List[str] = []

        # 1. SMARTS 警报匹配
        for pat, smarts in self._herg_patterns:
            if mol.HasSubstructMatch(pat):
                # 找到对应的元数据名称
                meta_name = None
                for key, meta in self.hERG_ALERT_META.items():
                    if key == smarts:
                        meta_name = meta['name']
                        break
                alert_names.append(meta_name or smarts)
                score += 0.20

        # 2. EGFR warhead exemption: acrylamides can still be hERG risks
        # (they are intentionally included, but we reduce the penalty)
        has_egfr_warhead = any(
            mol.HasSubstructMatch(p) for p in self._egfr_warhead_patterns
        )
        if has_egfr_warhead and alert_names:
            # EGFR warheads partially mitigate hERG risk from other moieties
            score *= 0.75

        # 3. High LogP: lipophilicity is a major hERG risk factor
        try:
            logp = Crippen.MolLogP(mol)
            if logp > 5.5:
                score += 0.20
            elif logp > 4.5:
                score += 0.10
        except Exception:
            pass

        # 4. MW penalty: larger molecules with basic nitrogens are higher risk
        try:
            mw = Descriptors.ExactMolWt(mol)
            if mw > 600:
                score += 0.10
        except Exception:
            pass

        # 5. Basic nitrogen count: hERG binders are typically bases with pKa > 7.5
        n_basic = sum(
            1 for pat, _ in self._herg_patterns
            if 'NX3' in str(pat) and mol.HasSubstructMatch(pat)
        )
        if n_basic >= 2:
            score += 0.15

        return {
            'score': min(1.0, max(0.0, score)),
            'alert_names': alert_names,
        }

    def _compute_dili(self, mol) -> Dict[str, Any]:
        """
        计算 DILI 风险概率 [0, 1]，并返回命中的警报名称。

        Returns:
            {'score': float, 'alert_names': List[str]}

        Design principles for GBM/EGFR covalent inhibitor context:
        1. Genuine toxicophores (acyl chloride, isothiocyanate, quinone, etc.) → penalty
        2. EGFR covalent warheads (acrylamide, cyanoacrylamide) → exemption (reduce/ignore penalty)
           These are intentional pharmacophores in approved EGFR inhibitors.
        3. MW / LogP / PAINS → standard penalty
        """
        score = 0.0
        alert_names: List[str] = []

        # --- 1. Check for EGFR covalent warheads (exemption) ---
        has_egfr_warhead = False
        for pat in self._egfr_warhead_patterns:
            if mol.HasSubstructMatch(pat):
                has_egfr_warhead = True
                break

        # --- 2. Genuine toxicophore detection (skip EGFR warheads) ---
        for pat, smarts in zip(self._dili_patterns, self.DILI_ALERT_SMARTS):
            if mol.HasSubstructMatch(pat):
                meta = self.DILI_ALERT_META.get(smarts, {})
                alert_names.append(meta.get('name', smarts))
                score += 0.25
        if len(alert_names) >= 2:
            score += 0.25  # multiple genuine toxicophores = high risk
        alert_names = list(set(alert_names))  # deduplicate

        # --- 3. EGFR warhead exemption: reduce penalty by 75% ---
        if has_egfr_warhead and alert_names:
            score *= 0.25

        # --- 4. MW risk (tightened upper bound for GBM CNS drugs) ---
        try:
            mw = Descriptors.ExactMolWt(mol)
            if mw > 800:
                score += 0.15
            elif mw > 700:
                score += 0.05
        except Exception:
            pass

        # --- 5. LogP risk ---
        try:
            logp = Crippen.MolLogP(mol)
            if logp > 5.5:
                score += 0.10
            elif logp > 4.5:
                score += 0.05
        except Exception:
            pass

        # --- 6. PAINS simplified detection ---
        pains_like = [
            r'c1cc(-n2nc(-c3ccccc3)nc2O)on1',
        ]
        for smarts in pains_like:
            try:
                pat = Chem.MolFromSmarts(smarts)
                if pat and mol.HasSubstructMatch(pat):
                    score += 0.20
                    break
            except Exception:
                pass

        return {
            'score': min(1.0, max(0.0, score)),
            'alert_names': alert_names,
        }

    def _compute_bbb(self, mol) -> float:
        """
        计算 BBB 评分 [0, 1]。

        第一步：Clark 公式（Clark & Delany, 2000）用于 logBB 粗估。
            logBB_est = 0.152 * LogP - 0.0148 * TPSA + 0.139
            仅使用脂水分配系数和拓扑极性表面积。

        第二步：启发式惩罚项（不属于 Clark 原始公式）。
            MW > 450：     -0.20（过大分子不利于 BBB 穿透）
            HBD > 2：      -0.15（过多氢键供体不利于 BBB 穿透）
            HBA > 6：      -0.10（过多氢键受体不利于 BBB 穿透）

        最终 bbb_score = logBB_est + penalty，并 clamp 到 [0, 1]。

        概率映射：logBB > 0.5 -> 1.0, logBB < -1.0 -> 0.0
        """
        try:
            logp  = Crippen.MolLogP(mol)
            tpsa  = Descriptors.TPSA(mol)
            mw    = Descriptors.ExactMolWt(mol)
            hbd   = Descriptors.NumHDonors(mol)
            hba   = Descriptors.NumHAcceptors(mol)
        except Exception:
            return 0.3

        # --- Clark 公式：logBB 粗估（仅 LogP + TPSA）---
        logBB_est = 0.152 * logp - 0.0148 * tpsa + 0.139

        # --- 启发式惩罚项（MW / HBD / HBA，不属于 Clark 原始公式）---
        penalty = 0.0
        if mw > 450:
            penalty -= 0.20
        if hbd > 2:
            penalty -= 0.15
        if hba > 6:
            penalty -= 0.10

        bbb_score = logBB_est + penalty

        # 映射到 [0, 1]: logBB > 0.5 = 1.0, logBB < -1.0 = 0.0
        return max(0.0, min(1.0, (bbb_score + 1.0) / 1.5))


# =============================================================================
# Section 4: 多目标合意度评分器 (MPO Reward Function)
# =============================================================================

class MPORewardCalculator:
    """
    多目标合意度评分器 (Multi-Parameter Optimization Reward)。

    参考 REINVENT4 scoring/aggregators/means.py 的加权几何平均模式：

    Reward = (Vina_norm ** w1) * (DILI_norm ** w2) * (hERG_norm ** w3) * (BBB_norm ** w4) * (TPSA_norm ** w5)

    其中 DILI_norm = 1 - DILI_prob, hERG_norm = 1 - hERG_prob（越低毒性越好，取补数）

    权重设定原则（GBM 场景）：
    - w1 (Vina): 0.30 — 结合亲和力是核心
    - w2 (DILI): 0.20 — 安全性必须保障
    - w3 (hERG): 0.20 — 心脏安全性
    - w4 (BBB):  0.20 — BBB 穿透对 GBM 至关重要
    - w5 (理化): 0.10 — 类药性约束

    硬截断 (Hard Pruning) 参考 REINVENT4 scoring/transforms/steps.py:
    - Vina > -5.0 kcal/mol → 立即剪枝（过弱的结合）
    - DILI_prob > 0.95    → 立即剪枝（过高肝毒性）
    - hERG_prob > 0.90   → 立即剪枝（过高心脏风险）
    - MW < 200 或 MW > 700 → 立即剪枝
    - TPSA < 20 或 TPSA > 120 → 立即剪枝

    Delta Scoring: 如果毒性只比父节点升高 <= 0.05，或有所下降，不触发硬截断。
    """

    # ---- 硬截断阈值（必须满足，否则分支被剪枝）----
    HARD_CUTOFFS = {
        'vina_min': -5.0,
        'dili_max': 0.95,
        'herg_max': 0.90,
        'mw_min': 200.0,
        'mw_max': 700.0,
        'tpsa_min': 20.0,
        'tpsa_max': 140.0,   # death line: > 140 -> hard prune
        'tpsa_valley_max': 10.0,  # hard valley line: < 10 -> hard prune
    }

    # ---- Reward 权重（归一化后相加为 1.0）----
    REWARD_WEIGHTS = {
        'vina': 0.30,
        'dili': 0.20,
        'herg': 0.20,
        'bbb': 0.20,
        'physchem': 0.10,
    }

    def __init__(self, custom_weights: Optional[Dict[str, float]] = None, custom_cutoffs: Optional[Dict[str, float]] = None):
        if custom_weights:
            self.weights = custom_weights
        else:
            self.weights = dict(self.REWARD_WEIGHTS)

        if custom_cutoffs:
            self.cutoffs = {**self.HARD_CUTOFFS, **custom_cutoffs}
        else:
            self.cutoffs = dict(self.HARD_CUTOFFS)

    def check_hard_cutoffs(
        self,
        result: PhysicalEvaluationResult,
        baseline_dili: Optional[float] = None,
        baseline_herg: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        检查硬截断条件，支持 Delta Scoring。

        Delta Scoring: 如果毒性比父节点升高不超过 0.05，或者有所下降，
        即使略微超过硬截断阈值也不触发剪枝，保留探索机会。

        Returns:
            (is_pruned: bool, reason: str)
        """
        cutoffs = self.cutoffs

        # Vina 截断
        if result.vina_score is not None and result.vina_score > cutoffs['vina_min']:
            return True, f"Vina={result.vina_score:.2f} > {cutoffs['vina_min']} (过弱结合)"

        # DILI 截断（应用 Delta Scoring）
        if result.dili_prob is not None and result.dili_prob > cutoffs['dili_max']:
            if baseline_dili is not None:
                delta = result.dili_prob - baseline_dili
                if delta <= 0.05:
                    logger.debug(f"[Delta DILI] 毒性升高 {delta:.3f} <= 0.05，免于剪枝")
                else:
                    return True, f"DILI={result.dili_prob:.2f} > {cutoffs['dili_max']} (高肝毒性, Δ={delta:+.3f})"
            else:
                return True, f"DILI={result.dili_prob:.2f} > {cutoffs['dili_max']} (高肝毒性)"

        # hERG 截断（应用 Delta Scoring）
        if result.herg_prob is not None and result.herg_prob > cutoffs['herg_max']:
            if baseline_herg is not None:
                delta = result.herg_prob - baseline_herg
                if delta <= 0.05:
                    logger.debug(f"[Delta hERG] 风险升高 {delta:.3f} <= 0.05，免于剪枝")
                else:
                    return True, f"hERG={result.herg_prob:.2f} > {cutoffs['herg_max']} (高心脏风险, Δ={delta:+.3f})"
            else:
                return True, f"hERG={result.herg_prob:.2f} > {cutoffs['herg_max']} (高心脏风险)"

        # MW 截断
        if not (cutoffs['mw_min'] <= result.rd_mw <= cutoffs['mw_max']):
            return True, f"MW={result.rd_mw:.1f} outside [{cutoffs['mw_min']}, {cutoffs['mw_max']}]"

        # TPSA 截断（双侧死亡线）
        if result.rd_tpsa > cutoffs['tpsa_max']:
            return True, f"TPSA={result.rd_tpsa:.1f} > {cutoffs['tpsa_max']} Å² (肠道吸收困难)"
        if result.rd_tpsa < cutoffs['tpsa_min']:
            return True, f"TPSA={result.rd_tpsa:.1f} < {cutoffs['tpsa_min']} Å² (水溶性极差)"

        return False, ""

    def normalize_vina(self, vina_score: Optional[float]) -> float:
        """
        将 Vina 分数归一化到 [0, 1]（越高越好）。

        Vina 典型范围: -12 ~ -4 kcal/mol
        norm = ((-4) - vina) / 8
        即: -12 -> 1.0, -4 -> 0.0
        """
        if vina_score is None:
            return 0.5
        normalized = ( -4.0 - vina_score) / 8.0
        return max(0.0, min(1.0, normalized))

    def normalize_tpsa(self, tpsa: float) -> float:
        """
        TPSA 梯段式惩罚（Stepped Penalty），分为五个区段：

        1. 死亡线 (Hard Pruning): TPSA > 140 Å² → 返回 -1
           分子连肠道吸收都成问题，直接死刑（在 compute_reward 前由 check_hard_cutoffs 处理）

        2. 重度惩罚: 90 < TPSA <= 140 Å²
           无法入脑，逐级扣分至死亡线
           norm = 1.0 - 2.0 * (tpsa - 90) / 50

        3. 理想区间: 40 <= TPSA <= 90 Å²
           不扣分，得满分

        4. 低谷警告: 20 <= TPSA < 40 Å²
           TPSA 过低意味着分子缺乏极性，水溶性差，在体内难以溶解
           norm = 0.8 - 0.8 * (40 - tpsa) / 20

        5. 死亡线 (Hard Pruning): TPSA < 20 Å² → 返回 -1
           由 check_hard_cutoffs 处理
        """
        if tpsa > 90.0:
            # 90 < TPSA <= 140: 逐级扣分至 0
            if tpsa > 140.0:
                return -1.0  # 死亡线
            norm = 1.0 - 2.0 * (tpsa - 90.0) / 50.0
            return max(-1.0, norm)
        if tpsa >= 40.0:
            # 理想区间
            return 1.0
        if tpsa >= 20.0:
            # 低谷警告区：20-40 Å² 缺乏极性，水溶性差，随 TPSA 递增逐渐恢复
            # 20 → 0.5（低谷底部），40 → 1.0（进入理想区），线性过渡
            norm = 0.5 + 0.5 * (tpsa - 20.0) / 20.0
            return norm
        # TPSA < 20: 硬截断，由 check_hard_cutoffs 处理
        return -1.0

    def normalize_mw(self, mw: float) -> float:
        """MW 归一化：最优范围 300-500 Da"""
        optimal_min = 300.0
        optimal_max = 500.0
        if optimal_min <= mw <= optimal_max:
            return 1.0
        if mw < optimal_min:
            return max(0.0, mw / optimal_min)
        if mw > optimal_max:
            return max(0.0, 1.0 - (mw - optimal_max) / 200.0)

    def compute_reward(
        self,
        vina_norm: float,
        dili_norm: float,
        herg_norm: float,
        bbb_norm: float,
        tpsa_norm: float,
        mw_norm: float,
    ) -> float:
        """
        计算 MPO Reward（加权几何平均，参考 REINVENT4 geometric_mean）。

        Reward = exp( sum_i [ w_i * ln(max(s_i, 1e-8)) ] )
               = product_i ( s_i ** w_i )

        因子顺序: vina, dili, herg, bbb, physchem
        """
        # TPSA 死亡线防御（>140 或 <20，由 check_hard_cutoffs 提前剪枝，但保险处理）
        if tpsa_norm < 0:
            return 0.0

        w = self.weights
        # 使用几何平均（product of powers）而非算术平均
        # 低谷区（tpsa < 40）已经由 normalize_tpsa 降权到 0.8，这里自然传导惩罚
        physchem = np.sqrt(max(tpsa_norm, 1e-8) * max(mw_norm, 1e-8))
        total = (
            w['vina']     * np.log(max(vina_norm, 1e-8))
            + w['dili']   * np.log(max(dili_norm, 1e-8))
            + w['herg']   * np.log(max(herg_norm, 1e-8))
            + w['bbb']    * np.log(max(bbb_norm, 1e-8))
            + w['physchem'] * np.log(max(physchem, 1e-8))
        )
        reward = np.exp(total)
        return float(max(0.0, min(1.0, reward)))

    def decide_verdict(self, reward: float, is_pruned: bool) -> EvaluationVerdict:
        """
        根据 Reward 决定 ToT verdict。

        注意：即使 is_pruned=True 也不标记为 IMPOSSIBLE，
        而是将 reward 压低让其自然沉底。目的是保留 Feedback 给 LLM 修正机会，
        而不是直接剪枝导致死胡同。
        - Reward >= 0.7 -> sure
        - Reward >= 0.4 -> likely
        - Reward < 0.4 -> likely（即使很差也保留探索机会）
        """
        if reward >= 0.7:
            return EvaluationVerdict.SURE
        if reward >= 0.4:
            return EvaluationVerdict.LIKELY
        return EvaluationVerdict.LIKELY


# =============================================================================
# Section 5: 外部物理评估器主类
# =============================================================================

class GBMPhysicalEvaluator:
    """
    GBM 外部物理评估器。

    这是唯一对外暴露的接口类。它整合了：
    1. Vina 对接（Docking Tool）
    2. ADMET 计算（DILI + BBB）
    3. RDKit 描述符
    4. MPO Reward 计算（加权几何平均 + 硬截断）
    5. 自然语言反馈生成（供 Prompt 注入使用）

    使用方式（参考 chemcrow/tools/rdkit.py 的 _run() 模式）:

        evaluator = GBMPhysicalEvaluator(
            vina_executable="/path/to/qvina",
            receptor_pdbqt="/path/to/EGFR.pdbqt",
            vina_center=(x, y, z),
            vina_size=(20, 20, 20),
        )

        # 评估单个 SMILES
        result = evaluator.evaluate("CCO")
        print(result.verdict)           # EvaluationVerdict.SURE/LIKELY/IMPOSSIBLE
        print(result.reward)            # 0.85
        print(result.build_feedback_text())  # "Vina得分=-9.2 kcal/mol | ..."

        # 批量评估
        results = evaluator.evaluate_batch(["CCO", "c1ccccc1", ...])
        for r in results:
            print(r.verdict, r.reward)

        # 获取反馈 Prompt（注入给 LLM）
        feedback = evaluator.build_feedback_for_llm(result)
    """

    def __init__(
        self,
        vina_executable: Optional[str] = None,
        receptor_pdbqt: Optional[str] = None,
        vina_center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        vina_size: Tuple[float, float, float] = (20.0, 20.0, 20.0),
        reward_weights: Optional[Dict[str, float]] = None,
        hard_cutoffs: Optional[Dict[str, float]] = None,
        enable_vina: bool = True,
        enable_dili: bool = True,
        enable_bbb: bool = True,
    ):
        """
        Args:
            vina_executable: QVina/Wina 路径，None=伪模式
            receptor_pdbqt: 受体 PDBQT 文件路径
            vina_center: 对接盒子中心
            vina_size: 对接盒子尺寸
            reward_weights: 自定义 Reward 权重
            hard_cutoffs: 自定义硬截断阈值
            enable_vina: 是否启用 Vina 计算（可关闭用于快速测试）
            enable_dili: 是否启用 DILI 计算
            enable_bbb: 是否启用 BBB 计算
        """
        self._vina_tool = None
        self._admet = ADMETCalculator()
        self._mpo = MPORewardCalculator(
            custom_weights=reward_weights,
            custom_cutoffs=hard_cutoffs,
        )
        self._enable_vina = enable_vina and vina_executable is not None
        self._enable_dili = enable_dili
        self._enable_bbb = enable_bbb

        if self._enable_vina:
            self._vina_tool = VinaDockingTool(
                vina_executable=vina_executable,
                receptor_pdbqt=receptor_pdbqt,
                center=vina_center,
                size=vina_size,
            )
            logger.info("[GBMPhysicalEvaluator] Vina enabled (real docking mode)")
        else:
            logger.warning("[GBMPhysicalEvaluator] Vina DISABLED — using pseudo-docking based on MW/LogP")

    def evaluate(
        self,
        smiles: str,
        baseline_dili: Optional[float] = None,
        baseline_herg: Optional[float] = None,
    ) -> PhysicalEvaluationResult:
        """
        评估单个 SMILES，支持 Delta Scoring（相对变化评估）。

        Args:
            smiles: 分子 SMILES 字符串
            baseline_dili: 父节点的原始 DILI 概率（用于 Delta Scoring）
            baseline_herg: 父节点的原始 hERG 概率（用于 Delta Scoring）

        调用顺序（无依赖，可并行化）：
        1. RDKit 描述符（必需，先算）
        2. ADMET（DILI + hERG + BBB）
        3. Vina 对接
        4. MPO Reward
        5. 硬截断检查（应用 Delta Scoring）
        """
        start_time = time.time()
        result = PhysicalEvaluationResult(
            smiles=smiles,
            baseline_dili=baseline_dili,
            baseline_herg=baseline_herg,
        )

        # =========================================================
        # 终极异常防御罩：任何步骤崩溃都不中断程序
        # =========================================================
        try:
            # ---- Step 1: RDKit 描述符 ----
            rd_data = self._compute_rdkit_descriptors(smiles)
            result.rd_tpsa = rd_data['tpsa']
            result.rd_mw = rd_data['mw']
            result.rd_logp = rd_data['logp']
            result.rd_hbd = rd_data['hbd']
            result.rd_hba = rd_data['hba']
            result.rd_n_rotatable = rd_data['n_rotatable']
            result.rd_n_rings = rd_data['n_rings']

            # ---- Step 2: ADMET（DILI + hERG + BBB）----
            if self._enable_dili or self._enable_bbb:
                admet = self._admet.compute(smiles)
                if self._enable_dili:
                    result.dili_prob = admet['dili_prob']
                    result.dili_norm = 1.0 - admet['dili_prob']
                    result.dili_alert_matches = admet['dili_alert_matches']
                # hERG 总是计算（不依赖开关）
                result.herg_prob = admet['herg_prob']
                result.herg_norm = 1.0 - admet['herg_prob']
                result.herg_alert_matches = admet['herg_alert_matches']
                if self._enable_bbb:
                    result.bbb_score = admet['bbb_score']
                    result.bbb_norm  = admet['bbb_score']

            # ---- Step 3: Vina 对接 ----
            if self._enable_vina and self._vina_tool is not None:
                vina_score, vina_error = self._vina_tool.dock(smiles)
                result.vina_score = vina_score
                result.vina_error = vina_error
                if vina_score is not None:
                    result.vina_norm = self._mpo.normalize_vina(vina_score)
                else:
                    result.vina_timeout = (vina_error == "Vina timeout")
                    result.vina_error = vina_error
            else:
                result.vina_score = self._vina_tool._pseudo_dock(smiles) if self._vina_tool else -7.5
                result.vina_norm = self._mpo.normalize_vina(result.vina_score)

            # ---- Step 4: 其他归一化分数 ----
            result.tpsa_norm = self._mpo.normalize_tpsa(result.rd_tpsa)
            result.mw_norm = self._mpo.normalize_mw(result.rd_mw)

            # ---- Step 5: 硬截断检查（应用 Delta Scoring）----
            is_pruned, prune_reason = self._mpo.check_hard_cutoffs(
                result,
                baseline_dili=baseline_dili,
                baseline_herg=baseline_herg,
            )
            result.is_pruned = is_pruned
            result.prune_reason = prune_reason

            # ---- Step 6: MPO Reward ----
            result.reward = self._mpo.compute_reward(
                vina_norm=result.vina_norm,
                dili_norm=result.dili_norm,
                herg_norm=result.herg_norm,
                bbb_norm=result.bbb_norm,
                tpsa_norm=result.tpsa_norm,
                mw_norm=result.mw_norm,
            )

            # ---- Step 7: 决定 verdict ----
            result.verdict = self._mpo.decide_verdict(result.reward, result.is_pruned)

        except Exception as e:
            logger.warning(
                f"[物理引擎跳过] 无法处理该结构: {smiles[:60]}... 原因: {type(e).__name__}: {e}"
            )
            result.vina_score = 0.0
            result.dili_prob = 1.0
            result.dili_norm = 0.0
            result.herg_prob = 1.0
            result.herg_norm = 0.0
            result.bbb_score = 0.0
            result.bbb_norm = 0.0
            result.rd_tpsa = 0.0
            result.rd_mw = 0.0
            result.rd_logp = 0.0
            result.rd_hbd = 0
            result.rd_hba = 0
            result.rd_n_rotatable = 0
            result.rd_n_rings = 0
            result.tpsa_norm = 0.0
            result.mw_norm = 0.0
            result.vina_norm = 0.0
            result.reward = 0.0
            result.is_pruned = False
            result.prune_reason = f"计算异常: {type(e).__name__}: {str(e)[:80]}"
            result.verdict = EvaluationVerdict.LIKELY

        elapsed = time.time() - start_time
        alert_info = ""
        if result.dili_alert_matches:
            alert_info += f" dili_alerts={result.dili_alert_matches}"
        if result.herg_alert_matches:
            alert_info += f" herg_alerts={result.herg_alert_matches}"
        logger.debug(
            f"[PhysicalEval] {smiles[:30]}... -> "
            f"verdict={result.verdict.value}, reward={result.reward:.4f}, "
            f"vina={result.vina_score}, dili={result.dili_prob}, "
            f"herg={result.herg_prob}, bbb={result.bbb_score}{alert_info} ({elapsed:.2f}s)"
        )

        return result

    def evaluate_batch(self, smiles_list: List[str]) -> List[PhysicalEvaluationResult]:
        """
        批量评估多个 SMILES。

        目前为顺序执行。真实部署时可使用多进程并行化。
        """
        results = []
        for smi in smiles_list:
            results.append(self.evaluate(smi))
        return results

    def build_feedback_for_llm(self, result: PhysicalEvaluationResult) -> str:
        """
        构建注入给 LLM 的反馈 Prompt。

        示例修改前:
            请基于上次的打分继续优化：{llm_fake_score}

        示例修改后:
            物理引擎真实测试结果如下：
            Vina得分=-9.2 kcal/mol (强结合模式)。
            肝毒性(DILI)=0.12 (低风险)。
            BBB穿透=0.78 (高CNS渗透)。
            理化性质: MW=452.3, LogP=3.21, TPSA=68.4Å²。
            Reward=0.84。
            请在保持原有结合模式的前提下，调整基团...
        """
        return result.build_feedback_text()

    def generate_toxicity_feedback(self, result: PhysicalEvaluationResult) -> str:
        """
        生成具象化的毒性反馈（供 LLM 在下一轮修改时使用）。

        如果命中了结构警报，返回具体的化学修改建议；
        如果没有命中但 ML 分数偏高，给出方向性建议。

        示例输出:
            "hERG 风险上升！因为新生成的分子匹配到了 '氯苯基团' 结构。
             氯苯是常见的 hERG 警示结构。建议将氯原子替换为极性基团如 -OH、-NH2，
             或移动至分子极性端。"
        """
        lines = []

        # ---- hERG 反馈 ----
        if result.herg_alert_matches:
            # 命中了 SMARTS 警报：给出精确的化学修改建议
            alerts_str = "、".join(result.herg_alert_matches)
            herg_level = "极高" if result.herg_prob > 0.7 else "升高"
            lines.append(
                f"hERG 风险 {herg_level}！因为你生成的分子匹配到了 hERG 警示结构：'{alerts_str}'。"
            )

            # 从元数据中提取修改建议
            for alert in result.herg_alert_matches:
                for key, meta in ADMETCalculator.hERG_ALERT_META.items():
                    if meta.get('name') == alert:
                        lines.append(meta.get('fix', ''))
                        break
                else:
                    # 没有精确匹配，给出通用建议
                    lines.append(
                        "请考虑用极性基团（-OH、-NH2、-COOH、-SO2NH2）替换该警示基团，"
                        "以降低 hERG 阻断风险。"
                    )
        elif result.herg_prob is not None and result.herg_prob > 0.5:
            # 没有命中警报但 ML 分数偏高：可能是假阳性，给出方向性建议
            lines.append(
                f"hERG 风险轻微升高（prob={result.herg_prob:.2f}），"
                "但未检测到已知结构警报。可能是分子整体高脂溶性导致的。"
                "建议适度增加分子的极性表面积（TPSA），或在 Lipinski 5 规则边界处优化。"
            )

        # ---- DILI 反馈 ----
        if result.dili_alert_matches:
            alerts_str = "、".join(result.dili_alert_matches)
            dili_level = "极高" if result.dili_prob > 0.7 else "升高"
            lines.append(
                f"肝毒性(DILI)风险 {dili_level}！因为你生成的分子匹配到了 '{alerts_str}' 结构。"
            )

            for alert in result.dili_alert_matches:
                for key, meta in ADMETCalculator.DILI_ALERT_META.items():
                    if meta.get('name') == alert:
                        lines.append(meta.get('fix', ''))
                        break
                else:
                    lines.append(
                        "该结构具有潜在的反应性代谢物风险。建议使用化学稳定性更好的生物电子等排体替换。"
                    )
        elif result.dili_prob is not None and result.dili_prob > 0.5:
            lines.append(
                f"肝毒性(DILI)风险轻微升高（prob={result.dili_prob:.2f}），"
                "但未检测到已知结构警报。可能是高 MW(>700) 或高 LogP(>5.0) 导致的。"
                "请控制分子量在 500 以内，LogP 在 4.0 以下。"
            )

        # ---- Delta Scoring 反馈 ----
        if result.baseline_dili is not None and result.dili_prob is not None:
            delta = result.dili_prob - result.baseline_dili
            if delta > 0.05:
                lines.append(
                    f"注意：相比父节点，DILI 毒性上升了 {delta:+.3f}。"
                    "本次优化在引入新基团时需特别关注代谢稳定性。"
                )

        if result.baseline_herg is not None and result.herg_prob is not None:
            delta = result.herg_prob - result.baseline_herg
            if delta > 0.05:
                lines.append(
                    f"注意：相比父节点，hERG 风险上升了 {delta:+.3f}。"
                    "新增的取代基可能引入了 hERG 阻断药效团，请优先替换。"
                )

        if not lines:
            lines.append("毒性评估通过，未检测到已知结构警报。")

        return "\n".join(lines)

    def build_batch_feedback(
        self,
        results: List[PhysicalEvaluationResult],
        top_k: int = 3,
    ) -> str:
        """
        为批量评估结果生成汇总反馈。

        用于多候选排序后，将 Top-K 的物理数据注入给 LLM 进行下一轮优化。
        """
        if not results:
            return "无有效候选分子。"

        # 按 Reward 排序
        sorted_results = sorted(results, key=lambda r: r.reward, reverse=True)
        top_results = sorted_results[:top_k]

        lines = ["【物理引擎批量评估结果（Top-K）】"]
        lines.append(f"共评估 {len(results)} 个分子，Top-{len(top_results)} 如下：\n")

        for i, r in enumerate(top_results, 1):
            status = "【已剪枝】" if r.is_pruned else "【通过】"
            lines.append(
                f"{i}. {status} {r.smiles[:50]}...\n"
                f"   {r.build_feedback_text()}"
            )

        # 剪枝统计
        pruned_count = sum(1 for r in results if r.is_pruned)
        if pruned_count > 0:
            lines.append(f"\n注意：{pruned_count}/{len(results)} 个分子因硬截断被剪枝。")

        return "\n".join(lines)

    def _compute_rdkit_descriptors(self, smiles: str) -> Dict[str, Any]:
        """计算 RDKit 分子描述符"""
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                return self._default_rdkit_data()
            try:
                Chem.SanitizeMol(mol, catchErrors=True)
            except Exception:
                pass
            return {
                'tpsa': float(Descriptors.TPSA(mol)),
                'mw': float(Descriptors.MolWt(mol)),
                'logp': float(Crippen.MolLogP(mol)),
                'hbd': int(Descriptors.NumHDonors(mol)),
                'hba': int(Descriptors.NumHAcceptors(mol)),
                'n_rotatable': int(Descriptors.NumRotatableBonds(mol)),
                'n_rings': int(Chem.rdMolDescriptors.CalcNumRings(mol)),
            }
        except Exception:
            return self._default_rdkit_data()

    def _default_rdkit_data(self) -> Dict[str, Any]:
        return {
            'tpsa': 0.0, 'mw': 0.0, 'logp': 0.0,
            'hbd': 0, 'hba': 0, 'n_rotatable': 0, 'n_rings': 0,
        }
