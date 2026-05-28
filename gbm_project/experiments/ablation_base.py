#!/usr/bin/env python3
"""
GBM Ablation 实验共享基类
========================
提供 ToT 消融实验所需的公共基础设施：
- 批量靶点生成分子
- Checkpoint 写入
- 结果保存
- 模型加载封装

所有消融实验脚本继承此类，并实现各自的生成逻辑。
"""

import os
import sys
import json
import torch
import argparse
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入 ToT 生成器（用于模型加载和结果保存）
sys.path.insert(0, str(PROJECT_ROOT / "gbm_project" / "scripts"))
from generate_tot_molecules import (
    TreeOfThoughtsGenerator, ToTNode
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _basic_validate_smiles_impl(smiles: str) -> bool:
    """Standalone SMILES validation (mirrors generate_tot_molecules._basic_validate_smiles)."""
    import re
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    if any(c.isspace() for c in smiles):
        return False
    if not re.match(r"^[A-Za-z0-9@+\-\[\]\(\)=#%/\\.]+$", smiles):
        return False
    invalid_values = [
        "Chain-of-Thought", "valid_smiles", "smiles_string", "<valid_smiles>",
        "analysis", "reasoning", "design", "output", "format", "scaffold",
        "strategy", "warhead", "enhancer", "rationale"
    ]
    if smiles.lower() in [v.lower() for v in invalid_values]:
        return False
    s_lower = smiles.lower()
    if any(sub in s_lower for sub in [
        "synthetic", "complexity", "therefore", "suggests", "moderately",
        "challenging", "retrosynthesis", "synthesize", "procedure",
        "designed molecule", "designed scaffold"
    ]):
        return False
    scaffold_names = [
        "quinazoline", "pyrimidine", "indole", "imidazotetrazine",
        "benzimidazole", "purine", "pyridine", "pyrrole", "thiazole"
    ]
    if smiles.lower().strip() in scaffold_names:
        return False
    if len(smiles) < 5 or len(smiles) > 600:
        return False
    stripped = smiles.strip()
    if re.match(r"^\[[A-Za-z][A-Za-z0-9]*[+-]?\]$", stripped):
        return False
    c_count = stripped.count("C") + stripped.count("c")
    if c_count < 2:
        return False
    open_brackets = sum(1 for c in smiles if c in "([{")
    close_brackets = sum(1 for c in smiles if c in ")]}")
    if open_brackets != close_brackets:
        return False
    if not any(c in smiles for c in "CBNOSPFIH"):
        return False
    invalid_chars = set("!@$%^&*=|\\{}:;\"<>,?`~")
    if any(c in invalid_chars for c in smiles):
        return False

    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return False
        Chem.SanitizeMol(mol, catchErrors=True)
        tpsa = Descriptors.TPSA(mol)
        tpsa_min, tpsa_max = 40.0, 120.0
        if not (tpsa_min <= tpsa <= tpsa_max):
            return False
    except Exception:
        return False
    return True


def _rdkit_validate(smiles: str, tpsa_min: float = 40.0, tpsa_max: float = 120.0,
                    mw_min: float = 100, mw_max: float = 900) -> tuple:
    """Standalone RDKit TPSA/MW validation."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return False, 0.0, 0.0, "MolFromSmiles returned None"
        Chem.SanitizeMol(mol, catchErrors=True)
        tpsa = Descriptors.TPSA(mol)
        mw = Descriptors.MolWt(mol)
        if not (tpsa_min <= tpsa <= tpsa_max):
            return False, tpsa, mw, f"TPSA={tpsa:.1f} outside [{tpsa_min:.0f}-{tpsa_max:.0f}]"
        if not (mw_min <= mw <= mw_max):
            return False, tpsa, mw, f"MW={mw:.1f} outside [{mw_min:.0f}-{mw_max:.0f}]"
        return True, tpsa, mw, ""
    except Exception as e:
        return False, 0.0, 0.0, str(e)


class AblationBase:
    """
    消融实验基类。

    子类必须实现:
        - _create_generator(config): 创建各自的 Generator 实例
        - _run_single_attempt(generator, target_name): 执行一次生成，返回分子列表
        - experiment_name: str, 实验名称（用于输出目录）
    """

    experiment_name: str = "ablation_base"
    ablation_description: str = ""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.generator: Optional[TreeOfThoughtsGenerator] = None

    # ── 子类必须实现 ───────────────────────────────────────────────────────────

    def _create_generator(self) -> TreeOfThoughtsGenerator:
        """Create and return a TreeOfThoughtsGenerator instance."""
        raise NotImplementedError

    def _run_single_attempt(self, generator: TreeOfThoughtsGenerator, target_name: str) -> List[Dict[str, Any]]:
        """
        Execute one generation attempt for target_name.
        Returns a list of molecule dicts (same format as ToT output).
        """
        raise NotImplementedError

    # ── 公共入口 ───────────────────────────────────────────────────────────────

    def run(self):
        """Run the ablation experiment for all configured targets."""
        targets = self._load_targets()
        output_dir = self._setup_output_dir()
        generator = self._create_generator()

        # 加载模型（只加载一次）
        logger.info("Loading models...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        generator.load_models(use_8bit=bool(self.config.get('use_8bit', False)))

        all_results = {}

        for target in targets:
            logger.info("=" * 80)
            logger.info(f"[{self.experiment_name}] Processing target: {target}")
            logger.info("=" * 80)

            try:
                molecules = self._generate_for_target(
                    generator, target,
                    num_molecules=self.config.get('num_molecules', 50),
                    max_attempts=self.config.get('max_attempts', 200),
                    max_no_new_streak=self.config.get('max_no_new_streak', 0),
                    output_dir=output_dir,
                    checkpoint_callback=self._make_checkpoint_callback(all_results, output_dir),
                )
                all_results[target] = molecules
                self._save_target_results(molecules, target, output_dir)
                logger.info(f"✓ Target {target} done: {len(molecules)} molecules")
            except Exception as e:
                logger.error(f"✗ Target {target} failed: {e}")
                traceback.print_exc()
                all_results[target] = []

        # 保存汇总
        self._save_summary(all_results, output_dir, targets)
        self._save_all_results(all_results, output_dir)
        logger.info(f"\n{'=' * 80}")
        logger.info(f"[{self.experiment_name}] Experiment complete!")
        logger.info(f"Output: {output_dir}")
        logger.info(f"Total molecules: {sum(len(v) for v in all_results.values())}")
        logger.info(f"{'=' * 80}")

    # ── 靶点加载 ───────────────────────────────────────────────────────────────

    def _load_targets(self) -> List[str]:
        """Load target list from config (CLI args) or gbm_targets.json."""
        if self.config.get('targets'):
            return self.config['targets']

        targets_file = PROJECT_ROOT / "gbm_project" / "data" / "gbm_targets" / "gbm_targets.json"
        with open(targets_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        result = []
        for td in data['gbm_targets']:
            name = td['name']
            if '/' in name:
                name = name.replace('/', '_')
            if ' ' in name:
                name = name.replace(' ', '_')
            result.append(name)
        return result

    # ── 输出目录 ───────────────────────────────────────────────────────────────

    def _setup_output_dir(self) -> Path:
        """Create and return output directory."""
        if self.config.get('output_dir'):
            out = Path(self.config['output_dir'])
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = PROJECT_ROOT / "gbm_project" / "experiments" / f"{self.experiment_name}_{ts}"
        out.mkdir(parents=True, exist_ok=True)

        # 保存实验配置
        exp_cfg = {
            'experiment_name': self.experiment_name,
            'ablation_description': self.ablation_description,
            'targets': self.config.get('targets', 'all'),
            'num_molecules': self.config.get('num_molecules', 50),
            'gpu_id': self.config.get('gpu_id', 0),
            'timestamp': datetime.now().isoformat(),
        }
        with open(out / "experiment_config.json", 'w', encoding='utf-8') as f:
            json.dump(exp_cfg, f, indent=2, ensure_ascii=False)
        return out

    # ── 核心生成循环 ───────────────────────────────────────────────────────────

    def _generate_for_target(
        self,
        generator: TreeOfThoughtsGenerator,
        target_name: str,
        num_molecules: int,
        max_attempts: int,
        max_no_new_streak: int,
        output_dir: Path,
        checkpoint_callback: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        """Generate num_molecules for one target with checkpoint support."""
        all_molecules = []
        attempts = 0
        no_new_streak = 0

        while len(all_molecules) < num_molecules and attempts < max_attempts:
            try:
                molecules = self._run_single_attempt(generator, target_name)

                # 去重
                existing_smiles = {m['smiles'] for m in all_molecules}
                new_mols = [m for m in molecules if m['smiles'] not in existing_smiles]
                all_molecules.extend(new_mols)
                attempts += 1

                if len(new_mols) == 0:
                    no_new_streak += 1
                else:
                    no_new_streak = 0

                logger.info(
                    f"  Attempt {attempts}: {len(molecules)} generated, "
                    f"{len(new_mols)} new, total {len(all_molecules)}/{num_molecules}"
                )

                if checkpoint_callback:
                    checkpoint_callback(list(all_molecules), target_name, attempts)

                if max_no_new_streak > 0 and no_new_streak >= max_no_new_streak:
                    logger.warning(
                        f"  Early stop: {no_new_streak} consecutive attempts with 0 new SMILES"
                    )
                    break

                if len(all_molecules) >= num_molecules:
                    break

            except Exception as e:
                logger.error(f"  Attempt {attempts + 1} error: {e}")
                attempts += 1
                continue

        return all_molecules[:num_molecules]

    # ── Checkpoint ─────────────────────────────────────────────────────────────

    def _make_checkpoint_callback(self, all_results: dict, output_dir: Path):
        def callback(molecules_so_far, target_name, attempts):
            all_results[target_name] = molecules_so_far
            cp_file = output_dir / "all_results_checkpoint.json"
            with open(cp_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            self._append_csv(molecules_so_far, target_name, output_dir)
        return callback

    def _append_csv(self, molecules: List[Dict], target_name: str, output_dir: Path):
        """Append molecules to per-target CSV (idempotent)."""
        csv_dir = output_dir / "csv_output"
        csv_dir.mkdir(exist_ok=True)
        csv_file = csv_dir / f"{target_name}.csv"

        existing_smiles = set()
        if csv_file.exists():
            with open(csv_file) as f:
                for line in f.readlines()[1:]:
                    if line.strip():
                        parts = line.strip().split(',', 2)
                        if len(parts) >= 2:
                            existing_smiles.add(parts[1])

        lines = []
        for mol in molecules:
            smiles = mol.get('smiles', '')
            if smiles not in existing_smiles:
                tpsa = mol.get('tpsa', 0)
                mw = mol.get('mw', 0)
                lines.append(
                    f"{target_name}_{mol.get('id', 0):05d},{smiles},{tpsa:.2f},{mw:.2f},{target_name}\n"
                )
                existing_smiles.add(smiles)

        if lines:
            with open(csv_file, 'a') as f:
                f.writelines(lines)

    # ── 结果保存 ───────────────────────────────────────────────────────────────

    def _save_target_results(self, molecules: List[Dict], target_name: str, output_dir: Path):
        target_dir = output_dir / target_name
        target_dir.mkdir(exist_ok=True)
        self.generator.save_results(molecules, target_name, target_dir)

    def _save_summary(self, all_results: dict, output_dir: Path, targets: List[str]):
        summary = {
            'experiment_name': self.experiment_name,
            'ablation_description': self.ablation_description,
            'targets_processed': list(all_results.keys()),
            'total_molecules': sum(len(v) for v in all_results.values()),
            'per_target': {
                t: len(all_results.get(t, [])) for t in targets
            },
            'timestamp': datetime.now().isoformat(),
        }
        with open(output_dir / "summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    def _save_all_results(self, all_results: dict, output_dir: Path):
        with open(output_dir / "all_results.json", 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)


def build_base_config(args) -> Dict[str, Any]:
    """Build common base config from CLI args."""
    cfg = {
        'gpu_id': args.gpu_id,
        'base_model_path': str(PROJECT_ROOT / "models" / "Qwen2-7B-Instruct"),
        'llamole_adapter_path': str(PROJECT_ROOT / "saves" / "Llamole-Qwen2-7B-Instruct-Adapter"),
        'gbm_adapter_path': str(PROJECT_ROOT / "saves" / "Llamole-Qwen2-7B-Instruct-Adapter-gbm-with-graph-models"),
        'prompts_config_path': str(PROJECT_ROOT / "gbm_project" / "configs" / "english_gbm_prompts.yaml"),
        'generation': {
            'max_new_tokens': 512,
            'temperature': 0.8,
            'top_p': 0.95,
            'do_sample': True,
        },
        'tot_step_max_new_tokens': {
            'scaffold': 200,
            'assembly': 280,
            'smiles': 450,
            'evaluate': 60,
            'vote': 100,
        },
        'tot_max_llm_calls_per_search': 50,
        'tot_max_search_seconds': 300,
        'tot_depth': 3,
        'tot_k': args.k,
        'tot_b': args.b,
        'tot_refinement_rounds': 1,
        'tot_good_reward_threshold': 0.65,
        'constraints': {
            'mw_range': '300-500',
            'bbb_requirement': 'high',
            'logp_range': '2.0-4.0',
            'tpsa_range': '40-120',
        },
        'num_molecules': args.num_molecules,
        'max_attempts': args.max_attempts,
        'max_no_new_streak': args.max_no_new_streak,
        'output_dir': args.output_dir,
        'targets': args.targets,
    }
    return cfg


def add_common_args(parser: argparse.ArgumentParser):
    """Add common CLI arguments to a parser."""
    parser.add_argument('--targets', type=str, nargs='+', default=None,
                        help='Target list (default: all targets)')
    parser.add_argument('--num_molecules', type=int, default=50,
                        help='Molecules per target (default: 50)')
    parser.add_argument('--k', type=int, default=3,
                        help='ToT k (candidates per level, default: 3)')
    parser.add_argument('--b', type=int, default=2,
                        help='ToT b (branches to keep, default: 2)')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU device ID (default: 0)')
    parser.add_argument('--use_8bit', action='store_true',
                        help='Use 8-bit quantization to save memory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: auto timestamp)')
    parser.add_argument('--max_attempts', type=int, default=200,
                        help='Max attempts per target (default: 200)')
    parser.add_argument('--max_no_new_streak', type=int, default=0,
                        help='Early stop after N attempts with 0 new SMILES (default: 0 = disabled)')
