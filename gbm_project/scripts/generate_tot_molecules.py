#!/usr/bin/env python3
"""
GBM Tree-of-Thoughts (ToT) 分子生成器
=====================================
功能:
1. 使用广度优先搜索 (BFS) 策略进行多步推理
2. 三层设计：Scaffold -> Assembly -> SMILES
3. 每层生成 k 个候选，保留 b 个最优分支
4. 使用评估 prompt 过滤不可行方案
5. 记录完整的思维树路径

使用方式:
python generate_tot_molecules.py --target EGFR --k 3 --b 2
"""

import os
import sys
import json
import torch
import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

# Suppress noisy peft / transformers warnings to keep console output clean
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="peft")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("peft").setLevel(logging.ERROR)

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit import RDLogger

# Silence RDKit console output
RDLogger.DisableLog('rdApp')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 添加gbm_project路径（用于导入src模块）
gbm_project_path = PROJECT_ROOT / "gbm_project"
if str(gbm_project_path) not in sys.path:
    sys.path.insert(0, str(gbm_project_path))


@dataclass
class ToTNode:
    """ToT 树节点"""
    level: int  # 0: scaffold, 1: assembly, 2: smiles
    content: str  # 节点内容（骨架名称、策略描述或SMILES）
    parent: Optional['ToTNode'] = None
    children: List['ToTNode'] = None
    evaluation: Optional[str] = None  # 'sure', 'likely', 'impossible'
    metadata: Dict[str, Any] = None
    # 新埋実物理评估结果（由 GBMPhysicalEvaluator 计算，甸于反馘 Prompt 注入）
    physical_result: Optional[Any] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.metadata is None:
            self.metadata = {}
        if self.physical_result is None:
            self.physical_result = None


class TreeOfThoughtsGenerator:
    """Tree-of-Thoughts 分子生成器（BFS策略）"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        gpu_id = config.get('gpu_id', 0)
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            # 如果设置了CUDA_VISIBLE_DEVICES，可见GPU会被重新映射
            # 例如 CUDA_VISIBLE_DEVICES=3 时，只有物理GPU 3可见，逻辑ID为0
            if gpu_id >= num_gpus:
                logger.warning(f"Requested GPU {gpu_id} not available (visible GPUs: {num_gpus}), using GPU 0")
                actual_gpu_id = 0
            else:
                actual_gpu_id = gpu_id
            torch.cuda.set_device(actual_gpu_id)
            self.device = f"cuda:{actual_gpu_id}"
            logger.info(f"Using device: {self.device} (requested GPU ID: {gpu_id}, visible GPUs: {num_gpus})")
        else:
            self.device = "cpu"
        
        # 路径配置
        self.base_model_path = config['base_model_path']
        self.llamole_adapter_path = config['llamole_adapter_path']
        self.gbm_adapter_path = config['gbm_adapter_path']
        self.prompts_config_path = config.get('prompts_config_path', 
            str(PROJECT_ROOT / "gbm_project" / "configs" / "gbm_prompts.yaml"))
        
        # ToT 参数
        self.depth = config.get('tot_depth', 3)  # T=3
        self.k = config.get('tot_k', 3)  # 每层生成候选数
        self.b = config.get('tot_b', 2)  # BFS保留分支数
        
        # 模型和tokenizer
        self.model = None
        self.tokenizer = None
        
        # Prompt生成器
        self.prompt_generator = None
        self._prompt_target_name: Optional[str] = None
        self.physical_evaluator = None  # 外部物理评估器（在 load_prompt_generator 中初始化）
        
        # 生成配置
        self.generation_config = config.get('generation', {
            'max_new_tokens': 512,
            'temperature': 0.8,
            'top_p': 0.95,
            'do_sample': True
        })

        # ToT 预算与分步生成配置（参考 tree-of-thought-llm-master：评估应尽量短且可批量）
        # - tot_*_max_new_tokens：不同阶段的输出长度需求差异很大
        # - tot_max_llm_calls_per_search：单次 bfs_tot_search 内允许的最大 generate 次数，避免卡死式消耗
        # - tot_max_search_seconds：单次 bfs_tot_search 的墙钟时间预算（软预算：超出后停止继续扩展）
        self.tot_step_max_new_tokens = config.get('tot_step_max_new_tokens', {
            'scaffold': 220,
            'assembly': 260,
            'smiles': 260,
            'evaluate': 40,
            'vote': 80
        })
        self.tot_max_llm_calls_per_search = int(config.get('tot_max_llm_calls_per_search', 30))
        self.tot_max_search_seconds = int(config.get('tot_max_search_seconds', 180))
        self._llm_calls_in_search: int = 0

        # ToT multi-round feedback optimization config
        self.tot_refinement_rounds = int(config.get("tot_refinement_rounds", 1))
        self.tot_good_reward_threshold = float(config.get("tot_good_reward_threshold", 0.65))

        # Guardrail fallback switches (default: False for production)
        self.enable_guardrail_fallback = bool(config.get('enable_guardrail_fallback', False))
        self.enable_default_strategy_fallback = bool(config.get('enable_default_strategy_fallback', False))

        # Constraint conditions
        self.constraints = config.get('constraints', {
            'mw_range': '300-500',
            'bbb_requirement': 'high',
            'logp_range': '2.0-4.0',
            'tpsa_range': '20-120'
        })

        # Guardrail statistics (tracks fallback usage)
        self.guardrail_stats: Dict[str, int] = {
            "vote_parse_incomplete": 0,
            "all_impossible_scaffold_fallback": 0,
            "all_impossible_assembly_fallback": 0,
            "default_strategy_fallback": 0,
            "llm_self_correction_retries": 0,
            "llm_self_correction_success": 0,
        }

    def load_models(self, use_8bit: bool = False):
        """Load models and tokenizer.

        Args:
            use_8bit: Whether to use 8-bit quantization to save memory
        """
        logger.info("Loading models with PeftModel...")

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            from peft import PeftModel

            # Load tokenizer
            logger.info(f"Loading tokenizer from {self.base_model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path,
                trust_remote_code=True,
                padding_side="right"
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Configure quantization if needed
            quantization_config = None
            if use_8bit and self.device != "cpu":
                try:
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                    logger.info("Using 8-bit quantization to save memory")
                except Exception as e:
                    logger.warning(f"Failed to configure 8-bit quantization: {e}. Falling back to FP16.")
                    quantization_config = None

            # Load base model
            logger.info(f"Loading base model from {self.base_model_path} on {self.device}")
            model_kwargs: Dict[str, Any] = {
                "trust_remote_code": True,
            }

            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
                device_id = int(self.device.split(":")[-1]) if ":" in self.device else 0
                # Use explicit device_map dict to avoid dispatch_model's .to() on quantized layers
                model_kwargs["device_map"] = {"": device_id}
                logger.info(f"Using 8-bit quantization on device {device_id}")
            else:
                model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["device_map"] = None
                logger.info(f"Using FP16 on device {self.device}")

            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                **model_kwargs
            )

            if not quantization_config and self.device != "cpu":
                device_id = int(self.device.split(":")[-1]) if ":" in self.device else 0
                torch.cuda.set_device(device_id)
                base_model = base_model.to(self.device)
                logger.info(f"Base model moved to {self.device}")

            # Load Llamole adapter and merge
            if self.llamole_adapter_path and os.path.exists(self.llamole_adapter_path):
                logger.info(f"Loading Llamole adapter from {self.llamole_adapter_path}")
                llamole_model = PeftModel.from_pretrained(base_model, self.llamole_adapter_path)
                if quantization_config and self.device != "cpu":
                    logger.info("8-bit mode: Llamole adapter stays on device via device_map")
                elif self.device != "cpu":
                    llamole_model = llamole_model.to(self.device)
                logger.info("Merging Llamole adapter into base model...")
                base_model = llamole_model.merge_and_unload()
                del llamole_model
                torch.cuda.empty_cache()

            # Load GBM adapter (PeftModel, not merged)
            if self.gbm_adapter_path and os.path.exists(self.gbm_adapter_path):
                logger.info(f"Loading GBM adapter from {self.gbm_adapter_path}")
                self.model = PeftModel.from_pretrained(base_model, self.gbm_adapter_path)
                if quantization_config and self.device != "cpu":
                    logger.info("8-bit mode: GBM adapter stays on device via device_map")
                elif self.device != "cpu":
                    self.model = self.model.to(self.device)
                self._base_merged_model = base_model
                logger.info("Saved base_merged_model for SMILES generation fallback")
            else:
                self.model = base_model
                self._base_merged_model = base_model

            logger.info(f"Model loaded: {self.model.__class__.__name__}")
            logger.info("All models loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    def load_prompt_generator(self, target_name: str, force_reload: bool = False):
        """Load prompt generator and physical evaluator."""
        try:
            if (not force_reload) and self.prompt_generator is not None and self._prompt_target_name == target_name:
                return
            try:
                from gbm_project.src.gbm_knowledge_base import GBMKnowledgeBase
                from gbm_project.src.gbm_prompt_generator import GBMPromptGenerator
                from gbm_project.src.gbm_physical_evaluator import GBMPhysicalEvaluator, PhysicalEvaluationResult
            except ImportError:
                from src.gbm_knowledge_base import GBMKnowledgeBase
                from src.gbm_prompt_generator import GBMPromptGenerator
                from src.gbm_physical_evaluator import GBMPhysicalEvaluator, PhysicalEvaluationResult

            kb = GBMKnowledgeBase(
                targets_path=str(PROJECT_ROOT / "gbm_project" / "data" / "gbm_targets" / "gbm_targets.json"),
                clinical_path=str(PROJECT_ROOT / "gbm_project" / "data" / "gbm_clinical" / "gbm_clinical_data.json"),
                molecules_path=str(PROJECT_ROOT / "gbm_project" / "data" / "gbm_molecules" / "gbm_molecules.json")
            )

            self.prompt_generator = GBMPromptGenerator(
                knowledge_base=kb,
                prompts_config_path=self.prompts_config_path,
                language="english"
            )

            self.physical_evaluator = GBMPhysicalEvaluator(
                vina_executable=None,
                receptor_pdbqt=None,
                enable_vina=True,
                enable_dili=True,
                enable_bbb=True,
            )

            self._prompt_target_name = target_name
            logger.info(f"Prompt generator and physical evaluator loaded for target: {target_name}")

        except Exception as e:
            logger.error(f"Failed to load prompt generator: {e}")
            raise

    def _budget_check(self, search_start_ts: float) -> bool:
        """Check if budget allows continued LLM calls."""
        if self._llm_calls_in_search >= self.tot_max_llm_calls_per_search:
            logger.error(f"ToT budget exceeded: llm_calls={self._llm_calls_in_search} >= max_llm_calls_per_search={self.tot_max_llm_calls_per_search}")
            return False
        elapsed = (datetime.now().timestamp() - search_start_ts)
        if elapsed >= self.tot_max_search_seconds:
            logger.error(f"ToT budget exceeded: elapsed={elapsed:.1f}s >= max_search_seconds={self.tot_max_search_seconds}s")
            return False
        return True

    def generate_with_model(self, prompt: str, *, max_new_tokens: Optional[int] = None) -> str:
        """Generate text using the model with chat template."""
        try:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            formatted_prompt = prompt

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        model_device = next(self.model.parameters()).device if hasattr(self.model, "parameters") else self.device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            self._llm_calls_in_search += 1
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens if max_new_tokens is not None else self.generation_config.get("max_new_tokens", 512),
                temperature=self.generation_config.get("temperature", 0.8),
                top_p=self.generation_config.get("top_p", 0.95),
                do_sample=self.generation_config.get("do_sample", True),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        if len(response) > len(prompt) * 0.8:
            scaffold_markers = ["Scaffold 1:", "Strategy 1:", "SMILES 1:"]
            for marker in scaffold_markers:
                if marker in response:
                    marker_idx = response.find(marker)
                    if marker_idx > 0 and marker_idx < len(response) * 0.3:
                        response = response[marker_idx:].strip()
                        break

        return response.strip()

    # ------------------------------------------------------------------
    # Self-correction retry mechanism for structured LLM outputs
    # ------------------------------------------------------------------
    def _generate_structured_with_retry(
        self,
        prompt: str,
        structured_type: str,
        max_new_tokens: int,
        max_retries: int = 3,
    ) -> str:
        """
        无状态重试 (Stateless Retry) + 前车之鉴 (Lesson-from-Errors) 模式。

        每次重试都构建全新的 messages，不累积对话历史。
        error_warning 变量在循环外部初始化，每次出错时更新其内容，
        在下一轮组装 Prompt 时将其拼接到原始 prompt 前面，起到"前车之鉴"作用。

        成功时立即 break，不浪费 token。
        """
        import re as _re

        error_warning = ""
        _smiles_pat = _re.compile(
            r"\b([CNOSPFI][A-Za-z0-9@+\-\[\]\(\)=#%\/\.]{5,})\b"
        )
        _last_response = ""

        for attempt in range(max_retries):
            is_retry = (attempt > 0)
            temperature = 0.2 if is_retry else self.generation_config.get("temperature", 0.8)
            top_p = 0.85 if is_retry else self.generation_config.get("top_p", 0.95)
            retry_tokens = int(max_new_tokens * 0.6) if is_retry else max_new_tokens

            if is_retry:
                self.guardrail_stats["llm_self_correction_retries"] += 1
                logger.info(
                    f"[Self-Correct] Attempt {attempt + 1}/{max_retries} "
                    f"for {structured_type}: temp={temperature}, tokens={retry_tokens}"
                )

            # ---- 组装当前轮次的 Prompt ----
            # 核心：无状态 —— 每次都是基于原始 prompt 重建
            # 如果有 error_warning（来自上一轮错误），拼在最前面
            _current_prompt = (error_warning + "\n\n" + prompt) if error_warning else prompt

            # 追加格式规则说明（仅在重试轮次追加，让首次生成"轻装上阵"）
            # 格式必须与 YAML 模板完全一致：每字段独立行，以触发 parse_scaffold_proposals 的 regex（571行）
            if is_retry:
                _current_prompt += (
                    "\n\n[STRICT FORMAT — copy this EXACT structure:]\n"
                    "Scaffold 1:\n"
                    "Name: <scaffold_name>\n"
                    "Rationale: <one-sentence_rationale>\n"
                    "Base MW: <number> Da\n"
                    "BBB Potential: <high/medium/low>\n"
                    "\n"
                    "Scaffold 2:\n"
                    "Name: <scaffold_name>\n"
                    "Rationale: <one-sentence_rationale>\n"
                    "Base MW: <number> Da\n"
                    "BBB Potential: <high/medium/low>\n"
                    "\n"
                    "Scaffold 3:\n"
                    "Name: <scaffold_name>\n"
                    "Rationale: <one-sentence_rationale>\n"
                    "Base MW: <number> Da\n"
                    "BBB Potential: <high/medium/low>\n\n"
                    "Output ONLY the scaffold block above, starting with 'Scaffold 1:'. No explanation."
                )

            # ---- Tokenize + LLM 调用 ----
            try:
                formatted = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": _current_prompt}],
                    tokenize=False, add_generation_prompt=True
                )
            except Exception:
                formatted = _current_prompt

            inputs = self.tokenizer(formatted, return_tensors="pt")
            model_device = next(self.model.parameters()).device if hasattr(self.model, "parameters") else self.device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            input_length = inputs["input_ids"].shape[1]

            with torch.no_grad():
                self._llm_calls_in_search += 1
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=retry_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=(not is_retry or temperature > 0.05),
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            # Strip echoed prompt
            for marker in ["Scaffold 1:", "Strategy 1:", "SMILES 1:"]:
                idx = response.find(marker)
                if idx >= 0 and idx < len(response) * 0.5:
                    response = response[idx:].strip()
                    break

            _last_response = response  # 保留最后一次输出，供后续判断用

            # ---- SMILES 化学合法性强制校验（仅对 Level 2 SMILES 输出生效）----
            # 注意：Level 0 (scaffold) 和 Level 1 (assembly) 的响应中常见 "Scaffold"、"Nitrile" 等词，
            # 它们会被 _smiles_pat 错误匹配导致误触发 error_warning，干扰重试。
            # 因此只在 structured_type == "smiles" 时启用此校验。
            if structured_type == "smiles":
                _invalid_smiles = [
                    (_c, _c) for _c in _smiles_pat.findall(response)
                    if Chem.MolFromSmiles(_c) is None
                ]
                if _invalid_smiles:
                    _bad = "; ".join(_c for _c, _ in _invalid_smiles[:3])
                    logger.warning(f"[SMILES 校验] 检测到非法 SMILES: {_bad}...")
                    error_warning = (
                        f"【系统警告】：你上一次生成的 SMILES ({_bad}) "
                        f"存在严重的化学语法错误，RDKit 无法解析。"
                        f"请仔细检查化合价、成环数字和括号闭合规则！"
                    )
                    continue  # 无状态：直接进入下一次重试，prompt = error_warning + original

            # ---- 成功：立即退出 ----
            break

        # 所有重试耗尽仍未成功（此时 error_warning != "" 表示有错误，error_warning == "" 表示格式问题）
        if error_warning or not _last_response:
            logger.warning(f"[Self-Correct] All {max_retries} attempts exhausted for {structured_type}")
            # 尝试用 base_merged_model（Llamole adapter only，无 GBM adapter）做最后一次兜底
            if hasattr(self, "_base_merged_model") and self._base_merged_model is not None:
                logger.warning(f"[Base-merged fallback] Retrying {structured_type} with base_merged model (Llamole-only, no GBM domain adapter)")
                try:
                    saved_model = self.model
                    saved_gen_config = dict(self.generation_config)

                    self.model = self._base_merged_model
                    self.generation_config = {
                        "temperature": 0.15,
                        "top_p": 0.85,
                        "do_sample": False,
                        "max_new_tokens": max_new_tokens,
                    }

                    # 极度简洁的格式约束，用换行示例强制模型对齐
                    minimal_format = (
                        "Output ONLY this format, nothing else:\n\n"
                        "Scaffold 1:\n"
                        "Name: <name>\n"
                        "Rationale: <reason>\n"
                        "Base MW: <N> Da\n"
                        "BBB Potential: <level>\n\n"
                        "Scaffold 2:\n"
                        "Name: <name>\n"
                        "Rationale: <reason>\n"
                        "Base MW: <N> Da\n"
                        "BBB Potential: <level>\n\n"
                        "Scaffold 3:\n"
                        "Name: <name>\n"
                        "Rationale: <reason>\n"
                        "Base MW: <N> Da\n"
                        "BBB Potential: <level>\n"
                    )
                    base_prompt = prompt + "\n\n" + minimal_format
                    inputs_base = self.tokenizer(base_prompt, return_tensors="pt")
                    model_device = next(self.model.parameters()).device if hasattr(self.model, "parameters") else self.device
                    inputs_base = {k: v.to(model_device) for k, v in inputs_base.items()}
                    input_length = inputs_base["input_ids"].shape[1]

                    with torch.no_grad():
                        outputs_base = self.model.generate(
                            **inputs_base,
                            max_new_tokens=max_new_tokens,
                            temperature=0.15,
                            top_p=0.85,
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )

                    response_base = self.tokenizer.decode(
                        outputs_base[0][input_length:], skip_special_tokens=True
                    ).strip()

                    self.model = saved_model
                    self.generation_config = saved_gen_config

                    if response_base:
                        logger.info(f"[Base-merged fallback] Got {len(response_base)} chars response")
                        return response_base
                    logger.warning("[Base-merged fallback] Still empty, returning original")

                except Exception as e:
                    self.model = saved_model
                    self.generation_config = saved_gen_config
                    logger.error(f"[Base-merged fallback] Exception: {e}")

        return _last_response


    def evaluate_states_vote(self, domain_prompt: str, partial_solutions: List[str]) -> List[str]:
        """
        Batch physical evaluation (replaces LLM vote-style self-evaluation).

        Delegates to GBMPhysicalEvaluator for real physical computation.
        """
        if not partial_solutions:
            return []

        uniq: List[str] = []
        seen = set()
        for s in partial_solutions:
            key = s.strip()
            if key and key not in seen:
                seen.add(key)
                uniq.append(key)

        results: List[str] = []

        if self.physical_evaluator is None:
            logger.warning("[PhysEval] physical_evaluator not initialized, falling back to heuristics")
            return ["likely"] * len(uniq)

        for s in uniq:
            if self._looks_like_smiles(s):
                smiles_candidate = self._extract_smiles_from_partial(s)
                if smiles_candidate:
                    eval_result = self.physical_evaluator.evaluate(smiles_candidate)
                    results.append(eval_result.verdict.value)
                    logger.info(
                        f"[PhysEval] SMILES={smiles_candidate[:40]}... -> "
                        f"verdict={eval_result.verdict.value}, reward={eval_result.reward:.4f}, "
                        f"vina={eval_result.vina_score:.2f}, dili={eval_result.dili_prob:.2f}, "
                        f"bbb={eval_result.bbb_score:.2f}"
                    )
                else:
                    rd_ok, rd_tpsa = self._rdkit_tpsa(s)
                    if rd_ok and 20 <= rd_tpsa <= 120:
                        results.append("likely")
                    else:
                        results.append("impossible")
            else:
                verdict = self._evaluate_partial_by_heuristics(s)
                results.append(verdict)

        return results

    def _looks_like_smiles(self, text: str) -> bool:
        """Check if text looks like a SMILES string."""
        text_lower = text.lower()
        non_smiles_keywords = [
            "scaffold", "strategy", "warhead", "rationale",
            "enhancer", "estimated", "expected", "name:",
            "base mw", "bbb potential",
        ]
        if any(kw in text_lower for kw in non_smiles_keywords):
            return False
        return any(c in text for c in "()[]=#@+-/") and len(text) > 10

    def _extract_smiles_from_partial(self, partial: str) -> Optional[str]:
        """Extract SMILES substring from partial solution text."""
        candidates = self._extract_smiles_from_text(partial)
        if candidates:
            return candidates[0]
        pattern = r"([CNOSPFI][A-Za-z0-9@+\-\[\]\(\)=#%/\.]{10,})"
        for match in re.finditer(pattern, partial):
            cand = match.group(1)
            cand = re.sub(r"\s+", "", cand).strip(" ,.;")
            if self._basic_validate_smiles(cand):
                return cand
        return None

    def _evaluate_partial_by_heuristics(self, text: str) -> str:
        """Heuristic evaluation for non-SMILES partial solutions."""
        text_lower = text.lower()
        bbb_high = any(kw in text_lower for kw in ["high", "excellent", "strong"])
        bbb_low = any(kw in text_lower for kw in ["low", "poor", "limited"])

        mw_match = re.search(r"(\d+)\s*(?:da|da\))", text_lower)
        if mw_match:
            mw = int(mw_match.group(1))
            if mw < 200 or mw > 600:
                return "impossible"
            if mw > 500 or mw < 250:
                return "likely"

        if bbb_high and not bbb_low:
            return "sure"
        if bbb_low and not bbb_high:
            return "likely"

        return "likely"

    def parse_scaffold_proposals(self, response: str) -> List[Dict[str, Any]]:
        """Parse scaffold proposals from model response."""
        scaffolds = []

        pattern = r"Scaffold\s+(\d+):\s*Name:\s*([^\n]+)\s*Rationale:\s*([^\n]+)\s*Base\s+MW:\s*(\d+)\s*Da\s*BBB\s+Potential:\s*([^\n]+)"
        matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)

        for match in matches:
            idx, name, rationale, mw, bbb = match
            scaffolds.append({
                "name": name.strip(),
                "rationale": rationale.strip(),
                "base_mw": int(mw),
                "bbb_potential": bbb.strip().lower()
            })

        if not scaffolds:
            lines = response.split("\n")
            current_scaffold = {}
            for line in lines:
                line_lower = line.lower()
                if "name:" in line_lower:
                    name_part = line.split(":", 1)[-1].strip()
                    if name_part and len(name_part) < 50:
                        current_scaffold["name"] = name_part
                elif "rationale:" in line_lower:
                    current_scaffold["rationale"] = line.split(":", 1)[-1].strip()[:200]
                elif "base mw:" in line_lower or "base mw" in line_lower:
                    mw_str = re.search(r"(\d+)", line)
                    if mw_str:
                        try:
                            current_scaffold["base_mw"] = int(mw_str.group(1))
                        except:
                            pass
                elif "bbb potential:" in line_lower or "bbb" in line_lower:
                    bbb_val = re.search(r"(high|medium|low)", line, re.IGNORECASE)
                    if bbb_val:
                        current_scaffold["bbb_potential"] = bbb_val.group(1).lower()
                    elif current_scaffold.get("name") and len(current_scaffold) >= 2:
                        current_scaffold["bbb_potential"] = "medium"

                if current_scaffold.get("name") and len(current_scaffold) >= 2:
                    if "base_mw" not in current_scaffold:
                        current_scaffold["base_mw"] = 150
                    if "bbb_potential" not in current_scaffold:
                        current_scaffold["bbb_potential"] = "medium"
                    if "rationale" not in current_scaffold:
                        current_scaffold["rationale"] = "Suitable for GBM drug design"
                    scaffolds.append(current_scaffold.copy())
                    current_scaffold = {}

        if not scaffolds:
            common_scaffolds = ["quinazoline", "pyrimidine", "indole", "imidazotetrazine",
                              "benzimidazole", "purine", "pyridine", "pyrrole", "thiazole"]
            found_names = []
            response_lower = response.lower()
            for scaffold in common_scaffolds:
                if scaffold in response_lower:
                    found_names.append(scaffold)
                    if len(found_names) >= self.k:
                        break
            for i, name in enumerate(found_names[:self.k], 1):
                scaffolds.append({
                    "name": name,
                    "rationale": f"Suitable scaffold for {name}",
                    "base_mw": 150 + i * 20,
                    "bbb_potential": "high" if i <= 2 else "medium"
                })

        if not scaffolds:
            known_scaffolds = [
                "quinazoline", "pyridine", "pyrimidine", "indole", "imidazotetrazine",
                "benzimidazole", "purine", "pyrrole", "thiazole", "quinoline",
                "isoquinoline", "benzothiazole", "triazine", "morpholine",
            ]
            found = [sc for sc in known_scaffolds if sc in response.lower()]
            for i, name in enumerate(found[:self.k], 1):
                scaffolds.append({
                    "name": name,
                    "rationale": f"Extracted from model output: {name}",
                    "base_mw": 150 + i * 20,
                    "bbb_potential": "high" if i <= 2 else "medium"
                })

        if scaffolds:
            self.guardrail_stats["llm_self_correction_success"] += 1

        return scaffolds[:self.k]

    def parse_assembly_strategies(self, response: str) -> List[Dict[str, Any]]:
        """Parse assembly strategies from model response."""
        strategies = []

        pattern = r"Strategy\s+(\d+):\s*Warhead:\s*([^\n]+)\s*BBB\s+Enhancers:\s*([^\n]+)\s*Estimated\s+Total\s+MW:\s*(\d+)\s*Da\s*Expected\s+LogP:\s*([^\n]+)\s*Expected\s+TPSA:\s*([^\n]+)\s*Rationale:\s*([^\n]+)"
        matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)

        for match in matches:
            idx, warhead, enhancers, mw, logp, tpsa, rationale = match
            strategies.append({
                "warhead": warhead.strip(),
                "bbb_enhancers": enhancers.strip(),
                "estimated_mw": int(mw),
                "expected_logp": logp.strip(),
                "expected_tpsa": tpsa.strip(),
                "rationale": rationale.strip()
            })

        if not strategies:
            lines = response.split("\n")
            current_strategy = {}
            for line in lines:
                if "Warhead:" in line:
                    current_strategy["warhead"] = line.split("Warhead:")[-1].strip()
                elif "BBB Enhancers:" in line:
                    current_strategy["bbb_enhancers"] = line.split("BBB Enhancers:")[-1].strip()
                elif "Estimated Total MW:" in line:
                    mw_str = re.search(r"(\d+)", line)
                    if mw_str:
                        current_strategy["estimated_mw"] = int(mw_str.group(1))
                elif "Expected LogP:" in line:
                    current_strategy["expected_logp"] = line.split("Expected LogP:")[-1].strip()
                elif "Expected TPSA:" in line:
                    current_strategy["expected_tpsa"] = line.split("Expected TPSA:")[-1].strip()
                elif "Rationale:" in line:
                    current_strategy["rationale"] = line.split("Rationale:")[-1].strip()
                    if current_strategy:
                        strategies.append(current_strategy)
                        current_strategy = {}

        if not strategies:
            warheads = ["acrylamide", "cyanoacrylamide", "reversible", "covalent",
                       "nitrogen mustard", "epoxide", "alkyl halide"]
            enhancers = ["fluorine", "methoxy", "methyl", "trifluoromethyl", "chloro", "bromo"]
            mw_vals = re.findall(r"(\d{3})", response)
            mw = int(mw_vals[0]) if mw_vals else 450
            warhead_found = next((w for w in warheads if w in response.lower()), "acrylamide")
            enhancer_found = next((e for e in enhancers if e in response.lower()), "fluorine")
            strategies.append({
                "warhead": warhead_found,
                "bbb_enhancers": enhancer_found,
                "estimated_mw": mw,
                "expected_logp": "2.5-3.5",
                "expected_tpsa": "60-90",
                "rationale": f"Extracted from output: warhead={warhead_found}",
            })

        if strategies:
            self.guardrail_stats["llm_self_correction_success"] += 1

        return strategies[:self.k]

    def parse_smiles(self, response: str) -> List[str]:
        """Parse SMILES strings from model response."""
        smiles_list = []

        pattern1 = r"^\s*SMILES\s+\d+:\s*([^\n]+)"
        matches = re.findall(pattern1, response, re.IGNORECASE | re.MULTILINE)

        if not matches:
            pattern2 = r"^\s*SMILES\s*\d*:?\s*([^\n]+)"
            matches = re.findall(pattern2, response, re.IGNORECASE | re.MULTILINE)

        if not matches:
            pattern3 = r"\b([CNOSPFI][A-Za-z0-9@+\-\[\]\(\)=#%/\.]{10,})\b"
            potential_smiles = re.findall(pattern3, response)
            for ps in potential_smiles:
                ps_clean = ps.strip()
                if self._basic_validate_smiles(ps_clean):
                    matches.append(ps_clean)

        if not matches:
            lines = response.split("\n")
            for line in lines:
                if any(tag in line.lower() for tag in ["smiles", "strategy", "scaffold", "name:", "rationale:", "therefore", "synthes"]):
                    continue
                potential = re.search(r"([CNOSPFI][A-Za-z0-9@+\-\[\]\(\)=#%/\.]{10,})", line)
                if potential:
                    ps_clean = potential.group(1).strip()
                    if self._basic_validate_smiles(ps_clean):
                        matches.append(ps_clean)

        for match in matches:
            raw = match.strip()
            raw = re.sub(r"\s+", "", raw).strip(" ,.;")
            cand_strings: List[str] = [raw]
            if ">>" in raw:
                cand_strings.insert(0, raw.split(">>")[-1])

            for cand in cand_strings:
                cand = cand.strip(" ,.;")
                for frag in cand.split("."):
                    frag = frag.strip(" ,.;")
                    if not frag:
                        continue
                    if self._basic_validate_smiles(frag):
                        smiles_list.append(frag)

        smiles_list = list(dict.fromkeys(smiles_list))
        return smiles_list[:self.k]

    def _rdkit_tpsa(self, smiles: str) -> Tuple[bool, float]:
        """Calculate TPSA using RDKit."""
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                return False, 0.0
            try:
                Chem.SanitizeMol(mol, catchErrors=True)
            except Exception:
                pass
            tpsa = Descriptors.TPSA(mol)
            return True, tpsa
        except Exception:
            return False, 0.0

    def _check_rdkit_smiles(self, smiles: str) -> Tuple[bool, float, float, str]:
        """Comprehensive SMILES validation via RDKit."""
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                return False, 0.0, 0.0, "RDKit failed to parse SMILES (MolFromSmiles returned None)"
        except Exception as e:
            return False, 0.0, 0.0, f"RDKit MolFromSmiles exception: {str(e)}"

        sanitize_ok = True
        try:
            Chem.SanitizeMol(mol, catchErrors=True)
        except Exception as e:
            sanitize_ok = False

        tpsa = 0.0
        if sanitize_ok:
            try:
                tpsa = Descriptors.TPSA(mol)
            except Exception:
                pass

        mw = 0.0
        if sanitize_ok:
            try:
                mw = Descriptors.MolWt(mol)
            except Exception:
                pass

        tpsa_range_str = self.constraints.get("tpsa_range", "20-120")
        try:
            tpsa_min, tpsa_max = map(float, tpsa_range_str.split("-"))
        except ValueError:
            tpsa_min, tpsa_max = 20.0, 120.0

        if tpsa > 0 and not (tpsa_min <= tpsa <= tpsa_max):
            return False, tpsa, mw, f"TPSA={tpsa:.1f} outside target range ({tpsa_min:.0f}-{tpsa_max:.0f}). Add polar groups."

        if mw > 0 and (mw < 100 or mw > 900):
            return False, tpsa, mw, f"MW={mw:.1f} outside target range (100-900 Da)."

        if mw > 0 and sanitize_ok:
            return True, tpsa, mw, ""

    def _correct_smiles_via_llm(self, bad_smiles: str, rdkit_error: str,
                                 scaffold: str, warhead: str,
                                 bbb_enhancers: str) -> Tuple[str, bool]:
        """Attempt to correct invalid SMILES using the LLM."""
        MAX_RETRIES = 2
        current_smiles = bad_smiles

        for attempt in range(MAX_RETRIES):
            correction_prompt = (
                "You are a medicinal chemist. The following SMILES has a chemical validity error:\n"
                f"  BAD SMILES: {current_smiles}\n"
                f"  RDKit Error: {rdkit_error}\n\n"
                "Your task is to generate a CORRECTED SMILES that:\n"
                f"1. Keeps the same scaffold motif: {scaffold}\n"
                f"2. Uses the same warhead type: {warhead}\n"
                f"3. Uses BBB enhancers: {bbb_enhancers}\n"
                "4. Is chemically valid (parseable by RDKit)\n"
                "5. TPSA: 20-120, MW: 150-700 Da\n\n"
                "IMPORTANT: Output ONLY the corrected SMILES string on a single line, nothing else.\n"
            )
            response = self.generate_with_model(
                correction_prompt,
                max_new_tokens=self.tot_step_max_new_tokens.get("smiles", 260)
            )

            candidates = self.parse_smiles(response)
            if not candidates:
                candidates = self._extract_smiles_from_text(response)
            if not candidates:
                lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
                for line in lines:
                    if not any(t in line.lower() for t in ["smiles", "corrected", "error", "fix", "valid"]):
                        cleaned = re.sub(r"\s+", "", line).strip(" ,.;")
                        if cleaned:
                            candidates.append(cleaned)
                        break

            for cand in candidates:
                cand = re.sub(r"\s+", "", cand).strip(" ,.;")
                if not cand or cand in (current_smiles, bad_smiles):
                    continue
                ok, tpsa, mw, err = self._check_rdkit_smiles(cand)
                if ok:
                    logger.info(f"  [Self-Correct] Attempt {attempt+1}: {cand[:50]}... -> OK (TPSA={tpsa:.1f}, MW={mw:.1f})")
                    return cand, True

            if candidates:
                for cand in candidates:
                    cand_clean = re.sub(r"\s+", "", cand).strip(" ,.;")
                    if cand_clean:
                        ok2, tpsa2, mw2, err2 = self._check_rdkit_smiles(cand_clean)
                        if not ok2 and err2:
                            current_smiles = cand_clean
                            rdkit_error = err2
                            break

        logger.warning(f"  [Self-Correct] Failed after {MAX_RETRIES} attempts for: {bad_smiles[:50]}...")
        return "", False

    def _basic_validate_smiles(self, smiles: str) -> bool:
        """Strict SMILES validation filter."""
        if any(c.isspace() for c in smiles):
            return False
        if not re.match(r"^[A-Za-z0-9@+\-\[\]\(\)=#%/\\.]+$", smiles):
            return False
        invalid_values = ["Chain-of-Thought", "valid_smiles", "smiles_string", "<valid_smiles>",
                         "analysis", "reasoning", "design", "output", "format", "scaffold",
                         "strategy", "warhead", "enhancer", "rationale"]
        if smiles.lower() in [v.lower() for v in invalid_values]:
            return False
        s_lower = smiles.lower()
        if any(sub in s_lower for sub in ["synthetic", "complexity", "therefore", "suggests", "moderately", "challenging",
            "retrosynthesis", "synthesize", "procedure", "designed molecule", "designed scaffold"]):
            return False
        scaffold_names = ["quinazoline", "pyrimidine", "indole", "imidazotetrazine",
                         "benzimidazole", "purine", "pyridine", "pyrrole", "thiazole"]
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
        rdkit_ok, tpsa = self._rdkit_tpsa(smiles)
        if not rdkit_ok:
            return False
        tpsa_range_str = self.constraints.get("tpsa_range", "40-120")
        try:
            tpsa_min, tpsa_max = map(float, tpsa_range_str.split("-"))
        except ValueError:
            tpsa_min, tpsa_max = 20.0, 120.0
        if not (tpsa_min <= tpsa <= tpsa_max):
            return False
        return True

    def _is_valid_smiles_format(self, content: str) -> bool:
        """Check if content is valid SMILES format (not scaffold name)."""
        scaffold_names = ["quinazoline", "pyrimidine", "indole", "imidazotetrazine",
                         "benzimidazole", "purine", "pyridine", "pyrrole", "thiazole",
                         "benzothiazole", "quinoline", "isoquinoline", "thiophene"]
        if content.lower().strip() in scaffold_names:
            return False
        if len(content) < 5:
            return False
        stripped = content.strip()
        if re.match(r"^\[[A-Za-z][A-Za-z0-9]*[+-]?\]$", stripped):
            return False
        c_count = stripped.count("C") + stripped.count("c")
        if c_count < 2:
            return False
        has_smiles_chars = any(c in content for c in "()[]=#@+-/")
        has_atoms = any(c in content for c in "CBNOSPFI")
        if len(content) < 10 and content.isalpha() and not has_smiles_chars:
            return False
        return has_atoms

    def _generate_smiles_fallback(
        self,
        finetuned_prompt: str,
        *,
        scaffold: str,
        assembly: str,
        warhead: str,
        bbb_enhancers: str,
        target_mw: int,
        physical_feedback: Optional[str] = None,
        max_new_tokens: int = 450
    ) -> str:
        """
        Two-model SMILES generation with physical feedback injection.

        Strategy 1: Fine-tuned model (GBM adapter)
        Strategy 2: Base merged + few-shot (greedy fallback)
        """
        # Inject feedback if provided
        if physical_feedback:
            feedback_prompt = self.prompt_generator.build_tot_propose_prompt_with_feedback(
                domain_prompt=finetuned_prompt,
                current_state={},
                step_type="smiles",
                physical_feedback=physical_feedback,
            )
        else:
            feedback_prompt = finetuned_prompt

        response1 = self.generate_with_model(feedback_prompt, max_new_tokens=max_new_tokens)
        raw1 = self.parse_smiles(response1)
        if raw1:
            logger.info(f"  [Fine-tuned model SMILES OK] got {len(raw1)} candidates")
            return response1

        logger.warning("  [SMILES fallback] Fine-tuned model returned no parseable SMILES. "
                       "Switching to base_merged + few-shot.")
        if not hasattr(self, "_base_merged_model") or self._base_merged_model is None:
            logger.error("  [SMILES fallback] No base_merged model available!")
            return response1

        fewshot_prompt = (
            "Generate valid SMILES for drug molecules.\n\n"
            "Example 1:\n"
            "SMILES 1: Cc1cc(nc(-c2ccc(Cl)cc2)n1)C(=O)Nc3ccc(F)cc3\n"
            "Example 2:\n"
            "SMILES 1: COc1ccc(cc1)C(=O)Nc2cc(C)nc(-c3ccccn3)c2\n"
            "Example 3:\n"
            "SMILES 1: CC(=O)Nc1ccc(-n2ccnc2)cc1C(=O)Nc3ccccc3\n\n"
            "Now generate 3 new valid SMILES:\n"
            "SMILES 1:"
        )

        try:
            saved_model = self.model
            saved_gen_config = dict(self.generation_config)

            self.model = self._base_merged_model
            self.generation_config = {
                "temperature": 0.5,
                "top_p": 0.92,
                "do_sample": True,
                "max_new_tokens": 512,
            }

            response2 = self.generate_with_model(fewshot_prompt, max_new_tokens=max_new_tokens)

            self.model = saved_model
            self.generation_config = saved_gen_config

            raw2 = self.parse_smiles(response2)
            if raw2:
                logger.info(f"  [Base merged SMILES OK] got {len(raw2)} candidates")
            else:
                fallback_smiles = self._extract_smiles_from_text(response2 + feedback_prompt)
                if fallback_smiles:
                    logger.warning(f"  [SMILES extract] Extracted {len(fallback_smiles)} SMILES from text")
                    return "SMILES 1: " + fallback_smiles[0]
                logger.warning(f"  [Base merged SMILES FAILED] Response:\n{response2[:400]}")
            return response2

        except Exception as e:
            self.model = saved_model
            self.generation_config = saved_gen_config
            logger.error(f"  [SMILES fallback] Exception: {e}")
            return response1

    def _extract_smiles_from_text(self, text: str) -> List[str]:
        """Extract SMILES substrings from natural language text."""
        candidates = []
        pattern = r"([A-Z][A-Za-z0-9@+\-\[\]\(\)=#%/\.]{8,})"
        for match in re.finditer(pattern, text):
            cand = match.group(1).strip()
            if not cand:
                continue
            bad_prefixes = [
                "Cytosine", "Quinazoline", "Pyrimidine", "Indole", "Pyridine",
                "Compound", "Molecule", "Scaffold", "Strategy", "Designed",
                "Therefore", "However", "Moreover", "Furthermore", "Synthesis",
                "Solution", "Mixture", "Product", "Reaction", "Procedure",
                "Step", "Example", "Following", "According", "Resulting",
                "Generat", "Output", "Format", "SMILES", "Name", "Ration"
            ]
            if any(cand.startswith(p) for p in bad_prefixes):
                continue
            atoms = set(c for c in cand if c.isupper() and c in "CBNOSPFIH")
            if len(atoms) < 2:
                continue
            if not any(c in cand for c in "()[]=#@+-/"):
                continue
            bad_suffixes = [" is ", " are ", "ing", "ted", "nal", "lar", "ble", "tion"]
            if any(cand.lower().endswith(s) for s in bad_suffixes):
                continue
            try:
                mol = Chem.MolFromSmiles(cand, sanitize=False)
                if mol is not None:
                    try:
                        Chem.SanitizeMol(mol, catchErrors=True)
                    except Exception:
                        pass
                    try:
                        mw = Descriptors.MolWt(mol)
                        if 100 < mw < 900:
                            candidates.append(cand)
                    except Exception:
                        pass
            except Exception:
                continue
        seen = set()
        result = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                result.append(c)
        return result[:self.k]

    def evaluate_state(self, domain_prompt: str, partial_solution: str) -> str:
        """Evaluate a partial solution state using physical engine."""
        if self.physical_evaluator is None:
            logger.warning("[PhysEval] physical_evaluator not initialized, falling back to heuristics")
            return self._evaluate_partial_by_heuristics(partial_solution)

        if self._looks_like_smiles(partial_solution):
            smiles_candidate = self._extract_smiles_from_partial(partial_solution)
            if smiles_candidate:
                eval_result = self.physical_evaluator.evaluate(smiles_candidate)
                logger.info(f"[PhysEval] Single evaluation: {smiles_candidate[:40]}... -> "
                            f"verdict={eval_result.verdict.value}, reward={eval_result.reward:.4f}")
                return eval_result.verdict.value

        return self._evaluate_partial_by_heuristics(partial_solution)

    def _build_physical_feedback_text(
        self,
        vina_score: Optional[float],
        dili_prob: Optional[float],
        bbb_score: Optional[float],
        reward: float,
        smiles: str,
        is_pruned: bool = False,
        prune_reason: str = ""
    ) -> str:
        """
        Build a natural language physical feedback string from real computed values.

        This is injected into the LLM prompt for the next round of refinement.
        """
        parts = []

        if vina_score is not None:
            if vina_score > -7.0:
                vina_str = f"Vina binding energy = {vina_score:.2f} kcal/mol (WEAK binding, need to strengthen)"
            elif vina_score < -10.0:
                vina_str = f"Vina binding energy = {vina_score:.2f} kcal/mol (STRONG binding)"
            else:
                vina_str = f"Vina binding energy = {vina_score:.2f} kcal/mol (moderate binding)"
            parts.append(vina_str)
        else:
            parts.append("Vina binding energy: N/A")

        if dili_prob is not None:
            if dili_prob > 0.8:
                dili_str = f"Drug-induced liver injury (DILI) probability = {dili_prob:.2f} (HIGH RISK - avoid)"
            elif dili_prob > 0.5:
                dili_str = f"DILI probability = {dili_prob:.2f} (moderate risk)"
            elif dili_prob > 0.3:
                dili_str = f"DILI probability = {dili_prob:.2f} (low risk)"
            else:
                dili_str = f"DILI probability = {dili_prob:.2f} (minimal risk)"
            parts.append(dili_str)
        else:
            parts.append("DILI probability: N/A")

        if bbb_score is not None:
            if bbb_score > 0.7:
                bbb_str = f"BBB permeability = {bbb_score:.2f} (HIGH CNS penetration)"
            elif bbb_score > 0.4:
                bbb_str = f"BBB permeability = {bbb_score:.2f} (moderate CNS penetration)"
            else:
                bbb_str = f"BBB permeability = {bbb_score:.2f} (LOW - insufficient for GBM)"
            parts.append(bbb_str)
        else:
            parts.append("BBB permeability: N/A")

        parts.append(f"Overall Reward = {reward:.4f}")

        if is_pruned:
            parts.append(f"[PRUNED: {prune_reason}]")

        return " | ".join(parts)

    def bfs_tot_search(self, target_name: str) -> List[ToTNode]:
        """
        Execute BFS Tree-of-Thoughts search with physical feedback injection.

        Key improvements:
        1. Physical evaluation feedback is injected into every LLM call
        2. Hard pruning: nodes with is_pruned=True are excluded from next round
        3. Multi-round refinement: each branch gets tot_refinement_rounds attempts
        4. Physical feedback closes the loop: real Vina/DILI/BBB scores guide next generation
        """
        logger.info(f"Starting BFS ToT search for target: {target_name}")
        logger.info(f"Parameters: depth={self.depth}, k={self.k}, b={self.b}, "
                   f"refinement_rounds={self.tot_refinement_rounds}")
        self._llm_calls_in_search = 0
        search_start_ts = datetime.now().timestamp()

        domain_prompt = self.prompt_generator.generate_domain_prompt(target_name, self.constraints)

        # Level 0: Scaffold Proposal
        logger.info("=" * 60)
        logger.info("Level 0: Scaffold Proposal")
        logger.info("=" * 60)

        scaffold_state = {
            "target_name": target_name,
            "mw_range": self.constraints.get("mw_range", "300-500"),
            "bbb_requirement": self.constraints.get("bbb_requirement", "high"),
            "logp_range": self.constraints.get("logp_range", "2.0-4.0")
        }

        scaffold_prompt = self.prompt_generator.build_tot_propose_prompt(
            domain_prompt, scaffold_state, "scaffold"
        )

        if not self._budget_check(search_start_ts):
            return []
        scaffold_response = self._generate_structured_with_retry(
            scaffold_prompt,
            structured_type="GBM drug scaffold proposal",
            max_new_tokens=self.tot_step_max_new_tokens.get("scaffold", 220),
            max_retries=3,
        )
        logger.info(f"Generated scaffold proposals (head):\n{scaffold_response[:800]}...")

        scaffold_proposals = self.parse_scaffold_proposals(scaffold_response)
        logger.info(f"Parsed {len(scaffold_proposals)} scaffold proposals")
        for i, prop in enumerate(scaffold_proposals, 1):
            logger.info(f"  Proposal {i}: {prop}")

        scaffold_nodes = []
        partial_solutions = [
            f"Scaffold: {p['name']}, Base MW: {p.get('base_mw', 150)} Da, "
            f"BBB Potential: {p.get('bbb_potential', 'medium')}"
            for p in scaffold_proposals
        ]
        if not self._budget_check(search_start_ts):
            return []
        scaffold_evals = self.evaluate_states_vote(domain_prompt, partial_solutions)
        for proposal, evaluation in zip(scaffold_proposals, scaffold_evals):
            node = ToTNode(level=0, content=proposal["name"],
                          evaluation=evaluation, metadata=proposal)
            scaffold_nodes.append(node)
            logger.info(f"  Scaffold '{proposal['name']}': {evaluation}")

        valid_scaffolds = [n for n in scaffold_nodes if n.evaluation != "impossible"]
        valid_scaffolds.sort(key=lambda x: {"sure": 2, "likely": 1}.get(x.evaluation, 0), reverse=True)
        selected_scaffolds = valid_scaffolds[:self.b]

        logger.info(f"Selected {len(selected_scaffolds)} scaffolds after evaluation")

        # Minimal scaffold fallback: if model proposal fails, use 3 hardcoded scaffolds
        # as a last resort. This is DIFFERENT from guardrail_fallback — it does NOT
        # force-select all "impossible" nodes; it only substitutes the model proposal
        # when it fails entirely. This keeps the LLM in control while preventing 0-output.
        if not selected_scaffolds:
            logger.warning("All scaffold proposals failed evaluation. Using hardcoded scaffold fallback.")
            self.guardrail_stats["all_impossible_scaffold_fallback"] += 1
            fallback_scaffolds = [
                {"name": "quinazoline", "rationale": "Hardcoded fallback: EGFR-approved scaffold",
                 "base_mw": 130, "bbb_potential": "high"},
                {"name": "pyridine", "rationale": "Hardcoded fallback: CNS-friendly heterocycle",
                 "base_mw": 79, "bbb_potential": "high"},
                {"name": "pyrimidine", "rationale": "Hardcoded fallback: core EGFR hinge binder",
                 "base_mw": 80, "bbb_potential": "high"},
            ]
            for prop in fallback_scaffolds[:self.b]:
                node = ToTNode(level=0, content=prop["name"],
                             evaluation="likely", metadata=prop)
                selected_scaffolds.append(node)
            logger.info(f"Scaffold fallback: using {len(selected_scaffolds)} hardcoded scaffolds")

        # Level 1 + Level 2 (nested for feedback propagation)
        logger.info("=" * 60)
        logger.info("Level 1: Assembly Strategy + Level 2: SMILES Generation (nested with feedback)")
        logger.info("=" * 60)

        final_smiles_nodes = []

        for scaffold_node in selected_scaffolds:
            if not self._budget_check(search_start_ts):
                break
            scaffold_name = scaffold_node.content
            scaffold_mw = scaffold_node.metadata.get("base_mw", 200)
            target_mw = int(self.constraints.get("mw_range", "300-500").split("-")[1])
            remaining_mw = target_mw - scaffold_mw

            assembly_state = {
                "selected_scaffold": scaffold_name,
                "scaffold_mw": scaffold_mw,
                "remaining_mw": remaining_mw,
                "target_mw": target_mw
            }

            assembly_prompt = self.prompt_generator.build_tot_propose_prompt(
                domain_prompt, assembly_state, "assembly"
            )

            if not self._budget_check(search_start_ts):
                break
            assembly_response = self._generate_structured_with_retry(
                assembly_prompt,
                structured_type="molecular assembly strategy",
                max_new_tokens=self.tot_step_max_new_tokens.get("assembly", 260),
                max_retries=3,
            )
            assembly_strategies = self.parse_assembly_strategies(assembly_response)
            logger.info(f"Parsed {len(assembly_strategies)} assembly strategies")

            strategy_partial_solutions = [
                f"Scaffold: {scaffold_name}, Strategy: {s.get('warhead', '')} + {s.get('bbb_enhancers', '')}, "
                f"Estimated MW: {s.get('estimated_mw', 500)} Da"
                for s in assembly_strategies
            ]
            if not self._budget_check(search_start_ts):
                break
            strategy_evals = self.evaluate_states_vote(domain_prompt, strategy_partial_solutions)

            assembly_nodes = []
            for strategy, evaluation in zip(assembly_strategies, strategy_evals):
                node = ToTNode(
                    level=1,
                    content=f"{scaffold_name} + {strategy.get('warhead', 'warhead')}",
                    parent=scaffold_node,
                    evaluation=evaluation,
                    metadata=strategy
                )
                scaffold_node.children.append(node)
                assembly_nodes.append(node)
                logger.info(f"  Strategy '{node.content}': {evaluation}")

            valid_assemblies = [n for n in assembly_nodes if n.evaluation != "impossible"]
            valid_assemblies.sort(key=lambda x: {"sure": 2, "likely": 1}.get(x.evaluation, 0), reverse=True)
            selected_assemblies = valid_assemblies[:self.b]

            # Minimal assembly fallback: if model proposal fails, use hardcoded EGFR-style strategies
            if not selected_assemblies:
                logger.warning("All assembly strategies failed. Using hardcoded strategy fallback.")
                self.guardrail_stats["all_impossible_assembly_fallback"] += 1
                fallback_strategies = [
                    {
                        "warhead": "acrylamide",
                        "bbb_enhancers": "fluorine, methoxy",
                        "estimated_mw": 450,
                        "expected_logp": "2.5-3.5",
                        "expected_tpsa": "60-90",
                        "rationale": "Hardcoded EGFR covalent warhead with BBB-friendly groups"
                    },
                    {
                        "warhead": "cyanoacrylamide",
                        "bbb_enhancers": "fluorine, trifluoromethyl",
                        "estimated_mw": 480,
                        "expected_logp": "2.5-4.0",
                        "expected_tpsa": "70-100",
                        "rationale": "Alternative electrophile with lipophilic BBB enhancers"
                    },
                ]
                for i, strat in enumerate(fallback_strategies[:self.b]):
                    node = ToTNode(
                        level=1,
                        content=f"{scaffold_name} + {strat['warhead']}",
                        parent=scaffold_node,
                        evaluation="likely",
                        metadata=strat
                    )
                    scaffold_node.children.append(node)
                    selected_assemblies.append(node)
                logger.info(f"Strategy fallback: using {len(selected_assemblies)} hardcoded strategies")

            # Level 2: SMILES Generation with physical feedback
            logger.info("Level 2: SMILES Generation with physical evaluation and feedback injection")

            for assembly_node in selected_assemblies:
                if not self._budget_check(search_start_ts):
                    break

                strategy_metadata = assembly_node.metadata
                smiles_state = {
                    "selected_scaffold": scaffold_name,
                    "assembly_strategy": assembly_node.content,
                    "warhead_type": strategy_metadata.get("warhead", "acrylamide"),
                    "bbb_enhancers": strategy_metadata.get("bbb_enhancers", "fluorine"),
                    "target_mw": strategy_metadata.get("estimated_mw", 500),
                    "target_logp_range": strategy_metadata.get("expected_logp", "2.0-4.0"),
                    "target_tpsa_range": strategy_metadata.get("expected_tpsa", "20-120"),
                }

                smiles_prompt = self.prompt_generator.build_tot_propose_prompt(
                    domain_prompt, smiles_state, "smiles"
                )

                # Multi-round refinement with physical feedback
                best_nodes_this_branch: List[ToTNode] = []
                best_reward_this_branch = -1.0

                for round_idx in range(max(1, self.tot_refinement_rounds)):
                    round_label = f"[Round {round_idx + 1}] " if self.tot_refinement_rounds > 1 else ""

                    # Build feedback from previous round's best result
                    physical_feedback_str: Optional[str] = None
                    if round_idx > 0 and best_nodes_this_branch:
                        best_node = max(
                            best_nodes_this_branch,
                            key=lambda n: n.physical_result.reward if n.physical_result else 0.0
                        )
                        phys = best_node.physical_result
                        if phys:
                            physical_feedback_str = self._build_physical_feedback_text(
                                vina_score=phys.vina_score,
                                dili_prob=phys.dili_prob,
                                bbb_score=phys.bbb_score,
                                reward=phys.reward,
                                smiles=best_node.content,
                                is_pruned=phys.is_pruned,
                                prune_reason=phys.prune_reason,
                            )
                            logger.info(f"{round_label}Feedback from previous round: {physical_feedback_str[:200]}")

                    if not self._budget_check(search_start_ts):
                        break

                    # Generate SMILES with physical feedback injection
                    smiles_response = self._generate_smiles_fallback(
                        smiles_prompt,
                        scaffold=smiles_state["selected_scaffold"],
                        assembly=smiles_state["assembly_strategy"],
                        warhead=smiles_state["warhead_type"],
                        bbb_enhancers=smiles_state["bbb_enhancers"],
                        target_mw=smiles_state["target_mw"],
                        physical_feedback=physical_feedback_str,
                        max_new_tokens=self.tot_step_max_new_tokens.get("smiles", 450)
                    )
                    logger.info(f"{round_label}Generated SMILES for '{assembly_node.content}':\n{smiles_response[:500]}...")

                    raw_smiles_list = self.parse_smiles(smiles_response)
                    logger.info(f"{round_label}Parsed {len(raw_smiles_list)} raw SMILES candidates")

                    if not raw_smiles_list:
                        logger.warning(f"  {round_label}[Last resort] parse_smiles found nothing. Trying text extraction...")
                        raw_smiles_list = self._extract_smiles_from_text(smiles_response)
                        logger.info(f"  {round_label}[Last resort] Text extraction found {len(raw_smiles_list)} candidates")

                    validated_smiles = []
                    for raw_smiles in raw_smiles_list:
                        cleaned = re.sub(r"\s+", "", raw_smiles).strip(" ,.;")
                        if not cleaned:
                            continue

                        ok, tpsa, mw, err = self._check_rdkit_smiles(cleaned)
                        if ok:
                            validated_smiles.append(cleaned)
                            logger.info(f"  {round_label}[Agentic OK] {cleaned[:60]}... (TPSA={tpsa:.1f}, MW={mw:.1f})")
                            continue

                        logger.warning(f"  {round_label}[Agentic Fix] '{cleaned[:60]}...' -> RDKit error: {err[:120]}")
                        corrected, success = self._correct_smiles_via_llm(
                            bad_smiles=cleaned,
                            rdkit_error=err,
                            scaffold=smiles_state.get("selected_scaffold", ""),
                            warhead=smiles_state.get("warhead_type", ""),
                            bbb_enhancers=smiles_state.get("bbb_enhancers", ""),
                        )
                        if success and corrected:
                            validated_smiles.append(corrected)
                            logger.info(f"  {round_label}[Agentic Corrected] -> {corrected[:60]}...")
                        else:
                            logger.warning(f"  {round_label}[Agentic FAILED] Could not correct: {cleaned[:60]}...")

                    if not validated_smiles:
                        logger.warning(f"{round_label}No valid SMILES after agentic correction for {assembly_node.content}")
                        continue

                    # Evaluate validated SMILES with physical engine and HARD PRUNE
                    for smiles in validated_smiles:
                        phys_result = self.physical_evaluator.evaluate(smiles)

                        # HARD PRUNING: immediately discard dead nodes
                        if phys_result.is_pruned:
                            logger.info(
                                f"  {round_label}[HARD PRUNE] {smiles[:50]}... -> "
                                f"PRUNED: {phys_result.prune_reason} "
                                f"(vina={phys_result.vina_score}, dili={phys_result.dili_prob:.2f})"
                            )
                            continue  # Do NOT add to queue, do NOT continue to next round

                        node = ToTNode(
                            level=2,
                            content=smiles,
                            parent=assembly_node,
                            evaluation=phys_result.verdict.value,
                            metadata={
                                "smiles": smiles,
                                "vina_score": phys_result.vina_score,
                                "dili_prob": phys_result.dili_prob,
                                "bbb_score": phys_result.bbb_score,
                                "reward": phys_result.reward,
                                "is_pruned": phys_result.is_pruned,
                                "refinement_round": round_idx,
                            },
                            physical_result=phys_result,
                        )
                        assembly_node.children.append(node)
                        best_nodes_this_branch.append(node)
                        final_smiles_nodes.append(node)

                        logger.info(
                            f"  {round_label}Final SMILES: {smiles[:50]}... -> "
                            f"verdict={phys_result.verdict.value}, reward={phys_result.reward:.4f}, "
                            f"vina={phys_result.vina_score:.2f}, dili={phys_result.dili_prob:.2f}, "
                            f"bbb={phys_result.bbb_score:.2f}"
                        )

                        if phys_result.reward > best_reward_this_branch:
                            best_reward_this_branch = phys_result.reward

                    # Early stopping for this branch if we already have a good molecule
                    if best_reward_this_branch >= self.tot_good_reward_threshold:
                        logger.info(f"{round_label}Early stop: reward {best_reward_this_branch:.4f} >= threshold {self.tot_good_reward_threshold:.2f}")
                        break

        logger.info("=" * 60)
        logger.info(f"BFS ToT search completed. Generated {len(final_smiles_nodes)} final SMILES")
        logger.info("=" * 60)

        if not final_smiles_nodes:
            logger.error("No valid SMILES generated in Level 2.")
            return []

        return final_smiles_nodes

    def generate_molecules(self, target_name: str) -> List[Dict[str, Any]]:
        """Main entry point for molecule generation."""
        self.load_prompt_generator(target_name)
        final_nodes = self.bfs_tot_search(target_name)

        if not final_nodes:
            logger.warning("No nodes returned from ToT search")
            return []

        non_smiles_nodes = [n for n in final_nodes if n.level != 2]
        if non_smiles_nodes:
            logger.error("ToT search returned non-SMILES nodes (level != 2)")
            return []

        molecules = []
        for node in final_nodes:
            smiles = node.content
            ok, rd_tpsa, rd_mw, err = self._check_rdkit_smiles(smiles)
            if not ok:
                logger.warning(f"[Final filter] Skipping: {smiles[:50]}... ({err[:80]})")
                continue
            try:
                rd_logp = round(Descriptors.MolLogP(Chem.MolFromSmiles(smiles)), 2)
            except Exception:
                rd_logp = None

            path = []
            current = node
            while current:
                path.insert(0, {
                    "level": current.level,
                    "content": current.content,
                    "evaluation": current.evaluation,
                    "metadata": current.metadata
                })
                current = current.parent

            phys_result = node.physical_result
            molecules.append({
                "smiles": smiles,
                "tpsa": round(rd_tpsa, 2),
                "mw": round(rd_mw, 2),
                "logp": rd_logp,
                "tot_path": path,
                "target": target_name,
                "generation_method": "tot_bfs_feedback",
                "physical_evaluation": {
                    "vina_score": phys_result.vina_score if phys_result else None,
                    "dili_prob": phys_result.dili_prob if phys_result else None,
                    "bbb_score": phys_result.bbb_score if phys_result else None,
                    "reward": phys_result.reward if phys_result else 0.0,
                    "verdict": phys_result.verdict.value if phys_result else "unknown",
                    "is_pruned": phys_result.is_pruned if phys_result else False,
                },
                "physical_feedback": phys_result.build_feedback_text() if phys_result else "",
            })

        return molecules

    def save_results(self, molecules: List[Dict[str, Any]], target_name: str, output_dir: Path):
        """Save generation results to JSON."""
        output_data = {
            "config": {
                "target": target_name,
                "tot_depth": self.depth,
                "tot_k": self.k,
                "tot_b": self.b,
                "constraints": self.constraints
            },
            "num_generated": len(molecules),
            "molecules": molecules,
            "timestamp": datetime.now().isoformat()
        }

        output_file = output_dir / "tot_generation_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Results saved to {output_file}")
        return output_file


def main():
    parser = argparse.ArgumentParser(description="Generate GBM molecules using Tree-of-Thoughts (BFS)")
    parser.add_argument("--target", type=str, default="EGFR", help="Target name")
    parser.add_argument("--k", type=int, default=3, help="Number of candidates per level")
    parser.add_argument("--b", type=int, default=2, help="Number of branches to keep in BFS")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--refinement_rounds", type=int, default=1,
                        help="Number of refinement rounds per branch")
    parser.add_argument("--good_reward_threshold", type=float, default=0.65,
                        help="Reward threshold for early stopping")

    args = parser.parse_args()

    config = {
        "gpu_id": args.gpu_id,
        "base_model_path": str(PROJECT_ROOT / "models" / "Qwen2-7B-Instruct"),
        "llamole_adapter_path": str(PROJECT_ROOT / "saves" / "Llamole-Qwen2-7B-Instruct-Adapter"),
        "gbm_adapter_path": str(PROJECT_ROOT / "saves" / "Llamole-Qwen2-7B-Instruct-Adapter-gbm-with-graph-models"),
        "prompts_config_path": str(PROJECT_ROOT / "gbm_project" / "configs" / "gbm_prompts.yaml"),
        "generation": {
            "max_new_tokens": 512,
            "temperature": 0.8,
            "top_p": 0.95,
            "do_sample": True
        },
        "tot_depth": 3,
        "tot_k": args.k,
        "tot_b": args.b,
        "tot_refinement_rounds": args.refinement_rounds,
        "tot_good_reward_threshold": args.good_reward_threshold,
        "constraints": {
            "mw_range": "300-500",
            "bbb_requirement": "high",
            "logp_range": "2.0-4.0"
        }
    }

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "gbm_project" / "experiments" / f"tot_bfs_{args.target}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    output_dir.mkdir(parents=True, exist_ok=True)

    generator = TreeOfThoughtsGenerator(config)
    generator.load_models()
    molecules = generator.generate_molecules(args.target)
    generator.save_results(molecules, args.target, output_dir)

    print("\n" + "=" * 60)
    print(f"ToT BFS Generation Complete for {args.target}")
    print("=" * 60)
    print(f"Target: {args.target}")
    print(f"Molecules generated: {len(molecules)}")
    print(f"ToT parameters: k={args.k}, b={args.b}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    print("Generated SMILES:")
    for i, mol in enumerate(molecules, 1):
        print(f"  {i}. {mol['smiles']}")
        print(f"     Path: {' -> '.join([p['content'][:30] for p in mol['tot_path']])}")
    print("=" * 60)

    return molecules


if __name__ == "__main__":
    main()
