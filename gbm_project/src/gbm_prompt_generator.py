"""
GBM Prompt Generator
基于GBM知识库生成专业prompt，支持CoT推理和约束条件
"""

import yaml
import random
from typing import Dict, List, Any, Optional
from .gbm_knowledge_base import GBMKnowledgeBase
from .structural_knowledge_injector import StructuralKnowledgeInjector


class GBMPromptGenerator:
    """GBM专业prompt生成器"""

    def __init__(self, knowledge_base: GBMKnowledgeBase, prompts_config_path: str, language: str = "english"):
        self.kb = knowledge_base
        self.language = language

        # 加载prompt模板
        with open(prompts_config_path, 'r', encoding='utf-8') as f:
            self.prompt_templates = yaml.safe_load(f)

        # 如果使用英文，加载英文配置
        if language == "english":
            english_config_path = prompts_config_path.replace("gbm_prompts.yaml", "english_gbm_prompts.yaml")
            try:
                with open(english_config_path, 'r', encoding='utf-8') as f:
                    self.prompt_templates = yaml.safe_load(f)
            except FileNotFoundError:
                print(f"Warning: English config not found at {english_config_path}, using Chinese version")

            # 显式检查 ToT 模板是否存在，避免静默退化为非结构化 prompt
            tot_templates = self.prompt_templates.get("tot_design_nodes", {})
            if not tot_templates:
                print(
                    "[GBMPromptGenerator] Warning: 'tot_design_nodes' is missing or empty in the English "
                    "prompt config. Tree-of-Thought generation will fall back to generic prompts and "
                    "may fail to parse scaffold proposals. Please ensure 'tot_design_nodes' is defined "
                    "in 'english_gbm_prompts.yaml'."
                )

        # 初始化结构知识注入器
        self.structural_injector = StructuralKnowledgeInjector(language=language)

    def generate_domain_prompt(self, target_name: str, constraints: Optional[Dict[str, Any]] = None) -> str:
        """生成GBM领域专业prompt"""
        target = self.kb.get_target_info(target_name)
        if not target:
            return self._generate_generic_gbm_prompt(constraints)

        # 获取靶点特定prompt
        target_prompt = self.prompt_templates['target_specific_prompts'].get(target_name, "")
        if not target_prompt:
            target_prompt = self._generate_target_specific_prompt(target)

        # 组合基础prompt
        base_prompt = self.prompt_templates['base_gbm_prompt']

        # 添加具体要求
        specific_requirements = self._format_target_requirements(target, constraints)

        full_prompt = base_prompt.format(specific_requirements=specific_requirements)
        full_prompt += "\n\n" + target_prompt

        # 添加临床洞察
        clinical_insights = self._add_clinical_insights()
        full_prompt += "\n\n" + clinical_insights

        # 添加约束条件
        if constraints:
            constraint_text = self._format_constraints(constraints)
            full_prompt += "\n\n" + constraint_text

        # 注入结构知识
        full_prompt = self.structural_injector.inject_structural_context(full_prompt, target_name, constraints)

        return full_prompt

    def generate_cot_prompt(self, domain_prompt: str, target_name: str, reasoning_type: str = "step_by_step_design") -> str:
        """生成Chain-of-Thought推理prompt"""
        cot_template = self.prompt_templates['cot_reasoning_templates'].get(reasoning_type, "")

        if not cot_template:
            cot_template = self.prompt_templates['cot_reasoning_templates']['step_by_step_design']

        # 添加靶点特定推理
        target_specific_reasoning = self._generate_target_reasoning(target_name)
        cot_template += "\n\n" + target_specific_reasoning

        # 组合完整CoT prompt
        full_cot_prompt = domain_prompt + "\n\n" + cot_template

        return full_cot_prompt

    def generate_evaluation_prompt(self, molecule_description: str, evaluation_type: str) -> str:
        """生成分子评估prompt"""
        eval_template = self.prompt_templates['evaluation_prompts'].get(evaluation_type, "")

        if not eval_template:
            eval_template = f"请评估以下分子的{evaluation_type}：\n{molecule_description}"

        return eval_template

    def _generate_generic_gbm_prompt(self, constraints: Optional[Dict[str, Any]]) -> str:
        """生成通用GBM prompt"""
        base_prompt = self.prompt_templates['base_gbm_prompt']

        if self.language == "english":
            requirements = "General GBM therapeutic molecule design requirements:\n"
            requirements += "- Good blood-brain barrier penetration\n"
            requirements += "- Selective toxicity to GBM cells\n"
            requirements += "- Overcome multidrug resistance mechanisms\n"
            requirements += "- High clinical translation potential"
        else:
            requirements = "通用GBM治疗分子设计要求：\n"
            requirements += "- 血脑屏障穿透性好\n"
            requirements += "- 对GBM细胞有选择性毒性\n"
            requirements += "- 克服多药耐药机制\n"
            requirements += "- 临床转化潜力高"

        if constraints:
            if self.language == "english":
                requirements += "\n\nConstraints:\n" + self._format_constraints(constraints)
            else:
                requirements += "\n\n约束条件：\n" + self._format_constraints(constraints)

        return base_prompt.format(specific_requirements=requirements)

    def _generate_target_specific_prompt(self, target: Any) -> str:
        """生成靶点特定prompt"""
        if self.language == "english":
            prompt = f"Design GBM candidates targeting the {target.name} pathway:\n\n"
            prompt += f"Mechanism of action:\n{target.description}\n\n"
            prompt += f"Mutation types:\n" + "\n".join(f"- {mut}" for mut in target.mutation_types) + "\n\n"
            prompt += f"Current drugs:\n" + "\n".join(f"- {drug}" for drug in target.current_drugs) + "\n\n"
            prompt += f"Challenges:\n" + "\n".join(f"- {challenge}" for challenge in target.challenges) + "\n\n"

            if target.structural_requirements:
                prompt += "Molecular design requirements:\n"
                for key, value in target.structural_requirements.items():
                    prompt += f"- {key}: {value}\n"
        else:
            prompt = f"针对{target.name}通路设计GBM候选药物：\n\n"
            prompt += f"作用机制：\n{target.description}\n\n"
            prompt += f"突变类型：\n" + "\n".join(f"- {mut}" for mut in target.mutation_types) + "\n\n"
            prompt += f"现有药物：\n" + "\n".join(f"- {drug}" for drug in target.current_drugs) + "\n\n"
            prompt += f"挑战：\n" + "\n".join(f"- {challenge}" for challenge in target.challenges) + "\n\n"

            if target.structural_requirements:
                prompt += "分子设计要求：\n"
                for key, value in target.structural_requirements.items():
                    prompt += f"- {key}: {value}\n"

        return prompt

    def _format_target_requirements(self, target: Any, constraints: Optional[Dict[str, Any]]) -> str:
        """格式化靶点具体要求"""
        if self.language == "english":
            requirements = f"Design candidate drugs for {target.name} target:\n\n"

            # 添加靶点信息
            requirements += f"Target description: {target.description}\n\n"

            # 添加设计要求
            requirements += "Design requirements:\n"
            for challenge in target.challenges:
                requirements += f"- Address challenge: {challenge}\n"

            # 添加结构要求
            if target.structural_requirements:
                requirements += "\nStructural requirements:\n"
                for key, value in target.structural_requirements.items():
                    requirements += f"- {key}: {value}\n"

            # 添加约束条件
            if constraints:
                requirements += "\n\nMolecular constraints:\n"
                for key, value in constraints.items():
                    requirements += f"- {key}: {value}\n"
        else:
            requirements = f"为{target.name}靶点设计候选药物：\n\n"

            # 添加靶点信息
            requirements += f"靶点描述：{target.description}\n\n"

            # 添加设计要求
            requirements += "设计要求：\n"
            for challenge in target.challenges:
                requirements += f"- 解决挑战：{challenge}\n"

            # 添加结构要求
            if target.structural_requirements:
                requirements += "\n结构要求：\n"
                for key, value in target.structural_requirements.items():
                    requirements += f"- {key}: {value}\n"

            # 添加约束条件
            if constraints:
                requirements += "\n\n分子约束：\n"
                for key, value in constraints.items():
                    requirements += f"- {key}: {value}\n"

        return requirements

    def _add_clinical_insights(self) -> str:
        """添加临床洞察"""
        insights = self.kb.get_clinical_insights()

        if self.language == "english":
            clinical_text = "Clinical treatment insights:\n\n"
            clinical_text += "GBM treatment challenges:\n"
            for challenge in insights['challenges'][:3]:  # 限制数量
                clinical_text += f"- {challenge}\n"

            clinical_text += "\nSuccessful treatment patterns:\n"
            for pattern in insights['successful_patterns'][:3]:
                clinical_text += f"- {pattern}\n"

            clinical_text += "\nClinical trial lessons:\n"
            for insight in insights['failed_insights'][:2]:
                clinical_text += f"- {insight}\n"
        else:
            clinical_text = "临床治疗洞察：\n\n"
            clinical_text += "GBM治疗挑战：\n"
            for challenge in insights['challenges'][:3]:  # 限制数量
                clinical_text += f"- {challenge}\n"

            clinical_text += "\n成功治疗模式：\n"
            for pattern in insights['successful_patterns'][:3]:
                clinical_text += f"- {pattern}\n"

            clinical_text += "\n临床试验教训：\n"
            for insight in insights['failed_insights'][:2]:
                clinical_text += f"- {insight}\n"

        return clinical_text

    def _format_constraints(self, constraints: Dict[str, Any]) -> str:
        """格式化约束条件"""
        constraint_templates = self.prompt_templates.get('constraint_templates', {})

        if self.language == "english":
            formatted_constraints = "Molecular constraints:\n"
        else:
            formatted_constraints = "分子约束条件：\n"

        for constraint_type, constraint_value in constraints.items():
            if constraint_type in constraint_templates:
                template = constraint_templates[constraint_type]
                formatted_constraints += template.format(**constraint_value)
            else:
                formatted_constraints += f"- {constraint_type}: {constraint_value}\n"

        return formatted_constraints

    def _generate_target_reasoning(self, target_name: str) -> str:
        """生成靶点特定推理步骤"""
        target = self.kb.get_target_info(target_name)
        if not target:
            return ""

        if self.language == "english":
            reasoning = f"Reasoning process for designing molecules targeting {target_name}:\n\n"

            # 获取相似分子作为参考
            similar_molecules = self.kb.get_similar_molecules(target_name, 3)
            if similar_molecules:
                reasoning += "Reference existing molecules:\n"
                for mol in similar_molecules:
                    reasoning += f"- {mol.name} ({mol.status}): {mol.mechanism}\n"
                reasoning += "\n"

            # 添加靶点特定推理
            reasoning += f"Special considerations for {target_name} target:\n"
            reasoning += f"- Overcome resistance mechanisms: {', '.join(target.challenges)}\n"
            reasoning += f"- Optimize clinical safety: minimize toxicity to normal cells\n"
            reasoning += f"- Improve therapeutic efficacy: subtype-based selectivity\n"
        else:
            reasoning = f"为{target_name}靶点设计分子的推理过程：\n\n"

            # 获取相似分子作为参考
            similar_molecules = self.kb.get_similar_molecules(target_name, 3)
            if similar_molecules:
                reasoning += "参考现有分子：\n"
                for mol in similar_molecules:
                    reasoning += f"- {mol.name} ({mol.status}): {mol.mechanism}\n"
                reasoning += "\n"

            # 添加靶点特定推理
            reasoning += f"靶点{target_name}的特殊考虑：\n"
            reasoning += f"- 克服耐药机制：{'、'.join(target.challenges)}\n"
            reasoning += f"- 优化临床安全性：最小化正常细胞毒性\n"
            reasoning += f"- 提高治疗效果：基于分子亚型选择性\n"

        return reasoning

    def generate_full_prompt(self, target_name: str = None, constraints: Optional[Dict[str, Any]] = None,
                           use_cot: bool = True, reasoning_type: str = "step_by_step_design") -> str:
        """生成完整的GBM生成prompt"""
        # 如果没有指定靶点，随机选择
        if target_name is None:
            target = self.kb.get_random_target()
            target_name = target.name

        # 生成领域prompt
        domain_prompt = self.generate_domain_prompt(target_name, constraints)

        # 添加CoT推理
        if use_cot:
            full_prompt = self.generate_cot_prompt(domain_prompt, target_name, reasoning_type)
        else:
            full_prompt = domain_prompt

        # 添加生成指令
        if self.language == "english":
            generation_instruction = "\n\nBased on the above analysis, generate a novel GBM candidate molecule, including SMILES string and design rationale."
        else:
            generation_instruction = "\n\n请基于以上分析，生成一个新的GBM候选分子，包括SMILES字符串和设计理由。"
        full_prompt += generation_instruction

        return full_prompt

    def build_tot_propose_prompt(self, domain_prompt: str, current_state: Dict[str, Any], 
                                 step_type: str) -> str:
        """
        构建ToT提议阶段的Prompt
        
        Args:
            domain_prompt: 领域知识prompt
            current_state: 当前状态字典，包含：
                - step_type='scaffold': target_name, mw_range, bbb_requirement, logp_range
                - step_type='assembly': selected_scaffold, scaffold_mw, remaining_mw, target_mw
                - step_type='smiles': selected_scaffold, assembly_strategy, warhead_type, 
                                     bbb_enhancers, target_mw, target_logp_range
            step_type: 'scaffold', 'assembly', 或 'smiles'
        
        Returns:
            格式化后的ToT提议prompt
        """
        tot_templates = self.prompt_templates.get('tot_design_nodes', {})
        
        if step_type == 'scaffold':
            template = tot_templates.get('propose_scaffold', '')
            if template:
                prompt = template.format(
                    domain_prompt=domain_prompt,
                    target_name=current_state.get('target_name', 'EGFR'),
                    mw_range=current_state.get('mw_range', '300-500'),
                    bbb_requirement=current_state.get('bbb_requirement', 'high'),
                    logp_range=current_state.get('logp_range', '2.0-4.0')
                )
                return prompt
        
        elif step_type == 'assembly':
            template = tot_templates.get('propose_assembly', '')
            if template:
                prompt = template.format(
                    domain_prompt=domain_prompt,
                    selected_scaffold=current_state.get('selected_scaffold', 'quinazoline'),
                    scaffold_mw=current_state.get('scaffold_mw', 200),
                    remaining_mw=current_state.get('remaining_mw', 300),
                    target_mw=current_state.get('target_mw', 500)
                )
                return prompt
        
        elif step_type == 'smiles':
            template = tot_templates.get('generate_smiles', '')
            if template:
                prompt = template.format(
                    domain_prompt=domain_prompt,
                    selected_scaffold=current_state.get('selected_scaffold', 'quinazoline'),
                    assembly_strategy=current_state.get('assembly_strategy', 'Strategy 1'),
                    warhead_type=current_state.get('warhead_type', 'acrylamide'),
                    bbb_enhancers=current_state.get('bbb_enhancers', 'fluorine, methoxy'),
                    target_mw=current_state.get('target_mw', 500),
                    target_logp_range=current_state.get('target_logp_range', '2.0-4.0'),
                    target_tpsa_range=current_state.get('target_tpsa_range', '40-120 Å²')
                )
                return prompt
        
        # 如果模板不存在，返回默认提示
        if self.language == "english":
            return f"{domain_prompt}\n\nPropose {step_type} options for GBM drug design."
        else:
            return f"{domain_prompt}\n\n为GBM药物设计提出{step_type}选项。"

    def build_feedback_injected_prompt(
        self,
        domain_prompt: str,
        physical_feedback: str,
        previous_smiles: str = "",
        next_action_hint: str = "请基于物理引擎的测试结果，继续优化分子结构。",
    ) -> str:
        """
        将外部物理评估结果注入到 RAG Prompt 中（替换原来的 LLM 自评估反馈）。

        示例修改前：
            请基于上次的打分继续优化：{llm_fake_score}

        示例修改后：
            物理引擎真实测试结果如下：
            Vina得分=-9.2 kcal/mol (强结合)。
            肝毒性(DILI)=0.12 (低风险)。
            BBB穿透=0.78 (高CNS渗透)。
            理化性质: MW=452.3, LogP=3.21, TPSA=68.4Å²。
            Reward=0.84。

            请在保持原有结合模式的前提下，调整基团...

        参考 chemcrow/prompts.py: 将工具运行结果嵌入 Prompt 的模式

        Args:
            domain_prompt: 领域知识 RAG prompt（保持不变）
            physical_feedback: 物理引擎反馈文本（来自 PhysicalEvaluationResult.build_feedback_text()）
            previous_smiles: 上一轮的 SMILES（可选，用于对比）
            next_action_hint: 给 LLM 的下一步行动提示

        Returns:
            完整的注入物理反馈的 Prompt
        """
        tot_templates = self.prompt_templates.get('tot_design_nodes', {})

        # 优先使用 YAML 中的模板（如果存在）
        template = tot_templates.get('feedback_injected_propose', '')

        if template:
            return template.format(
                domain_prompt=domain_prompt,
                physical_feedback=physical_feedback,
                previous_smiles=previous_smiles,
                next_action_hint=next_action_hint,
            )

        # 回退：手动拼接反馈
        if self.language == "english":
            feedback_prompt = (
                f"{domain_prompt}\n\n"
                "=== PHYSICAL ENGINE EVALUATION RESULTS ===\n"
                f"{physical_feedback}\n\n"
                "=== GUIDANCE FOR NEXT ITERATION ===\n"
                f"{next_action_hint}\n"
            )
        else:
            feedback_prompt = (
                f"{domain_prompt}\n\n"
                "=== 物理引擎真实测试结果 ===\n"
                f"{physical_feedback}\n\n"
                "=== 下一步优化指导 ===\n"
                f"{next_action_hint}\n"
            )

        return feedback_prompt

    def build_tot_propose_prompt_with_feedback(
        self,
        domain_prompt: str,
        current_state: Dict[str, Any],
        step_type: str,
        physical_feedback: Optional[str] = None,
    ) -> str:
        """
        构建 ToT 提议 Prompt，附带物理评估反馈。

        如果提供了 physical_feedback，将其无缝嵌入到 RAG prompt 中，
        让 LLM 在有真实数据的情况下进行下一轮优化。

        Args:
            domain_prompt: 领域知识 RAG prompt
            current_state: 当前状态字典
            step_type: 'scaffold', 'assembly', 或 'smiles'
            physical_feedback: 物理引擎反馈文本（可选）

        Returns:
            完整的 ToT 提议 prompt
        """
        # 先获取基础 prompt
        base_prompt = self.build_tot_propose_prompt(domain_prompt, current_state, step_type)

        # 如果有物理反馈，注入
        if physical_feedback:
            return self.build_feedback_injected_prompt(
                domain_prompt=base_prompt,
                physical_feedback=physical_feedback,
                next_action_hint=self._get_feedback_action_hint(step_type),
            )

        return base_prompt

    def _get_feedback_action_hint(self, step_type: str) -> str:
        """根据当前 ToT 步骤获取对应的行动提示"""
        hints = {
            'scaffold': (
                "Please propose new scaffold options that address the issues above, "
                "while maintaining good BBB penetration potential."
            ),
            'assembly': (
                "Please propose improved assembly strategies that enhance binding affinity "
                "and maintain BBB penetration."
            ),
            'smiles': (
                "Please generate new SMILES that address the issues above, "
                "maintaining the binding mode while reducing toxicity."
            ),
        }
        return hints.get(step_type, "Please continue optimizing the molecular design.")
