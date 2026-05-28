"""
结构知识注入系统
使用字符串形式存储和注入分子结构信息到GBM生成prompt中
"""

from typing import Dict, List, Any, Optional
import json


class StructuralKnowledgeInjector:
    """分子结构知识注入器"""

    def __init__(self, language: str = "english"):
        self.language = language
        self.structure_patterns = self._load_structure_patterns()

    def _load_structure_patterns(self) -> Dict[str, Any]:
        """加载分子结构模式库"""
        if self.language == "english":
            return {
                "scaffold_types": {
                    "quinazoline": "tyrosine kinase inhibitor scaffold with EGFR selectivity",
                    "pyrimidine": "versatile scaffold for multi-target kinase inhibition",
                    "benzimidazole": "privileged scaffold with good BBB permeability",
                    "indole": "common scaffold in GBM stem cell targeting agents",
                    "pyrrole": "small scaffold for CNS penetration optimization"
                },

                "functional_groups": {
                    "acrylamide": "electrophilic warhead for covalent EGFR inhibition",
                    "morpholine": "solubility enhancer and H-bond acceptor",
                    "fluorine": "metabolic stability improver and logP optimizer",
                    "methoxy": "electronic modulator and BBB permeability enhancer",
                    "piperazine": "basic group for aqueous solubility",
                    "dimethylamine": "cationic group for transporter-mediated BBB entry"
                },

                "pharmacophores": {
                    "Type_I_kinase": "ATP-competitive binding in active kinase conformation",
                    "Type_II_kinase": "ATP-competitive binding in inactive DFG-out conformation",
                    "Type_III_kinase": "allosteric non-ATP competitive inhibition",
                    "covalent_kinase": "irreversible binding via cysteine residue",
                    "monopolar": "single charged group for BBB transport",
                    "zwitterionic": "balanced charge for enhanced CNS penetration"
                },

                "linking_strategies": {
                    "amide_linkage": "stable linker with H-bonding capability",
                    "ether_linkage": "flexible linker for conformational freedom",
                    "urea_linkage": "H-bond donor/acceptor for kinase hinge binding",
                    "sulfonamide": "acidic group for pKa modulation"
                },

                "bbb_optimization": {
                    "molecular_weight": "< 450 Da for optimal passive diffusion",
                    "logP_range": "2-4 for balanced lipophilicity",
                    "tpsa_limit": "< 90 Å² to avoid efflux transporter substrates",
                    "rotatable_bonds": "< 8 for reduced conformational flexibility",
                    "polar_groups": "strategic placement for transporter interaction"
                },

                "gbm_specific_motifs": {
                    "egfr_viii_selective": "truncated extracellular domain targeting",
                    "stem_cell_targets": "SOX2/OLIG2/CD133 expression modulators",
                    "mesenchymal_transition": "EMT pathway inhibitors",
                    "hypoxia_adaptation": "HIF-1α pathway interference",
                    "immune_checkpoint": "PD-L1/PD-1 interaction blockers"
                }
            }
        else:
            return {
                "scaffold_types": {
                    "喹唑啉": "酪氨酸激酶抑制剂骨架，具有EGFR选择性",
                    "嘧啶": "多靶点激酶抑制的通用骨架",
                    "苯并咪唑": "特权结构，具有良好的BBB渗透性",
                    "吲哚": "GBM干细胞靶向剂中的常见骨架",
                    "吡咯": "小骨架，用于CNS渗透优化"
                },

                "functional_groups": {
                    "丙烯酰胺": "共价EGFR抑制的亲电性战斗部",
                    "吗啉": "溶解度增强剂和氢键受体",
                    "氟": "代谢稳定性改善剂和logP优化剂",
                    "甲氧基": "电子调节剂和BBB渗透增强剂",
                    "哌嗪": "碱性基团，用于水溶性",
                    "二甲胺": "阳离子基团，用于转运蛋白介导的BBB进入"
                },

                "pharmacophores": {
                    "I型激酶": "活性激酶构象中的ATP竞争性结合",
                    "II型激酶": "无活性DFG-out构象中的ATP竞争性结合",
                    "III型激酶": "别构非ATP竞争性抑制",
                    "共价激酶": "通过半胱氨酸残基的不可逆结合",
                    "单极": "单个带电基团用于BBB转运",
                    "两性离子": "平衡电荷以增强CNS渗透"
                },

                "linking_strategies": {
                    "酰胺键": "稳定的键合，具有氢键能力",
                    "醚键": "灵活的键合，用于构象自由度",
                    "脲键": "激酶铰链结合的氢键供/受体",
                    "磺酰胺": "酸性基团用于pKa调节"
                },

                "bbb_optimization": {
                    "分子量": "< 450 Da 以实现最佳被动扩散",
                    "logP范围": "2-4 以实现平衡的亲脂性",
                    "TPSA限制": "< 90 Å² 以避免外排转运蛋白底物",
                    "可旋转键": "< 8 以减少构象灵活性",
                    "极性基团": "策略性放置以与转运蛋白相互作用"
                },

                "gbm_specific_motifs": {
                    "EGFRvIII选择性": "截短的细胞外结构域靶向",
                    "干细胞靶点": "SOX2/OLIG2/CD133表达调节剂",
                    "间充质转化": "EMT通路抑制剂",
                    "缺氧适应": "HIF-1α通路干扰",
                    "免疫检查点": "PD-L1/PD-1相互作用阻断剂"
                }
            }

    def inject_structural_context(self, prompt: str, target: str, molecule_context: Optional[Dict[str, Any]] = None) -> str:
        """注入结构知识到prompt中"""
        structural_info = self.get_target_specific_structures(target, molecule_context)

        if self.language == "english":
            injection_text = f"\n\nStructural Design Guidelines:\n{structural_info}"
        else:
            injection_text = f"\n\n分子设计指导原则：\n{structural_info}"

        return prompt + injection_text

    def get_target_specific_structures(self, target: str, molecule_context: Optional[Dict[str, Any]] = None) -> str:
        """获取靶点特定的结构指导"""
        guidance = []

        # 靶点特定的结构建议
        target_structures = self._get_target_structure_guidance(target)
        guidance.extend(target_structures)

        # BBB优化指导
        bbb_guidance = self._get_bbb_optimization_guidance()
        guidance.extend(bbb_guidance)

        # 分子上下文特定的建议
        if molecule_context:
            context_guidance = self._get_context_specific_guidance(molecule_context)
            guidance.extend(context_guidance)

        return "\n".join(guidance)

    def _get_target_structure_guidance(self, target: str) -> List[str]:
        """获取靶点特定的结构指导"""
        guidance = []

        if target == "EGFR":
            if self.language == "english":
                guidance = [
                    "• Scaffold preference: Quinazoline or pyrimidine-based structures",
                    "• Key pharmacophore: Type I or Type II kinase binding mode",
                    "• Functional groups: Acrylamide for covalent inhibition, morpholine for solubility",
                    "• Selectivity optimization: EGFRvIII-specific binding pockets",
                    "• BBB considerations: <400 Da molecular weight, logP 2-3"
                ]
            else:
                guidance = [
                    "• 骨架偏好：喹唑啉或嘧啶类结构",
                    "• 关键药效团：I型或II型激酶结合模式",
                    "• 功能基团：丙烯酰胺用于共价抑制，吗啉用于溶解度",
                    "• 选择性优化：EGFRvIII特异性结合口袋",
                    "• BBB考虑：分子量<400 Da，logP 2-3"
                ]

        elif target == "VEGF_VEGFR":
            if self.language == "english":
                guidance = [
                    "• Scaffold preference: Indazole or pyridine-based structures",
                    "• Key pharmacophore: Type II kinase inhibitors",
                    "• Functional groups: Urea linkages, sulfonamides for acidity",
                    "• Anti-angiogenic optimization: VEGFR2 selectivity",
                    "• BBB considerations: Zwitterionic character for transport"
                ]
            else:
                guidance = [
                    "• 骨架偏好：吲唑或吡啶类结构",
                    "• 关键药效团：II型激酶抑制剂",
                    "• 功能基团：脲键，磺酰胺用于酸性",
                    "• 抗血管生成优化：VEGFR2选择性",
                    "• BBB考虑：两性离子特性用于转运"
                ]

        elif target == "GBM_Stem_Cells":
            if self.language == "english":
                guidance = [
                    "• Scaffold preference: Natural product-inspired structures",
                    "• Key targets: SOX2, OLIG2, CD133 expression",
                    "• Functional groups: Phenolic groups, Michael acceptors",
                    "• Differentiation induction: Epigenetic modulators",
                    "• BBB considerations: Small molecules <350 Da"
                ]
            else:
                guidance = [
                    "• 骨架偏好：天然产物启发的结构",
                    "• 关键靶点：SOX2、OLIG2、CD133表达",
                    "• 功能基团：酚基团，Michael受体",
                    "• 分化诱导：表观遗传调控剂",
                    "• BBB考虑：小分子<350 Da"
                ]

        else:
            # 通用GBM指导
            if self.language == "english":
                guidance = [
                    "• Scaffold preference: CNS-penetrant privileged structures",
                    "• Key considerations: BBB permeability, kinase selectivity",
                    "• Functional groups: Optimize for metabolic stability",
                    "• Multi-target potential: Address GBM heterogeneity"
                ]
            else:
                guidance = [
                    "• 骨架偏好：CNS渗透的特权结构",
                    "• 关键考虑：BBB渗透性，激酶选择性",
                    "• 功能基团：优化代谢稳定性",
                    "• 多靶点潜力：解决GBM异质性"
                ]

        return guidance

    def _get_bbb_optimization_guidance(self) -> List[str]:
        """获取BBB优化指导"""
        if self.language == "english":
            return [
                "• Molecular weight: Target <450 Da for passive diffusion",
                "• Lipophilicity: logP 2-4 for optimal brain penetration",
                "• Polar surface area: <90 Å² to avoid efflux transporters",
                "• Hydrogen bonding: ≤3 donors, ≤6 acceptors",
                "• Rotatable bonds: Minimize (<8) for reduced flexibility"
            ]
        else:
            return [
                "• 分子量：目标<450 Da以实现被动扩散",
                "• 亲脂性：logP 2-4以实现最佳脑渗透",
                "• 极性表面积：<90 Å²以避免外排转运蛋白",
                "• 氢键：≤3个供体，≤6个受体",
                "• 可旋转键：最小化(<8)以减少灵活性"
            ]

    def _get_context_specific_guidance(self, molecule_context: Dict[str, Any]) -> List[str]:
        """基于分子上下文提供特定指导"""
        guidance = []

        # 基于现有分子性质的建议
        if "current_logp" in molecule_context:
            logp = molecule_context["current_logp"]
            if self.language == "english":
                if logp < 2:
                    guidance.append("• Increase lipophilicity: Add alkyl groups or reduce polar groups")
                elif logp > 4:
                    guidance.append("• Reduce lipophilicity: Add polar groups or remove hydrophobic substituents")
            else:
                if logp < 2:
                    guidance.append("• 增加亲脂性：添加烷基或减少极性基团")
                elif logp > 4:
                    guidance.append("• 减少亲脂性：添加极性基团或去除疏水取代基")

        # 基于分子大小的建议
        if "current_mw" in molecule_context:
            mw = molecule_context["current_mw"]
            if self.language == "english":
                if mw > 500:
                    guidance.append("• Reduce molecular weight: Simplify structure or remove heavy groups")
            else:
                if mw > 500:
                    guidance.append("• 减少分子量：简化结构或去除重基团")

        return guidance

    def get_scaffold_recommendations(self, target: str) -> List[str]:
        """获取骨架推荐"""
        if target in ["EGFR", "VEGF_VEGFR"]:
            return ["quinazoline", "pyrimidine", "indazole"]
        elif target == "GBM_Stem_Cells":
            return ["indole", "benzimidazole", "natural_product_inspired"]
        else:
            return ["bbb_optimized_small_molecules"]

    def get_functional_group_suggestions(self, context: str) -> List[str]:
        """获取功能基团建议"""
        if context == "solubility":
            return ["morpholine", "piperazine", "hydroxyl"]
        elif context == "bbb_penetration":
            return ["methyl", "fluoro", "methoxy"]
        elif context == "target_binding":
            return ["acrylamide", "urea", "sulfonamide"]
        else:
            return ["optimize_based_on_adme_properties"]
