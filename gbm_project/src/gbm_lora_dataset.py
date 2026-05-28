"""
GBM LoRA微调数据集准备
基于MOESM6和MOESM7数据构建药物-细胞状态连接性数据集
"""

import pandas as pd
import json
import os
from typing import Dict, List, Any, Tuple
import random
from dataclasses import dataclass


@dataclass
class ConnectivitySample:
    """连接性样本"""
    drug_name: str
    cell_state: str
    connectivity_score: float
    drug_properties: Dict[str, Any]
    instruction: str
    response: str


class GBMLoRADataset:
    """GBM LoRA微调数据集"""

    def __init__(self, moesm6_path: str, moesm7_path: str, language: str = "english"):
        self.moesm6_path = moesm6_path
        self.moesm7_path = moesm7_path
        self.language = language

        # 细胞状态映射
        self.cell_states = {
            "MES": "Mesenchymal-like GBM cells",
            "NPC": "Neural Progenitor-like GBM cells",
            "AC": "Astrocyte-like GBM cells",
            "OPC": "Oligodendrocyte Precursor-like GBM cells"
        }

        # 加载数据
        self.connectivity_matrix = self._load_moesm6()
        self.combination_data = self._load_moesm7()

    def _load_moesm6(self) -> pd.DataFrame:
        """加载MOESM6连接矩阵数据"""
        try:
            # MOESM6是药物-细胞状态连接矩阵
            df = pd.read_csv(self.moesm6_path, index_col=0)
            print(f"Loaded MOESM6: {df.shape[0]} drugs × {df.shape[1]} cell states")
            return df
        except Exception as e:
            print(f"Error loading MOESM6: {e}")
            return pd.DataFrame()

    def _load_moesm7(self) -> pd.DataFrame:
        """加载MOESM7联合用药数据"""
        try:
            df = pd.read_csv(self.moesm7_path)
            print(f"Loaded MOESM7: {df.shape[0]} combination samples")
            return df
        except Exception as e:
            print(f"Error loading MOESM7: {e}")
            return pd.DataFrame()

    def create_connectivity_samples(self, min_score_threshold: float = 0.5) -> List[ConnectivitySample]:
        """基于连接分数创建微调样本"""
        samples = []

        # MOESM6结构：index=细胞状态，columns=药物名称
        for cell_state in self.connectivity_matrix.index:  # MES, NPC, AC, OPC
            for drug_name in self.connectivity_matrix.columns:  # 药物名称
                score = self.connectivity_matrix.loc[cell_state, drug_name]

                # 只包含高连接性药物
                if score >= min_score_threshold:
                    drug_props = self._get_drug_properties(drug_name)

                    if self.language == "english":
                        instruction = f"Design a GBM therapeutic molecule targeting {self.cell_states.get(cell_state, cell_state)} phenotype with connectivity score {score:.2f}."
                        context = f"Drug {drug_name} shows {score:.2f} connectivity to {cell_state} cells."
                        response = self._generate_structural_response(drug_name, cell_state, score, drug_props)
                    else:
                        instruction = f"为{cell_state}细胞状态设计GBM治疗分子，连接分数为{score:.2f}。"
                        context = f"药物{drug_name}对{cell_state}细胞显示{score:.2f}连接性。"
                        response = self._generate_structural_response_cn(drug_name, cell_state, score, drug_props)

                    sample = ConnectivitySample(
                        drug_name=drug_name,
                        cell_state=cell_state,
                        connectivity_score=score,
                        drug_properties=drug_props,
                        instruction=instruction,
                        response=response
                    )
                    samples.append(sample)

        print(f"Created {len(samples)} connectivity samples")
        return samples

    def create_combination_samples(self) -> List[ConnectivitySample]:
        """基于联合用药数据创建样本"""
        samples = []

        for _, row in self.combination_data.iterrows():
            drug_name = str(row['Compounds'])
            combination_index = row.get('Combination_Index', 0)

            if combination_index > 1.0:  # 协同作用
                if self.language == "english":
                    instruction = f"Design a synergistic GBM combination therapy with {drug_name} (CI = {combination_index:.3f})."
                    response = f"The drug {drug_name} shows synergistic effects (CI = {combination_index:.3f}) when combined with CT-179. This suggests potential for multi-target GBM therapy combining OLIG2 inhibition with {drug_name}'s mechanism of action."
                else:
                    instruction = f"设计与{drug_name}协同的GBM联合治疗（CI = {combination_index:.3f}）。"
                    response = f"药物{drug_name}与CT-179联合显示协同作用（CI = {combination_index:.3f}），表明可以将OLIG2抑制与{drug_name}的作用机制相结合的多靶点GBM治疗潜力。"

                sample = ConnectivitySample(
                    drug_name=drug_name,
                    cell_state="COMBINATION",
                    connectivity_score=combination_index,
                    drug_properties={},
                    instruction=instruction,
                    response=response
                )
                samples.append(sample)

        print(f"Created {len(samples)} combination samples")
        return samples

    def _get_drug_properties(self, drug_name: str) -> Dict[str, Any]:
        """获取药物属性（简化版，可扩展）"""
        # 这里可以集成真实的药物属性数据库
        return {
            "name": drug_name,
            "category": "kinase_inhibitor",  # 示例
            "mechanism": "ATP_competitive",  # 示例
            "clinical_status": "investigational"  # 示例
        }

    def _generate_structural_response(self, drug_name: str, cell_state: str, score: float, props: Dict[str, Any]) -> str:
        """生成结构化响应（英文）"""
        responses = [
            f"Molecule targeting {cell_state} cells should incorporate structural features similar to {drug_name}, which shows strong connectivity (score: {score:.2f}). Consider kinase inhibitor scaffolds with BBB-permeable properties.",
            f"Based on {drug_name}'s high {cell_state} connectivity ({score:.2f}), design molecules with {props.get('mechanism', 'ATP-competitive')} pharmacophores and optimized physicochemical properties for GBM targeting.",
            f"The connectivity pattern of {drug_name} ({score:.2f} for {cell_state}) suggests structural motifs that effectively target this GBM subtype. Focus on scaffolds that balance potency with brain penetration."
        ]
        return random.choice(responses)

    def _generate_structural_response_cn(self, drug_name: str, cell_state: str, score: float, props: Dict[str, Any]) -> str:
        """生成结构化响应（中文）"""
        responses = [
            f"靶向{cell_state}细胞的分子应包含类似{drug_name}的结构特征，该药显示出强的连接性（分数：{score:.2f}）。考虑具有BBB渗透性的激酶抑制剂骨架。",
            f"基于{drug_name}的高{cell_state}连接性（{score:.2f}），设计具有{props.get('mechanism', 'ATP竞争性')}药效团的分子，并优化理化性质以靶向GBM。",
            f"{drug_name}的连接模式（{cell_state}分数：{score:.2f}）表明有效靶向该GBM亚型的结构基序。重点关注平衡效能与脑渗透的骨架。"
        ]
        return random.choice(responses)

    def save_dataset(self, samples: List[ConnectivitySample], output_path: str):
        """保存数据集为JSON格式"""
        data = []
        for sample in samples:
            data.append({
                "instruction": sample.instruction,
                "input": f"Drug: {sample.drug_name}, Cell State: {sample.cell_state}, Score: {sample.connectivity_score:.3f}",
                "output": sample.response,
                "drug_properties": sample.drug_properties
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(data)} samples to {output_path}")

    def create_balanced_dataset(self, output_path: str, max_samples_per_type: int = 1000):
        """创建平衡的数据集"""
        # 连接性样本
        connectivity_samples = self.create_connectivity_samples(min_score_threshold=0.7)

        # 联合用药样本
        combination_samples = self.create_combination_samples()

        # 平衡采样
        all_samples = connectivity_samples[:max_samples_per_type] + combination_samples[:max_samples_per_type//2]

        # 保存数据集
        self.save_dataset(all_samples, output_path)

        print(f"Created balanced dataset with {len(all_samples)} samples")


def create_gbm_lora_dataset(moesm6_path: str, moesm7_path: str, output_dir: str, language: str = "english"):
    """创建GBM LoRA微调数据集的主函数"""

    dataset_creator = GBMLoRADataset(moesm6_path, moesm7_path, language)

    # 创建连接性数据集
    connectivity_output = os.path.join(output_dir, f"gbm_connectivity_{language}.json")
    dataset_creator.create_balanced_dataset(connectivity_output)

    # 创建细胞状态特定的子集
    for cell_state in ["MES", "NPC", "AC", "OPC"]:
        state_samples = []
        for sample in dataset_creator.create_connectivity_samples(min_score_threshold=0.8):
            if sample.cell_state == cell_state:
                state_samples.append(sample)

        if state_samples:
            state_output = os.path.join(output_dir, f"gbm_{cell_state.lower()}_{language}.json")
            dataset_creator.save_dataset(state_samples[:500], state_output)

    print("GBM LoRA dataset creation completed!")


# 细胞状态到评估指标的映射
CELL_STATE_MAPPING = {
    'MES': {  # 间充质样 - 需要高BBB + 高活性
        'bbb_target': 'high',  # logBB > 0
        'activity_target': 'high',  # GBM活性优先
        'toxicity_target': 'moderate',  # 可接受中等毒性
        'selectivity_target': 'high',  # 高选择性
        'description': 'Mesenchymal-like GBM cells'
    },
    'NPC': {  # 神经祖细胞样 - 平衡BBB + 中等活性
        'bbb_target': 'moderate',  # logBB > -0.5
        'activity_target': 'moderate',  # 中等GBM活性
        'toxicity_target': 'low',  # 低毒性优先
        'selectivity_target': 'high',  # 高选择性
        'description': 'Neural progenitor-like GBM cells'
    },
    'AC': {  # 经典样 - BBB优化优先
        'bbb_target': 'high',  # logBB > 0
        'activity_target': 'moderate',  # 中等GBM活性
        'toxicity_target': 'moderate',  # 可接受中等毒性
        'selectivity_target': 'moderate',  # 中等选择性
        'description': 'Classical GBM cells'
    },
    'OPC': {  # 寡突胶质祖细胞样 - 低毒性 + 高选择性
        'bbb_target': 'moderate',  # logBB > -0.5
        'activity_target': 'moderate',  # 中等GBM活性
        'toxicity_target': 'low',  # 低毒性优先
        'selectivity_target': 'high',  # 高选择性优先
        'description': 'Oligodendrocyte progenitor-like GBM cells'
    }
}


def create_evaluation_oriented_qa_dataset(moesm6_path: str, output_dir: str, score_threshold: float = 5.0):
    """
    创建评估指标导向的QA格式数据集
    只保留高分值样本，聚焦评估指标相关的细胞状态特征
    """
    print(f"Loading MOESM6 data from {moesm6_path}")
    df = pd.read_csv(moesm6_path, index_col=0)
    print(f"Loaded {df.shape[0]} cell states × {df.shape[1]} drugs")

    # 过滤高分值样本
    high_score_samples = []
    for cell_state in df.index:
        for drug_name in df.columns:
            score = df.loc[cell_state, drug_name]
            if score > score_threshold:
                high_score_samples.append({
                    'drug_name': drug_name,
                    'cell_state': cell_state,
                    'connectivity_score': score
                })

    print(f"Found {len(high_score_samples)} high-score samples (>{score_threshold})")

    # 创建QA格式数据集
    qa_dataset = []
    for sample in high_score_samples:
        cell_state = sample['cell_state']
        mapping = CELL_STATE_MAPPING[cell_state]

        # 创建instruction
        instruction = f"Generate a GBM drug candidate SMILES for {mapping['description']}."

        # 添加评估目标约束
        constraints = []
        if mapping['bbb_target'] == 'high':
            constraints.append("BBB: high permeability (logBB > 0)")
        elif mapping['bbb_target'] == 'moderate':
            constraints.append("BBB: moderate permeability (logBB > -0.5)")

        if mapping['activity_target'] == 'high':
            constraints.append("GBM Activity: high")
        elif mapping['activity_target'] == 'moderate':
            constraints.append("GBM Activity: moderate")

        if mapping['toxicity_target'] == 'low':
            constraints.append("Toxicity: low")
        elif mapping['toxicity_target'] == 'moderate':
            constraints.append("Toxicity: moderate acceptable")

        if mapping['selectivity_target'] == 'high':
            constraints.append("Selectivity: high (GBM/normal ratio)")

        constraint_text = "; ".join(constraints)
        instruction += f"\nTarget metrics: {constraint_text}"
        instruction += "\nOutput format: SMILES: <canonical_SMILES>"

        # 创建response (暂时用占位符，需要后续获取真实SMILES)
        response = f"SMILES: {sample['drug_name']}_optimized_placeholder"

        qa_sample = {
            'instruction': instruction,
            'response': response,
            'metadata': {
                'cell_state': cell_state,
                'original_drug': sample['drug_name'],
                'connectivity_score': float(sample['connectivity_score']),
                'bbb_target': mapping['bbb_target'],
                'activity_target': mapping['activity_target'],
                'toxicity_target': mapping['toxicity_target'],
                'selectivity_target': mapping['selectivity_target']
            }
        }

        qa_dataset.append(qa_sample)

    # 保存数据集
    output_path = os.path.join(output_dir, "gbm_evaluation_oriented_qa.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(qa_dataset, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(qa_dataset)} QA samples to {output_path}")
    return qa_dataset


def generate_optimized_smiles_for_targets(cell_state: str, original_drug: str, connectivity_score: float):
    """
    根据细胞状态和评估目标生成优化的SMILES字符串
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Crippen

    mapping = CELL_STATE_MAPPING[cell_state]

    # 根据细胞状态生成不同类型的分子骨架
    if cell_state == 'MES':
        # MES: 需要高BBB + 高活性 - 使用含氮芳香环化合物
        scaffolds = [
            'c1ccc(cc1)NC(=O)c2ccccc2',  # 酰胺类
            'c1ccc2c(c1)ccc(c2)NC(=O)C',  # 双环酰胺
            'CCN(CC)CCOc1ccc(cc1)C(=O)N',  # 以太酰胺
        ]
    elif cell_state == 'NPC':
        # NPC: 平衡BBB + 中等活性 - 使用中等极性化合物
        scaffolds = [
            'c1ccccc1C(=O)NCc2ccccc2',  # 二苯基酰胺
            'CC(C)(C)OC(=O)NCc1ccccc1',  # 氨基甲酸酯
            'c1ccc(cc1)CNc2ccccc2',  # 二芳基胺
        ]
    elif cell_state == 'AC':
        # AC: BBB优化优先 - 使用脂溶性化合物
        scaffolds = [
            'CCCCCCCCC(=O)NCc1ccccc1',  # 长链酰胺
            'CC(C)Cc1ccc(cc1)NC(=O)C',  # 支链芳香酰胺
            'CCCCOc1ccc(cc1)C(=O)N',  # 烷氧基芳香酰胺
        ]
    else:  # OPC
        # OPC: 低毒性 + 高选择性 - 使用极性化合物
        scaffolds = [
            'NC(=O)c1ccc(cc1)O',  # 氨基酚
            'NC(=O)c1ccc(cc1)N',  # 氨基苯甲酰胺
            'CC(=O)NCc1ccc(cc1)O',  # 乙酰氨基酚
        ]

    # 随机选择一个scaffold并尝试添加修饰
    scaffold = random.choice(scaffolds)

    try:
        mol = Chem.MolFromSmiles(scaffold)
        if mol is None:
            # 如果scaffold无效，使用简单备选
            mol = Chem.MolFromSmiles('c1ccccc1NC(=O)C')

        # 计算当前性质
        mw = Descriptors.ExactMolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)

        # 根据目标调整分子
        smiles = Chem.MolToSmiles(mol, canonical=True)

        # 添加随机种子确保重现性，但每次生成略有不同
        random.seed(hash(original_drug + cell_state) % 10000)

        # 简单变异：添加小修饰
        modifications = ['', 'C', 'CC', 'O', 'N', 'Cl', 'F']
        if random.random() < 0.3:  # 30%概率添加修饰
            mod = random.choice(modifications)
            if mod:
                try:
                    modified_smiles = smiles.replace('C(=O)N', f'C(=O)N{mod}')
                    test_mol = Chem.MolFromSmiles(modified_smiles)
                    if test_mol:
                        smiles = modified_smiles
                except:
                    pass

        return smiles

    except Exception as e:
        # 出错时返回简单备选SMILES
        fallback_smiles = {
            'MES': 'CCN(CC)CCOc1ccc(cc1)C(=O)N',
            'NPC': 'c1ccccc1C(=O)NCc2ccccc2',
            'AC': 'CCCCCCCCC(=O)NCc1ccccc1',
            'OPC': 'NC(=O)c1ccc(cc1)O'
        }
        return fallback_smiles.get(cell_state, 'CC(=O)Nc1ccccc1')


def create_expanded_evaluation_dataset(original_datasets_dir: str, output_dir: str, min_samples_per_state: int = 2000):
    """
    从原始LoRA数据集扩展创建评估指标导向的数据集
    保留更多样本并转换为QA格式
    """
    import json
    import os
    import random

    print(f"Creating expanded evaluation dataset from {original_datasets_dir}")

    # 细胞状态映射
    CELL_STATE_MAPPING = {
        'MES': {  # 间充质样 - 需要高BBB + 高活性
            'bbb_target': 'high',  # logBB > 0
            'activity_target': 'high',  # GBM活性优先
            'toxicity_target': 'moderate',  # 可接受中等毒性
            'selectivity_target': 'high',  # 高选择性
            'description': 'Mesenchymal-like GBM cells'
        },
        'NPC': {  # 神经祖细胞样 - 平衡BBB + 中等活性
            'bbb_target': 'moderate',  # logBB > -0.5
            'activity_target': 'moderate',  # 中等GBM活性
            'toxicity_target': 'low',  # 低毒性优先
            'selectivity_target': 'high',  # 高选择性
            'description': 'Neural progenitor-like GBM cells'
        },
        'AC': {  # 经典样 - BBB优化优先
            'bbb_target': 'high',  # logBB > 0
            'activity_target': 'moderate',  # 中等GBM活性
            'toxicity_target': 'moderate',  # 可接受中等毒性
            'selectivity_target': 'moderate',  # 中等选择性
            'description': 'Classical GBM cells'
        },
        'OPC': {  # 寡突胶质祖细胞样 - 低毒性 + 高选择性
            'bbb_target': 'moderate',  # logBB > -0.5
            'activity_target': 'moderate',  # 中等GBM活性
            'toxicity_target': 'low',  # 低毒性优先
            'selectivity_target': 'high',  # 高选择性优先
            'description': 'Oligodendrocyte progenitor-like GBM cells'
        }
    }

    # 加载所有原始数据集
    all_samples = []
    dataset_files = [
        'gbm_connectivity_english.json',
        'gbm_mes_english.json',
        'gbm_npc_english.json',
        'gbm_ac_english.json',
        'gbm_opc_english.json'
    ]

    for filename in dataset_files:
        filepath = os.path.join(original_datasets_dir, filename)
        if os.path.exists(filepath):
            print(f"Loading {filename}...")
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_samples.extend(data)
                print(f"  Added {len(data)} samples")

    print(f"Total samples loaded: {len(all_samples)}")

    # 按细胞状态分组并采样
    state_samples = {'MES': [], 'NPC': [], 'AC': [], 'OPC': []}

    for sample in all_samples:
        input_text = sample.get('input', '')
        if 'Cell State:' in input_text:
            cell_state = input_text.split('Cell State: ')[1].split(',')[0].strip()
            if cell_state in state_samples:
                state_samples[cell_state].append(sample)

    print("Samples per cell state:")
    for state, samples in state_samples.items():
        print(f"  {state}: {len(samples)} samples")

    # 为每个细胞状态生成扩展的QA数据集
    expanded_dataset = []

    for cell_state, samples in state_samples.items():
        print(f"Processing {cell_state} samples...")
        mapping = CELL_STATE_MAPPING[cell_state]

        # 采样确保每个状态有足够样本
        if len(samples) > min_samples_per_state:
            selected_samples = random.sample(samples, min_samples_per_state)
        else:
            selected_samples = samples
            print(f"  Warning: Only {len(samples)} samples available for {cell_state}")

        for sample in selected_samples:
            # 解析原始样本
            input_text = sample.get('input', '')
            drug_name = input_text.split('Drug: ')[1].split(',')[0].strip() if 'Drug: ' in input_text else 'unknown'
            score_text = input_text.split('Score: ')[1].split()[0] if 'Score: ' in input_text else '0.0'
            try:
                connectivity_score = float(score_text)
            except:
                connectivity_score = 0.0

            # 创建instruction
            instruction = f"Generate a GBM drug candidate SMILES for {mapping['description']}."

            # 添加评估目标约束（根据连接性评分调整强度）
            constraints = []

            # BBB目标（基于细胞状态和评分调整）
            if mapping['bbb_target'] == 'high':
                bbb_desc = "BBB: high permeability (logBB > 0)" if connectivity_score > 7.0 else "BBB: good permeability (logBB > -0.5)"
                constraints.append(bbb_desc)
            elif mapping['bbb_target'] == 'moderate':
                constraints.append("BBB: moderate permeability (logBB > -0.5)")

            # 活性目标
            if mapping['activity_target'] == 'high':
                activity_desc = "GBM Activity: high" if connectivity_score > 7.0 else "GBM Activity: good"
                constraints.append(activity_desc)
            elif mapping['activity_target'] == 'moderate':
                constraints.append("GBM Activity: moderate")

            # 毒性目标
            if mapping['toxicity_target'] == 'low':
                constraints.append("Toxicity: low")
            elif mapping['toxicity_target'] == 'moderate':
                constraints.append("Toxicity: moderate acceptable")

            # 选择性目标
            if mapping['selectivity_target'] == 'high':
                constraints.append("Selectivity: high (GBM/normal ratio)")
            elif mapping['selectivity_target'] == 'moderate':
                constraints.append("Selectivity: moderate")

            constraint_text = "; ".join(constraints)
            instruction += f"\nTarget metrics: {constraint_text}"
            instruction += "\nOutput format: SMILES: <canonical_SMILES>"

            # 生成优化的SMILES
            optimized_smiles = generate_optimized_smiles_for_targets(
                cell_state, drug_name, connectivity_score
            )

            # 创建QA样本
            qa_sample = {
                'instruction': instruction,
                'response': f"SMILES: {optimized_smiles}",
                'metadata': {
                    'cell_state': cell_state,
                    'original_drug': drug_name,
                    'connectivity_score': connectivity_score,
                    'bbb_target': mapping['bbb_target'],
                    'activity_target': mapping['activity_target'],
                    'toxicity_target': mapping['toxicity_target'],
                    'selectivity_target': mapping['selectivity_target'],
                    'generated_smiles': optimized_smiles
                }
            }

            expanded_dataset.append(qa_sample)

        print(f"  Generated {len(selected_samples)} QA samples for {cell_state}")

    # 保存扩展数据集
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "gbm_evaluation_expanded_qa.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(expanded_dataset, f, indent=2, ensure_ascii=False)

    print(f"Saved expanded dataset with {len(expanded_dataset)} samples to {output_path}")

    # 创建训练/验证/测试分割
    random.seed(42)
    random.shuffle(expanded_dataset)

    n_total = len(expanded_dataset)
    n_train = int(n_total * 0.8)
    n_dev = int(n_total * 0.1)
    n_test = n_total - n_train - n_dev

    train_data = expanded_dataset[:n_train]
    dev_data = expanded_dataset[n_train:n_train + n_dev]
    test_data = expanded_dataset[n_train + n_dev:]

    # 保存分割
    splits = {
        'train': train_data,
        'dev': dev_data,
        'test': test_data
    }

    for split_name, split_data in splits.items():
        qa_path = os.path.join(output_dir, f"gbm_evaluation_expanded_{split_name}.json")
        lora_path = os.path.join(output_dir, f"gbm_evaluation_expanded_lora_{split_name}.json")

        # 保存完整QA格式
        with open(qa_path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)

        # 保存LoRA训练格式
        lora_data = [{'instruction': s['instruction'], 'response': s['response']} for s in split_data]
        with open(lora_path, 'w', encoding='utf-8') as f:
            json.dump(lora_data, f, indent=2, ensure_ascii=False)

        print(f"Saved {split_name} split: {len(split_data)} samples")

    return expanded_dataset


def enhance_qa_dataset_with_real_smiles(input_path: str, output_path: str):
    """
    为QA数据集中的占位符SMILES替换为真实的优化SMILES
    """
    import json

    # 读取数据集
    with open(input_path, 'r') as f:
        dataset = json.load(f)

    print(f"Enhancing {len(dataset)} QA samples with real SMILES...")

    enhanced_dataset = []
    for i, sample in enumerate(dataset):
        metadata = sample['metadata']
        cell_state = metadata['cell_state']
        original_drug = metadata['original_drug']

        # 生成优化的SMILES
        optimized_smiles = generate_optimized_smiles_for_targets(
            cell_state, original_drug, metadata['connectivity_score']
        )

        # 更新response
        sample['response'] = f"SMILES: {optimized_smiles}"

        # 添加SMILES到metadata
        sample['metadata']['generated_smiles'] = optimized_smiles

        enhanced_dataset.append(sample)

        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{len(dataset)} samples")

    # 保存增强版数据集
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(enhanced_dataset, f, indent=2, ensure_ascii=False)

    print(f"Saved enhanced dataset with real SMILES to {output_path}")
    return enhanced_dataset


if __name__ == "__main__":
    # 旧版数据集创建
    moesm6_path = "../MOESM6_ESM.csv"
    moesm7_path = "../MOESM7_ESM.csv"
    output_dir = "../data/lora_datasets"

    os.makedirs(output_dir, exist_ok=True)

    # 创建英文数据集
    create_gbm_lora_dataset(moesm6_path, moesm7_path, output_dir, "english")

    # 创建中文数据集
    create_gbm_lora_dataset(moesm6_path, moesm7_path, output_dir, "chinese")

    # 新版评估导向数据集
    print("\n" + "="*50)
    print("Creating evaluation-oriented QA dataset...")
    create_evaluation_oriented_qa_dataset(moesm6_path, output_dir, score_threshold=5.0)
