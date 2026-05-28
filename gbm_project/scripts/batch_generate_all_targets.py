#!/usr/bin/env python3
"""
批量GBM靶向药物分子生成脚本（简化版）
==============================
直接使用PeftModel加载模型
为所有靶点分别生成5个独立分子
使用微调的LoRA模型+Qwen2-7B-Instruct+CoT思维链
无需rdkit依赖

使用方式:
python batch_generate_all_targets.py --num_molecules 5 --gpu_id 0
"""

import os
import sys
import json
import torch
import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import logging
import traceback

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'batch_generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class BatchMoleculeGenerator:
    """批量分子生成器（无评估版）"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = f"cuda:{config['gpu_id']}" if torch.cuda.is_available() else "cpu"
        
        # 路径配置
        self.base_model_path = config['base_model_path']
        self.llamole_adapter_path = config['llamole_adapter_path']
        self.gbm_adapter_path = config['gbm_adapter_path']
        
        # 模型
        self.model = None
        self.tokenizer = None
        
        # 靶点列表
        self.targets = config.get('targets', [
            'EGFR', 'VEGFR', 'IDH1', 'MGMT', 'PD1_PDL1', 'PI3K_AKT_mTOR', 'MDM2'
        ])
        
        # 生成配置
        self.generation_config = config.get('generation', {
            'max_new_tokens': 1024,
            'temperature': 0.9,
            'top_p': 0.95,
            'do_sample': True
        })
        
        # COT配置
        self.cot_config = config.get('cot', {
            'enabled': True,
            'reasoning_steps': 7,
            'include_market_reference': True
        })
        
        # 输出目录
        self.output_base_dir = config.get('output_dir', str(
            PROJECT_ROOT / "gbm_project" / "experiments" / 
            f"all_targets_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ))
        
    def load_models(self):
        """直接使用PeftModel加载模型"""
        logger.info("=" * 60)
        logger.info("Loading models with PeftModel...")
        logger.info("=" * 60)
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel
            
            # 加载tokenizer
            logger.info(f"Loading tokenizer from {self.base_model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path,
                trust_remote_code=True,
                padding_side="right"
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载基础模型
            logger.info(f"Loading base model from {self.base_model_path}")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float16,
                device_map={"": self.device},
                trust_remote_code=True
            )
            
            # 加载Llamole适配器
            if self.llamole_adapter_path and os.path.exists(self.llamole_adapter_path):
                logger.info(f"Loading Llamole adapter from {self.llamole_adapter_path}")
                base_model = PeftModel.from_pretrained(base_model, self.llamole_adapter_path)
            
            # 加载GBM适配器
            if self.gbm_adapter_path and os.path.exists(self.gbm_adapter_path):
                logger.info(f"Loading GBM adapter from {self.gbm_adapter_path}")
                self.model = PeftModel.from_pretrained(base_model, self.gbm_adapter_path)
            else:
                self.model = base_model
            
            logger.info(f"Model loaded: {self.model.__class__.__name__}")
            logger.info("✓ All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def get_target_info(self, target_name: str) -> Dict:
        """获取靶点信息"""
        summary_file = PROJECT_ROOT / "gbm_project" / "data" / "processed_targets" / f"{target_name}_summary.json"
        
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                return json.load(f)
        return {}
    
    def generate_prompt(self, target_name: str, target_info: Dict) -> str:
        """生成CoT推理提示词"""
        
        seq_info = target_info.get('sequence', {})
        binding_sites = target_info.get('binding_sites', [])
        
        # 靶点描述
        descriptions = {
            'EGFR': """
EGFR (Epidermal Growth Factor Receptor) in GBM:
- Most common gene amplification (~40% of GBM cases)
- Drives excessive tumor cell proliferation
- Key mutation: EGFRvIII with immunogenicity
- Resistance: T790M, MET amplification
Binding site: Kinase domain ATP pocket
Key residues: Lys745, Asp855, Met769
            """,
            'VEGFR': """
VEGF/VEGFR pathway in GBM:
- Drives pathological angiogenesis
- Creates immunosuppressive microenvironment
Binding site: VEGFR kinase domain
            """,
            'IDH1': """
IDH1/IDH2 mutations in GBM:
- Secondary GBM (~10%)
- 2-HG accumulation, epigenetic effects
Binding site: IDH active site with NADP+ pocket
            """,
            'MGMT': """
MGMT in GBM:
- Repairs TMZ-induced DNA damage
- Epigenetic silencing can reverse resistance
Target: MGMT repair mechanism bypass
            """,
            'PD1_PDL1': """
PD-1/PD-L1 pathway in GBM:
- Immune microenvironment regulation
- Target for immunotherapy combinations
            """,
            'PI3K_AKT_mTOR': """
PI3K/AKT/mTOR pathway in GBM:
- Most commonly activated (~80%)
- PTEN loss, PIK3CA mutation, AKT amp
Binding site: PI3K kinase domain
            """,
            'MDM2': """
MDM2 in GBM:
- E3 ubiquitin ligase for p53
- Amplification ~10%
Target: p53 pathway restoration
            """
        }
        
        target_desc = descriptions.get(target_name, f"Target: {target_name}")
        
        # 结合位点
        sites_desc = ""
        for i, site in enumerate(binding_sites[:3], 1):
            sites_desc += f"\nSite {i}: {site.get('site_name', 'Unknown')}"
            sites_desc += f"\n  Key residues: {', '.join(site.get('key_residues', [])[:5])}"
        
        prompt = f"""<design_start>
Chain-of-Thought Reasoning for {target_name} Inhibitor Design

## Analysis Phase

### Target Analysis
{target_desc}

Sequence length: {seq_info.get('length', 'N/A')} aa
Binding sites:{sites_desc}

### Market Reference
- MW: 350-500 Da, LogP: 2.0-4.0
- Core: heterocyclic aromatic scaffolds
- Interactions: H-bond + hydrophobic contacts

### Design Strategy
1. BBB-permeable core (MW 150-250 Da)
2. Match pharmacophore profile
3. H-bond with key residues
4. Lipinski's rules compliance

### Safety
- AVOID aniline structures (hepatotoxicity)
- Aromatic rings ≤3
- LogP < 4.0

## Generation Phase

Generate 5 novel {target_name} inhibitors.

First, provide your Chain-of-Thought reasoning for molecular design.
Then, output exactly 5 valid SMILES strings for novel {target_name} inhibitors.

After your reasoning, end with:
SMILES1:<valid_smiles_1>
SMILES2:<valid_smiles_2>
SMILES3:<valid_smiles_3>
SMILES4:<valid_smiles_4>
SMILES5:<valid_smiles_5>
"""
        
        return prompt
    
    def generate_molecules(self, target_name: str, num_molecules: int = 5) -> List[Dict]:
        """为特定靶点生成分子"""
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Generating molecules for {target_name}")
        logger.info(f"{'='*60}")
        
        target_info = self.get_target_info(target_name)
        
        if not target_info:
            logger.warning(f"No target info found for {target_name}")
            return []
        
        prompt = self.generate_prompt(target_name, target_info)
        logger.info(f"Prompt length: {len(prompt)} chars")
        
        # 尝试多次生成，直到获得50个SMILES或达到最大尝试次数
        all_smiles = []
        max_attempts = 10
        target_count = num_molecules  # 默认为50
        
        for attempt in range(max_attempts):
            if len(all_smiles) >= target_count:
                break
                
            try:
                response = self._generate(prompt)
                smiles_list = self._extract_smiles(response)
                
                # 记录这次生成的SMILES
                for smiles in smiles_list:
                    if smiles not in all_smiles:
                        all_smiles.append(smiles)
                
                logger.info(f"  Attempt {attempt+1}/{max_attempts}: found {len(all_smiles)}/{target_count} molecules")
                
                if len(all_smiles) >= target_count:
                    break
                    
            except Exception as e:
                logger.error(f"    ✗ Error: {e}")
        
        # 只保留前target_count个SMILES
        molecules = []
        for i, smiles in enumerate(all_smiles[:target_count], 1):
            molecules.append({
                'smiles': smiles,
                'raw_response': "",
                'target': target_name,
                'generation_id': i
            })
        
        logger.info(f"  Generated {len(molecules)} molecules for {target_name}")
        return molecules
    
    def _generate(self, prompt: str) -> str:
        """生成响应"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.generation_config.get('max_new_tokens', 1024),
                temperature=self.generation_config.get('temperature', 0.9),
                top_p=self.generation_config.get('top_p', 0.95),
                do_sample=self.generation_config.get('do_sample', True),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "Response:" in response:
            response = response.split("Response:")[-1].strip()
        elif "<design_end>" in response:
            response = response.split("<design_end>")[0].strip()
        
        return response
    
    def _extract_smiles(self, response: str) -> List[str]:
        """从响应中提取多个SMILES"""
        smiles_list = []
        
        # 模式1: SMILES1:xxx SMILES2:xxx ...
        pattern1 = r'SMILES\d*:\s*([A-Za-z0-9@+\-\[\]\(\)=#%/\s]+)'
        matches1 = re.findall(pattern1, response)
        for match in matches1:
            smiles = match.strip()
            if self._is_valid_smiles_format(smiles):
                smiles_list.append(smiles)
        
        # 模式2: SMILES:xxx
        pattern2 = r'SMILES:\s*([A-Za-z0-9@+\-\[\]\(\)=#%/\s]+)'
        matches2 = re.findall(pattern2, response)
        for match in matches2:
            smiles = match.strip()
            if self._is_valid_smiles_format(smiles) and smiles not in smiles_list:
                smiles_list.append(smiles)
        
        # 模式3: 备用 - 提取可能的SMILES字符串
        # 匹配以大写字母开头，包含SMILES常见字符的字符串
        pattern3 = r'\b([A-Z][A-Za-z0-9@+\-\[\]\(\)=#]{4,100})\b'
        matches3 = re.findall(pattern3, response)
        for match in matches3:
            smiles = match.strip()
            # 排除纯英文单词
            if re.match(r'^[A-Za-z]+$', smiles):
                continue
            if smiles not in smiles_list:
                smiles_list.append(smiles)
        
        return smiles_list
    
    def _is_valid_smiles_format(self, smiles: str) -> bool:
        """检查是否为有效的SMILES格式（宽松验证）"""
        if not smiles or len(smiles) < 5 or len(smiles) > 150:
            return False
        
        # 排除纯英文单词
        if re.match(r'^[A-Za-z]+$', smiles):
            return False
        
        # 排除包含空格的内容（说明是多个词）
        if ' ' in smiles:
            return False
        
        # 排除包含中文的内容
        if re.search(r'[^\x00-\x7F]', smiles):
            return False
        
        # SMILES必须以大写字母或[开头
        if not re.match(r'^[A-Za-z\[\(]', smiles):
            return False
        
        # 必须包含至少一个SMILES特征字符
        # 包括: 大写字母(原子)、@(立体)、+(正电荷)、-(负电荷)、=(双键)、#(三键)
        # :、(环闭合数字)、(、)(支链)、[、](特殊原子)
        if not re.search(r'[A-Za-z@+\-=#()\[\]]', smiles):
            return False
        
        return True
    
    def run_batch(self, num_molecules: int = 50):
        """运行批量生成"""
        
        os.makedirs(self.output_base_dir, exist_ok=True)
        
        logger.info("=" * 60)
        logger.info("Batch Molecule Generation Started")
        logger.info("=" * 60)
        logger.info(f"Targets: {len(self.targets)} - {self.targets}")
        logger.info(f"Molecules per target: {num_molecules}")
        logger.info(f"CoT enabled: {self.cot_config['enabled']}")
        logger.info(f"Output: {self.output_base_dir}")
        logger.info("=" * 60)
        
        # 加载模型
        self.load_models()
        
        all_results = {
            'config': self.config,
            'targets': self.targets,
            'num_molecules_per_target': num_molecules,
            'timestamp': datetime.now().isoformat(),
            'results_by_target': {}
        }
        
        for target in self.targets:
            try:
                molecules = self.generate_molecules(target, num_molecules)
                all_results['results_by_target'][target] = molecules
            except Exception as e:
                logger.error(f"Error processing {target}: {e}")
                all_results['results_by_target'][target] = []
        
        # 保存完整结果
        output_file = os.path.join(self.output_base_dir, "all_targets_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n{'='*60}")
        logger.info("Batch Generation Complete!")
        logger.info(f"{'='*60}")
        logger.info(f"Output saved to: {output_file}")
        
        # 打印摘要
        print("\n" + "=" * 60)
        print("Batch Generation Complete!")
        print("=" * 60)
        total_molecules = 0
        for target, molecules in all_results['results_by_target'].items():
            valid = len(molecules)
            total_molecules += valid
            print(f"  {target}: {valid} molecules")
        print("=" * 60)
        print(f"Total molecules: {total_molecules}")
        print(f"Output: {output_file}")
        print("=" * 60)
        
        return all_results


def main():
    parser = argparse.ArgumentParser(description="Batch generate GBM drug molecules for all targets")
    parser.add_argument('--num_molecules', type=int, default=50, help='Molecules per target')
    parser.add_argument('--gpu_id', type=int, default=1, help='GPU device ID')
    parser.add_argument('--use_cot', action='store_true', default=True, help='Use CoT reasoning')
    parser.add_argument('--targets', type=str, default='EGFR,VEGFR,IDH1,MGMT,PD1_PDL1,PI3K_AKT_mTOR,MDM2',
                        help='Comma-separated target list')
    
    args = parser.parse_args()
    
    config = {
        'gpu_id': args.gpu_id,
        'base_model_path': str(PROJECT_ROOT / "models" / "Qwen2-7B-Instruct"),
        'llamole_adapter_path': str(PROJECT_ROOT / "saves" / "Llamole-Qwen2-7B-Instruct-Adapter"),
        'gbm_adapter_path': str(PROJECT_ROOT / "saves" / "Llamole-Qwen2-7B-Instruct-Adapter-gbm-with-graph-models"),
        'targets': args.targets.split(','),
        'generation': {
            'max_new_tokens': 1024,
            'temperature': 0.9,
            'top_p': 0.95,
            'do_sample': True
        },
        'cot': {
            'enabled': args.use_cot,
            'reasoning_steps': 7,
            'include_market_reference': True
        },
        'output_dir': str(PROJECT_ROOT / "gbm_project" / "experiments" / 
                         f"all_targets_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    }
    
    generator = BatchMoleculeGenerator(config)
    generator.run_batch(args.num_molecules)


if __name__ == "__main__":
    main()
