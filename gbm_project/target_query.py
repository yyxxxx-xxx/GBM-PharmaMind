#!/usr/bin/env python3
"""
靶点结构数据查询工具
====================
功能: 快速查询和处理已提取的靶点结构信息
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from target_processor import TargetProcessor


class TargetQuery:
    """靶点结构信息查询工具"""
    
    def __init__(self, processed_dir: str = None):
        if processed_dir is None:
            self.processed_dir = Path(__file__).parent / "data/processed_targets"
        else:
            self.processed_dir = Path(processed_dir)
        
        # 加载所有靶点摘要
        self.targets = {}
        self._load_all_targets()
    
    def _load_all_targets(self):
        """加载所有靶点摘要"""
        for summary_file in self.processed_dir.glob("*_summary.json"):
            target_name = summary_file.stem.replace("_summary", "")
            with open(summary_file, 'r') as f:
                self.targets[target_name] = json.load(f)
    
    def list_targets(self) -> List[str]:
        """列出所有靶点"""
        return list(self.targets.keys())
    
    def get_target(self, name: str) -> Optional[Dict]:
        """获取靶点信息"""
        return self.targets.get(name)
    
    def get_sequence(self, name: str) -> Optional[str]:
        """获取靶点序列"""
        target = self.get_target(name)
        if target:
            return target.get("sequence", {}).get("sequence")
        return None
    
    def get_binding_sites(self, name: str) -> List[Dict]:
        """获取靶点结合位点"""
        target = self.get_target(name)
        if target:
            return target.get("binding_sites", [])
        return []
    
    def get_pharmacophore(self, name: str) -> Dict:
        """获取药效团特征"""
        target = self.get_target(name)
        if target:
            return target.get("pharmacophore_features", {})
        return {}
    
    def get_key_residues(self, name: str) -> Dict[str, List[str]]:
        """获取关键残基分类"""
        pharm = self.get_pharmacophore(name)
        return {
            "hydrophobic": pharm.get("hydrophobic_regions", [])[:50],
            "hbd": pharm.get("hydrogen_bond_donors", [])[:50],
            "hba": pharm.get("hydrogen_bond_acceptors", [])[:50],
            "aromatic": pharm.get("aromatic_regions", [])[:30],
            "positive": pharm.get("positive_charge", [])[:30],
            "negative": pharm.get("negative_charge", [])[:30]
        }
    
    def generate_model_input(self, name: str) -> Dict:
        """生成模型输入格式"""
        target = self.get_target(name)
        if not target:
            raise ValueError(f"靶点 {name} 不存在")
        
        sequence_info = target.get("sequence", {})
        binding_sites = target.get("binding_sites", [])
        pharm = target.get("pharmacophore_features", {})
        
        return {
            "target_id": name,
            "sequence": sequence_info.get("sequence", ""),
            "sequence_length": sequence_info.get("length", 0),
            "binding_pockets": [
                {
                    "pocket_id": site.get("site_id"),
                    "pocket_name": site.get("site_name"),
                    "center_3d": site.get("center"),
                    "radius": site.get("radius"),
                    "key_residues": site.get("key_residues", []),
                    "properties": site.get("properties", {})
                }
                for site in binding_sites
            ],
            "pharmacophore": {
                "hydrophobic_count": len(pharm.get("hydrophobic_regions", [])),
                "hbd_count": len(pharm.get("hydrogen_bond_donors", [])),
                "hba_count": len(pharm.get("hydrogen_bond_acceptors", [])),
                "aromatic_count": len(pharm.get("aromatic_regions", [])),
                "positive_count": len(pharm.get("positive_charge", [])),
                "negative_count": len(pharm.get("negative_charge", [])),
                "metal_sites": len(pharm.get("metal_coordination_sites", []))
            }
        }
    
    def compare_targets(self, target_names: List[str]) -> Dict:
        """比较多个靶点的特征"""
        comparison = {
            "targets": target_names,
            "sequences": {},
            "binding_sites": {},
            "pharmacophore_summary": {}
        }
        
        for name in target_names:
            target = self.get_target(name)
            if target:
                seq_info = target.get("sequence", {})
                comparison["sequences"][name] = {
                    "length": seq_info.get("length", 0)
                }
                comparison["binding_sites"][name] = len(
                    target.get("binding_sites", [])
                )
                pharm = target.get("pharmacophore_features", {})
                comparison["pharmacophore_summary"][name] = {
                    "hydrophobic": len(pharm.get("hydrophobic_regions", [])),
                    "hbd": len(pharm.get("hydrogen_bond_donors", [])),
                    "hba": len(pharm.get("hydrogen_bond_acceptors", [])),
                    "aromatic": len(pharm.get("aromatic_regions", []))
                }
        
        return comparison
    
    def get_target_info(self, name: str) -> str:
        """获取靶点详细信息（格式化输出）"""
        target = self.get_target(name)
        if not target:
            return f"靶点 {name} 不存在"
        
        seq_info = target.get("sequence", {})
        sites = target.get("binding_sites", [])
        pharm = target.get("pharmacophore_features", {})
        
        info = f"""
{'='*60}
靶点: {name}
{'='*60}
PDB ID: {target.get('pdb_id', 'N/A')}
序列长度: {seq_info.get('length', 'N/A')} 氨基酸
链: {seq_info.get('chains', [])}

结合位点: {len(sites)}
"""
        for i, site in enumerate(sites, 1):
            info += f"  {i}. {site.get('site_name', 'N/A')}\n"
            info += f"     半径: {site.get('radius', 'N/A')} Å\n"
            info += f"     关键残基: {', '.join(site.get('key_residues', [])[:5])}\n"
        
        info += f"""
药效团特征:
  - 疏水残基: {len(pharm.get('hydrophobic_regions', []))}
  - 氢键供体: {len(pharm.get('hydrogen_bond_donors', []))}
  - 氢键受体: {len(pharm.get('hydrogen_bond_acceptors', []))}
  - 芳香环: {len(pharm.get('aromatic_regions', []))}
  - 正电荷: {len(pharm.get('positive_charge', []))}
  - 负电荷: {len(pharm.get('negative_charge', []))}
"""
        
        return info


def main():
    """主函数 - 演示查询工具使用"""
    query = TargetQuery()
    
    print("=" * 60)
    print("GBM靶点结构信息查询工具")
    print("=" * 60)
    
    # 列出所有靶点
    print("\n可用靶点:")
    for i, name in enumerate(query.list_targets(), 1):
        print(f"  {i}. {name}")
    
    # 获取EGFR详细信息
    print(query.get_target_info("EGFR"))
    
    # 生成模型输入示例
    print("\n生成EGFR模型输入:")
    model_input = query.generate_model_input("EGFR")
    print(f"  靶点ID: {model_input['target_id']}")
    print(f"  序列长度: {model_input['sequence_length']}")
    print(f"  结合口袋数: {len(model_input['binding_pockets'])}")
    print(f"  药效团特征: {model_input['pharmacophore']}")
    
    # 比较靶点
    print("\n靶点比较:")
    comparison = query.compare_targets(["EGFR", "VEGFR", "IDH1"])
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()


