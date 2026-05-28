#!/usr/bin/env python3
"""
GBM靶点结构信息提取与处理流程
==============================
功能:
1. 解析CIF文件提取完整结构信息
2. 计算结合位点和药效团特征
3. 生成标准化靶点描述文件
4. 支持靶点隔离和细化查询
"""

import gzip
import json
import re
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
import numpy as np


# ============ 数据结构定义 ============

@dataclass
class ResidueInfo:
    """残基基本信息"""
    seq_num: int          # 序列编号
    aa_type: str          # 氨基酸类型
    chain_id: str         # 链ID
    auth_seq_id: int      # 权威序列编号
    auth_comp_id: str     # 权威组件ID

@dataclass
class AtomInfo:
    """原子信息"""
    id: int
    element: str
    x: float
    y: float
    z: float
    residue_seq_num: int
    residue_name: str
    chain_id: str
    atom_type: str        # 原子类型 (CA, N, C, O, etc.)
    charge: Optional[float] = None
    occupancy: Optional[float] = None
    b_factor: Optional[float] = None

@dataclass
class BindingSite:
    """结合位点信息"""
    site_id: str
    site_name: str
    residues: List[Dict]       # 残基信息列表
    center: Tuple[float, float, float]  # 位点中心坐标
    radius: float              # 口袋半径
    volume: float              # 口袋体积估算
    properties: Dict           # 理化性质
    atoms: List[AtomInfo]      # 关键原子

@dataclass
class LigandInfo:
    """配体/小分子信息"""
    comp_id: str
    name: str
    formula: str
    molecular_weight: float
    atoms: List[AtomInfo]
    binding_residues: List[str]  # 结合的残基
    binding_site_id: Optional[str] = None

@dataclass
class MetalIon:
    """金属离子信息"""
    comp_id: str
    element: str
    coordinates: Tuple[float, float, float]
    coordination_residues: List[str]  # 配位的残基

@dataclass
class SecondaryStructure:
    """二级结构"""
    type: str          # HELIX, SHEET, TURN
    start_seq_num: int
    end_seq_num: int
    chain_id: str

@dataclass
class TargetStructure:
    """完整的靶点结构信息"""
    # 基本信息
    target_name: str
    pdb_id: str
    entity_id: str
    
    # 序列信息
    sequence: str
    sequence_length: int
    
    # 链信息
    chains: List[str]
    
    # 原子信息
    atoms: List[AtomInfo]
    atoms_count: int
    
    # 残基信息
    residues: List[ResidueInfo]
    residues_count: int
    
    # 结合位点
    binding_sites: List[BindingSite]
    
    # 配体信息
    ligands: List[LigandInfo]
    
    # 金属离子
    metal_ions: List[MetalIon]
    
    # 二级结构
    secondary_structures: List[SecondaryStructure]
    
    # 元数据
    resolution: Optional[float]
    experimental_method: str
    title: str
    deposition_date: str
    
    # 额外信息
    gene_name: Optional[str] = None
    organism: Optional[str] = None
    mutation_info: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "target_name": self.target_name,
            "pdb_id": self.pdb_id,
            "metadata": {
                "entity_id": self.entity_id,
                "resolution": self.resolution,
                "experimental_method": self.experimental_method,
                "title": self.title,
                "deposition_date": self.deposition_date,
                "gene_name": self.gene_name,
                "organism": self.organism,
                "mutation_info": self.mutation_info
            },
            "sequence": {
                "sequence": self.sequence,
                "length": self.sequence_length,
                "chains": self.chains
            },
            "structure": {
                "atoms_count": self.atoms_count,
                "residues_count": self.residues_count,
                "secondary_structures_count": len(self.secondary_structures)
            },
            "binding_sites": [
                {
                    "site_id": site.site_id,
                    "site_name": site.site_name,
                    "center": list(site.center),
                    "radius": site.radius,
                    "volume": site.volume,
                    "properties": site.properties,
                    "residue_count": len(site.residues),
                    "atoms_count": len(site.atoms),
                    "key_residues": [r["auth_comp_id"] + str(r["auth_seq_id"]) for r in site.residues[:10]]
                }
                for site in self.binding_sites
            ],
            "ligands": [
                {
                    "comp_id": lig.comp_id,
                    "name": lig.name,
                    "formula": lig.formula,
                    "molecular_weight": lig.molecular_weight,
                    "binding_residues": lig.binding_residues
                }
                for lig in self.ligands
            ],
            "metal_ions": [
                {
                    "comp_id": ion.comp_id,
                    "element": ion.element,
                    "coordinates": list(ion.coordinates),
                    "coordination_residues": ion.coordination_residues
                }
                for ion in self.metal_ions
            ],
            "secondary_structures": [
                {
                    "type": ss.type,
                    "start": ss.start_seq_num,
                    "end": ss.end_seq_num,
                    "chain": ss.chain_id
                }
                for ss in self.secondary_structures
            ],
            "detailed_residues": [
                {
                    "seq_num": r.seq_num,
                    "aa_type": r.aa_type,
                    "chain_id": r.chain_id,
                    "auth_seq_id": r.auth_seq_id
                }
                for r in self.residues
            ],
            "pharmacophore_features": self._generate_pharmacophore_features()
        }
    
    def _generate_pharmacophore_features(self) -> Dict:
        """生成药效团特征"""
        features = {
            "hydrophobic_regions": [],
            "hydrogen_bond_donors": [],
            "hydrogen_bond_acceptors": [],
            "aromatic_regions": [],
            "positive_charge": [],
            "negative_charge": [],
            "metal_coordination_sites": []
        }
        
        # 统计各类型残基
        aa_properties = {
            'hydrophobic': ['ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO'],
            'hbd': ['SER', 'THR', 'TYR', 'CYS', 'LYS', 'ARG', 'HIS', 'TRP'],
            'hba': ['SER', 'THR', 'TYR', 'ASN', 'GLN', 'ASP', 'GLU', 'CYS'],
            'aromatic': ['PHE', 'TYR', 'TRP', 'HIS'],
            'positive': ['LYS', 'ARG', 'HIS'],
            'negative': ['ASP', 'GLU']
        }
        
        for res in self.residues:
            res_info = f"{res.aa_type}{res.auth_seq_id}"
            pos = (res.seq_num,)
            
            if res.aa_type in aa_properties['hydrophobic']:
                features['hydrophobic_regions'].append(res_info)
            if res.aa_type in aa_properties['hbd']:
                features['hydrogen_bond_donors'].append(res_info)
            if res.aa_type in aa_properties['hba']:
                features['hydrogen_bond_acceptors'].append(res_info)
            if res.aa_type in aa_properties['aromatic']:
                features['aromatic_regions'].append(res_info)
            if res.aa_type in aa_properties['positive']:
                features['positive_charge'].append(res_info)
            if res.aa_type in aa_properties['negative']:
                features['negative_charge'].append(res_info)
        
        # 金属配位位点
        for ion in self.metal_ions:
            features['metal_coordination_sites'].append({
                "ion": ion.element,
                "coordination_residues": ion.coordination_residues
            })
        
        return features


# ============ CIF文件解析器 ============

class CIFParser:
    """CIF文件解析器 - 简化版本"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = {}
        
    def parse(self) -> Dict:
        """解析CIF文件"""
        if self.filepath.endswith('.gz'):
            with gzip.open(self.filepath, 'rt') as f:
                content = f.read()
        else:
            with open(self.filepath, 'r') as f:
                content = f.read()
        
        # 解析数据块
        blocks = re.split(r'data_(\S+)\n', content)
        
        for i in range(1, len(blocks), 2):
            block_name = blocks[i]
            block_content = blocks[i + 1] if i + 1 < len(blocks) else ''
            self.data[block_name] = self._parse_block(block_content)
        
        return self.data
    
    def _parse_block(self, content: str) -> Dict:
        """解析单个数据块"""
        result = {'_': {}}
        
        lines = content.split('\n')
        
        # 状态追踪
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # 检测数据块结束
            if line.strip().startswith('data_'):
                break
            
            # 检测loop开始
            if line.strip() == 'loop_':
                loop_start = i
                # 收集所有字段名
                loop_headers = []
                i += 1
                while i < len(lines) and lines[i].strip().startswith('_') and not lines[i].strip().startswith('loop_'):
                    field_name = lines[i].strip().split()[0]
                    loop_headers.append(field_name)
                    i += 1
                
                # 收集数据行
                loop_data = []
                while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith('_') and not lines[i].strip().startswith('loop_') and not lines[i].strip().startswith('data_'):
                    row = lines[i].strip()
                    if not row.startswith('#'):
                        values = self._split_row(row)
                        if len(values) == len(loop_headers):
                            row_dict = {}
                            for h, v in zip(loop_headers, values):
                                if v not in ['?', '.']:
                                    row_dict[h] = v
                            if row_dict:
                                loop_data.append(row_dict)
                    i += 1
                
                # 保存loop数据
                if loop_headers and loop_data:
                    # 从第一个字段名提取表名
                    first_header = loop_headers[0]
                    if first_header.startswith('_'):
                        parts = first_header[1:].split('.', 1)
                        table_name = parts[0] if parts else first_header[1:]
                    else:
                        table_name = first_header
                    
                    # 如果表名已存在，添加后缀
                    base_name = table_name
                    counter = 1
                    while table_name in result:
                        table_name = f"{base_name}_{counter}"
                        counter += 1
                    result[table_name] = loop_data
                
                continue
            
            # 处理简单键值对
            if line.strip().startswith('_') and not line.strip().startswith('loop_'):
                parts = line.strip().split(None, 1)
                if len(parts) == 2:
                    key = parts[0]
                    value = parts[1]
                    if value.strip() not in ['?', '.']:
                        result['_'][key] = value.strip()
            
            i += 1
        
        return result
    
    def _split_row(self, row: str) -> List[str]:
        """分割行，处理带引号的值"""
        values = []
        current = ''
        in_quote = False
        quote_char = None
        
        for char in row:
            if char in ['"', "'"] and not in_quote:
                in_quote = True
                quote_char = char
            elif char == quote_char and in_quote:
                in_quote = False
            elif char == ' ' and not in_quote:
                if current:
                    values.append(current.strip('"').strip("'"))
                    current = ''
            else:
                current += char
        
        if current:
            values.append(current.strip('"').strip("'"))
        
        return values


# ============ 靶点结构提取器 ============

class TargetStructureExtractor:
    """靶点结构信息提取器"""
    
    # 氨基酸三字母到单字母映射
    AA_3TO1 = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }
    
    # 已知配体名称映射
    LIGAND_NAMES = {
        'AZD': 'Osimertinib (AZD9291)',
        'ANP': 'Adenosine phosphate',
        'ATP': 'Adenosine triphosphate',
        'HEM': 'Heme',
        'HEM': 'Heme',
        'ZN': 'Zinc ion',
        'MG': 'Magnesium ion',
        'MN': 'Manganese ion',
        'FE': 'Iron ion',
        'NAP': 'NADP',
        'NAD': 'NADH',
        'FAD': 'FAD',
        'STU': 'Staurosporine',
        'IMA': 'Imatinib',
        'ERL': 'Erlotinib',
        'GFI': 'Gefitinib',
        'LAF': 'Lapatinib',
    }
    
    def __init__(self, cif_path: str, target_name: str):
        self.cif_path = cif_path
        self.target_name = target_name
        self.parser = CIFParser(cif_path)
        self.data = {}
        
    def extract(self) -> TargetStructure:
        """提取完整结构信息"""
        self.data = self.parser.parse()
        
        # 获取第一个数据块
        block_name = list(self.data.keys())[0]
        block = self.data[block_name]
        
        # 提取基本信息
        metadata = self._extract_metadata(block)
        
        # 提取序列
        sequence_info = self._extract_sequence(block)
        
        # 提取原子
        atoms = self._extract_atoms(block)
        
        # 提取残基
        residues = self._extract_residues(block)
        
        # 提取配体
        ligands = self._extract_ligands(block, atoms, residues)
        
        # 提取金属离子
        metal_ions = self._extract_metal_ions(block, atoms)
        
        # 提取二级结构
        secondary_structures = self._extract_secondary_structures(block)
        
        # 计算结合位点
        binding_sites = self._calculate_binding_sites(
            block, atoms, residues, ligands, metal_ions
        )
        
        return TargetStructure(
            target_name=self.target_name,
            pdb_id=metadata['pdb_id'],
            entity_id='1',
            sequence=sequence_info['sequence'],
            sequence_length=sequence_info['length'],
            chains=sequence_info['chains'],
            atoms=atoms,
            atoms_count=len(atoms),
            residues=residues,
            residues_count=len(residues),
            binding_sites=binding_sites,
            ligands=ligands,
            metal_ions=metal_ions,
            secondary_structures=secondary_structures,
            resolution=metadata['resolution'],
            experimental_method=metadata['experimental_method'],
            title=metadata['title'],
            deposition_date=metadata['deposition_date'],
            gene_name=metadata.get('gene_name'),
            organism=metadata.get('organism'),
            mutation_info=metadata.get('mutation_info')
        )
    
    def _extract_metadata(self, block: Dict) -> Dict:
        """提取元数据"""
        underscore = block.get('_', {})
        
        pdb_id = underscore.get('struct.entry_id', 'XXXX')
        resolution = None
        if 'exptl.resolution' in underscore:
            try:
                resolution = float(underscore['exptl.resolution'])
            except:
                pass
        
        title = ''
        if 'struct.title' in underscore:
            title = underscore['struct.title']
        
        method = 'X-RAY DIFFRACTION'
        if 'exptl.method' in underscore:
            method = underscore['exptl.method']
        
        date = ''
        if 'struct_ref.seq_dif_one' in str(block):
            pass  # 简化处理
        
        # 尝试提取基因名称和物种
        gene_name = None
        organism = None
        mutation_info = None
        
        if 'entity' in block:
            for entity in block['entity']:
                desc = entity.get('pdbx_description', '')
                if 'epidermal growth factor receptor' in desc.lower():
                    gene_name = 'EGFR'
                    organism = 'Homo sapiens'
                elif 'vascular endothelial growth factor' in desc.lower():
                    gene_name = 'VEGFA'
                    organism = 'Homo sapiens'
                elif 'isocitrate dehydrogenase' in desc.lower():
                    gene_name = 'IDH1'
                    organism = 'Homo sapiens'
                elif 'p53' in desc.lower() or 'cellular tumor antigen p53' in desc.lower():
                    gene_name = 'TP53'
                    organism = 'Homo sapiens'
                elif 'mdm2' in desc.lower():
                    gene_name = 'MDM2'
                    organism = 'Homo sapiens'
                elif 'mgmt' in desc.lower():
                    gene_name = 'MGMT'
                    organism = 'Homo sapiens'
                elif 'programmed cell death protein 1' in desc.lower():
                    gene_name = 'PDCD1'
                    organism = 'Homo sapiens'
                elif 'phosphatidylinositol-4,5-bisphosphate 3-kinase' in desc.lower():
                    gene_name = 'PIK3CA'
                    organism = 'Homo sapiens'
        
        return {
            'pdb_id': pdb_id,
            'resolution': resolution,
            'title': title,
            'experimental_method': method,
            'deposition_date': date,
            'gene_name': gene_name,
            'organism': organism,
            'mutation_info': mutation_info
        }
    
    def _extract_sequence(self, block: Dict) -> Dict:
        """提取序列信息"""
        underscore = block.get('_', {})
        entity_poly_seq = block.get('entity_poly_seq', [])
        
        sequence = ''
        chains = ['A']
        
        # 方法1: 从entity_poly_seq构建序列
        if entity_poly_seq and isinstance(entity_poly_seq, list):
            aa_codes = []
            for row in entity_poly_seq:
                if isinstance(row, dict):
                    mon_id = row.get('_entity_poly_seq.mon_id', '')
                    if mon_id and mon_id in self.AA_3TO1:
                        aa_codes.append(self.AA_3TO1[mon_id])
                    elif mon_id:
                        aa_codes.append('X')  # 未知氨基酸
            sequence = ''.join(aa_codes)
        
        # 方法2: 检查_字段中的pdbx_seq_one_letter_code
        if not sequence or len(sequence) < 10:
            seq_key = '_entity_poly.pdbx_seq_one_letter_code'
            if seq_key in underscore:
                seq_val = underscore[seq_key]
                if seq_val and seq_val.strip():
                    # 处理分号包围的多行字符串
                    if ';' in seq_val:
                        # 提取分号之间的内容
                        seq_content = re.search(r';\s*([A-Z]+)\s*;', seq_val, re.DOTALL)
                        if seq_content:
                            sequence = seq_content.group(1).replace('\n', '')
                    elif len(seq_val) > 50:
                        sequence = seq_val.replace('\n', '').replace(' ', '')
        
        # 获取链信息
        strand_key = '_entity_poly.pdbx_strand_id'
        if strand_key in underscore:
            chains = [underscore[strand_key].strip()]
        
        return {
            'sequence': sequence,
            'length': len(sequence) if sequence else 0,
            'chains': chains if chains else ['A']
        }
    
    def _extract_atoms(self, block: Dict) -> List[AtomInfo]:
        """提取原子信息"""
        atom_sites = block.get('atom_site', [])
        atoms = []
        
        for i, site in enumerate(atom_sites):
            try:
                # 跳过非ATOM记录
                group_pdb = site.get('_atom_site.group_pdb', '')
                if group_pdb and group_pdb.strip() not in ['ATOM', 'HETATM']:
                    continue
                
                atom = AtomInfo(
                    id=i + 1,
                    element=site.get('_atom_site.type_symbol', 'X'),
                    x=float(site.get('_atom_site.Cartn_x', 0)),
                    y=float(site.get('_atom_site.Cartn_y', 0)),
                    z=float(site.get('_atom_site.Cartn_z', 0)),
                    residue_seq_num=int(site.get('_atom_site.label_seq_id', 0)),
                    residue_name=site.get('_atom_site.label_comp_id', 'XAA'),
                    chain_id=site.get('_atom_site.label_asym_id', 'A'),
                    atom_type=site.get('_atom_site.type_symbol', 'X')
                )
                atoms.append(atom)
            except (ValueError, KeyError):
                continue
        
        return atoms
    
    def _extract_residues(self, block: Dict) -> List[ResidueInfo]:
        """提取残基信息"""
        entity_poly_seq = block.get('entity_poly_seq', [])
        residues = []
        
        # 映射label_seq_id到auth_seq_id
        atom_sites = block.get('atom_site', [])
        seq_to_auth = {}
        for site in atom_sites:
            try:
                label_seq = int(site.get('_atom_site.label_seq_id', 0))
                auth_seq = int(site.get('_atom_site.auth_seq_id', label_seq))
                seq_to_auth[label_seq] = auth_seq
            except ValueError:
                continue
        
        for res in entity_poly_seq:
            try:
                if isinstance(res, dict):
                    seq_num = int(res.get('_entity_poly_seq.num', 0))
                    res_info = ResidueInfo(
                        seq_num=seq_num,
                        aa_type=res.get('_entity_poly_seq.mon_id', 'XAA'),
                        chain_id='A',
                        auth_seq_id=seq_to_auth.get(seq_num, seq_num),
                        auth_comp_id=res.get('_entity_poly_seq.mon_id', 'XAA')
                    )
                    residues.append(res_info)
            except (ValueError, KeyError):
                continue
        
        return residues
    
    def _extract_ligands(self, block: Dict, atoms: List[AtomInfo], 
                         residues: List[ResidueInfo]) -> List[LigandInfo]:
        """提取配体信息"""
        ligands = []
        
        # 提取非聚合物实体
        entity_nonpoly = block.get('pdbx_entity_nonpoly', [])
        
        # 配体名称映射
        ligand_comp_ids = set()
        for np in entity_nonpoly:
            comp_id = np.get('comp_id', '')
            name = np.get('name', '')
            ligand_comp_ids.add(comp_id)
        
        # 从原子数据中提取配体原子
        res_to_ligand = defaultdict(list)
        for atom in atoms:
            if atom.residue_name in ligand_comp_ids:
                res_to_ligand[atom.residue_name].append(atom)
        
        # 创建配体信息
        for comp_id, lig_atoms in res_to_ligand.items():
            if len(lig_atoms) < 50:  # 过滤掉大分子
                name = self.LIGAND_NAMES.get(comp_id, comp_id)
                
                # 计算分子量 (简化估算)
                mw = self._estimate_molecular_weight(comp_id, lig_atoms)
                
                ligand = LigandInfo(
                    comp_id=comp_id,
                    name=name,
                    formula='',
                    molecular_weight=mw,
                    atoms=lig_atoms,
                    binding_residues=[],
                    binding_site_id=None
                )
                ligands.append(ligand)
        
        return ligands
    
    def _extract_metal_ions(self, block: Dict, atoms: List[AtomInfo]) -> List[MetalIon]:
        """提取金属离子"""
        metal_ions = []
        
        metal_elements = ['ZN', 'MG', 'MN', 'FE', 'CU', 'CA', 'CO', 'NI', 'CD']
        res_to_atoms = defaultdict(list)
        
        for atom in atoms:
            if atom.residue_name in metal_elements:
                res_to_atoms[atom.residue_name].append(atom)
        
        for comp_id, ion_atoms in res_to_atoms.items():
            if len(ion_atoms) == 1:  # 金属离子通常是单个原子
                atom = ion_atoms[0]
                metal_ion = MetalIon(
                    comp_id=comp_id,
                    element=atom.element,
                    coordinates=(atom.x, atom.y, atom.z),
                    coordination_residues=[]
                )
                metal_ions.append(metal_ion)
        
        return metal_ions
    
    def _extract_secondary_structures(self, block: Dict) -> List[SecondaryStructure]:
        """提取二级结构"""
        structures = []
        
        # 从entity_poly_seq提取简化信息
        entity_poly_seq = block.get('entity_poly_seq', [])
        
        current_type = None
        start_seq = None
        
        for i, res in enumerate(entity_poly_seq):
            seq_num = int(res.get('num', 0))
            hetero = res.get('hetero', 'n')
            
            # 简化处理：基于残基类型推断
            if res.get('mon_id') in ['PRO']:
                if current_type != 'TURN' and start_seq is not None:
                    structures.append(SecondaryStructure(
                        type='TURN',
                        start_seq_num=start_seq,
                        end_seq_num=seq_num,
                        chain_id='A'
                    ))
                    start_seq = None
                current_type = None
        
        return structures
    
    def _calculate_binding_sites(self, block: Dict, atoms: List[AtomInfo],
                                  residues: List[ResidueInfo], 
                                  ligands: List[LigandInfo],
                                  metal_ions: List[MetalIon]) -> List[BindingSite]:
        """计算结合位点"""
        binding_sites = []
        
        # 方法1: 基于配体位置确定结合位点
        for i, ligand in enumerate(ligands):
            if ligand.atoms:
                # 计算配体周围的结合位点
                ligand_coords = [(a.x, a.y, a.z) for a in ligand.atoms]
                center = tuple(np.mean(ligand_coords, axis=0))
                
                # 找到附近的残基
                nearby_residues = self._find_nearby_residues(
                    atoms, center, cutoff=6.0
                )
                
                # 计算口袋性质
                properties = self._calculate_site_properties(
                    center, nearby_residues, residues
                )
                
                # 计算口袋体积
                radius = self._calculate_pocket_radius(atoms, center)
                volume = 4/3 * np.pi * radius**3
                
                site = BindingSite(
                    site_id=f"site_{i+1}_{ligand.comp_id.lower()}",
                    site_name=f"{ligand.name} binding site",
                    residues=nearby_residues,
                    center=center,
                    radius=radius,
                    volume=volume,
                    properties=properties,
                    atoms=ligand.atoms
                )
                binding_sites.append(site)
        
        # 方法2: 基于ATP/ADP结合位点预测
        atp_site = self._predict_atp_binding_site(atoms, residues)
        if atp_site:
            binding_sites.append(atp_site)
        
        # 方法3: 基于金属离子确定金属结合位点
        for i, metal in enumerate(metal_ions):
            metal_site = self._create_metal_binding_site(metal, atoms, residues)
            if metal_site:
                binding_sites.append(metal_site)
        
        # 方法4: 预测潜在的别构位点
        allosteric_sites = self._predict_allosteric_sites(atoms, residues)
        binding_sites.extend(allosteric_sites)
        
        return binding_sites
    
    def _find_nearby_residues(self, atoms: List[AtomInfo], 
                               center: Tuple[float, float, float],
                               cutoff: float = 6.0) -> List[Dict]:
        """找到中心点附近的残基"""
        nearby = []
        seen_residues = set()
        
        for atom in atoms:
            if atom.residue_name in ['HOH', 'WAT']:  # 跳过水分子
                continue
            
            dist = np.sqrt(
                (atom.x - center[0])**2 + 
                (atom.y - center[1])**2 + 
                (atom.z - center[2])**2
            )
            
            if dist < cutoff:
                res_key = (atom.residue_name, atom.residue_seq_num)
                if res_key not in seen_residues:
                    seen_residues.add(res_key)
                    nearby.append({
                        "aa_type": atom.residue_name,
                        "seq_num": atom.residue_seq_num,
                        "auth_comp_id": atom.residue_name,
                        "auth_seq_id": atom.residue_seq_num,
                        "distance": dist
                    })
        
        return nearby
    
    def _calculate_site_properties(self, center: Tuple, 
                                    nearby_residues: List[Dict],
                                    all_residues: List[ResidueInfo]) -> Dict:
        """计算结合位点的理化性质"""
        properties = {
            "hydrophobic_count": 0,
            "polar_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "aromatic_count": 0,
            "hbd_count": 0,
            "hba_count": 0
        }
        
        aa_categories = {
            'hydrophobic': ['ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO'],
            'polar': ['SER', 'THR', 'ASN', 'GLN'],
            'positive': ['LYS', 'ARG', 'HIS'],
            'negative': ['ASP', 'GLU'],
            'aromatic': ['PHE', 'TYR', 'TRP', 'HIS'],
            'hbd': ['SER', 'THR', 'TYR', 'CYS', 'LYS', 'ARG', 'HIS', 'TRP'],
            'hba': ['SER', 'THR', 'TYR', 'ASN', 'GLN', 'ASP', 'GLU', 'CYS']
        }
        
        for res in nearby_residues:
            aa = res['aa_type']
            if aa in aa_categories['hydrophobic']:
                properties['hydrophobic_count'] += 1
            if aa in aa_categories['polar']:
                properties['polar_count'] += 1
            if aa in aa_categories['positive']:
                properties['positive_count'] += 1
            if aa in aa_categories['negative']:
                properties['negative_count'] += 1
            if aa in aa_categories['aromatic']:
                properties['aromatic_count'] += 1
            if aa in aa_categories['hbd']:
                properties['hbd_count'] += 1
            if aa in aa_categories['hba']:
                properties['hba_count'] += 1
        
        # 口袋极性评分
        total = sum(properties.values())
        if total > 0:
            properties['polarity_ratio'] = (
                properties['polar_count'] + properties['hbd_count'] + properties['hba_count']
            ) / total
        else:
            properties['polarity_ratio'] = 0
        
        return properties
    
    def _calculate_pocket_radius(self, atoms: List[AtomInfo],
                                  center: Tuple) -> float:
        """计算口袋半径"""
        distances = []
        for atom in atoms:
            dist = np.sqrt(
                (atom.x - center[0])**2 + 
                (atom.y - center[1])**2 + 
                (atom.z - center[2])**2
            )
            distances.append(dist)
        
        if distances:
            return np.percentile(distances, 75)  # 使用75百分位数
        return 5.0
    
    def _predict_atp_binding_site(self, atoms: List[AtomInfo],
                                   residues: List[ResidueInfo]) -> Optional[BindingSite]:
        """预测ATP结合位点"""
        # 查找ATP相关配体或基于序列特征预测
        # 激酶的ATP结合位点通常包含特定的gatekeeper残基
        
        # 查找包含 'A' 残基的区域（典型ATP结合特征）
        gatekeeper_positions = {
            'EGFR': 855,   # T855
            'VEGFR2': 883, # T883
            'SRC': 338,    # T338
        }
        
        for gatekeeper_pos in gatekeeper_positions.values():
            gatekeeper_residues = [r for r in residues if r.auth_seq_id == gatekeeper_pos]
            if gatekeeper_residues:
                res = gatekeeper_residues[0]
                # 搜索附近的残基形成ATP口袋
                center = (0, 0, 0)  # 需要实际坐标
                
                # 基于残基序列位置估算
                nearby_residues = []
                for r in residues:
                    if 800 <= r.auth_seq_id <= 900:  # 激酶结构域典型范围
                        nearby_residues.append({
                            "aa_type": r.aa_type,
                            "seq_num": r.seq_num,
                            "auth_comp_id": r.auth_comp_id,
                            "auth_seq_id": r.auth_seq_id,
                            "distance": abs(r.auth_seq_id - gatekeeper_pos)
                        })
                
                if nearby_residues:
                    return BindingSite(
                        site_id="site_atp_binding",
                        site_name="ATP-binding pocket (kinase domain)",
                        residues=sorted(nearby_residues, key=lambda x: x['distance'])[:30],
                        center=center,
                        radius=8.0,
                        volume=268.0,
                        properties={"type": "ATP binding", "kinase_specific": True},
                        atoms=[]
                    )
        
        return None
    
    def _create_metal_binding_site(self, metal: MetalIon,
                                    atoms: List[AtomInfo],
                                    residues: List[ResidueInfo]) -> Optional[BindingSite]:
        """创建金属离子结合位点"""
        center = metal.coordinates
        
        # 找到配位的残基
        nearby_residues = self._find_nearby_residues(atoms, center, cutoff=3.5)
        
        if nearby_residues:
            # 识别常见的金属配位残基
            coordination_residues = ['CYS', 'HIS', 'ASP', 'GLU', 'MET']
            coord_res_names = [
                r['auth_comp_id'] + str(r['auth_seq_id']) 
                for r in nearby_residues 
                if r['aa_type'] in coordination_residues
            ]
            
            return BindingSite(
                site_id=f"site_metal_{metal.element.lower()}",
                site_name=f"{metal.element} metal ion binding site",
                residues=nearby_residues,
                center=center,
                radius=4.0,
                volume=67.0,
                properties={
                    "metal_type": metal.element,
                    "coordination_residues": coord_res_names
                },
                atoms=[a for a in atoms if a.residue_name == metal.comp_id]
            )
        
        return None
    
    def _predict_allosteric_sites(self, atoms: List[AtomInfo],
                                   residues: List[ResidueInfo]) -> List[BindingSite]:
        """预测潜在的别构位点"""
        allosteric_sites = []
        
        # 基于序列特征和结构分析预测别构位点
        # 激酶的别构位点通常远离ATP结合位点
        
        # 简化实现：基于表面口袋预测
        surface_residues = self._find_surface_residues(atoms, residues)
        
        if surface_residues:
            # 尝试识别已知的别构区域
            allosteric_site = BindingSite(
                site_id="site_allosteric_1",
                site_name="Predicted allosteric site 1",
                residues=surface_residues[:20],
                center=(0, 0, 0),
                radius=7.0,
                volume=343.0,
                properties={"type": "allosteric", "predicted": True},
                atoms=[]
            )
            allosteric_sites.append(allosteric_site)
        
        return allosteric_sites
    
    def _find_surface_residues(self, atoms: List[AtomInfo],
                                residues: List[ResidueInfo]) -> List[Dict]:
        """找到表面残基（简化实现）"""
        # 基于可及性估算
        surface = []
        seen = set()
        
        for res in residues:
            res_key = res.auth_seq_id
            if res_key in seen:
                continue
            seen.add(res_key)
            
            # 简化判断：末端残基和特定类型残基
            if res.aa_type in ['ARG', 'LYS', 'GLU', 'ASP']:
                surface.append({
                    "aa_type": res.aa_type,
                    "seq_num": res.seq_num,
                    "auth_comp_id": res.auth_comp_id,
                    "auth_seq_id": res.auth_seq_id,
                    "distance": 0
                })
        
        return surface
    
    def _estimate_molecular_weight(self, comp_id: str, atoms: List[AtomInfo]) -> float:
        """估算分子量"""
        if comp_id == 'ZN':
            return 65.38
        elif comp_id == 'MG':
            return 24.31
        elif comp_id == 'MN':
            return 54.94
        elif comp_id == 'FE':
            return 55.85
        
        # 基于原子类型估算
        atomic_weights = {
            'C': 12.01, 'N': 14.01, 'O': 16.00, 'S': 32.07,
            'H': 1.008, 'P': 30.97, 'F': 19.00, 'CL': 35.45,
            'BR': 79.90, 'I': 126.90
        }
        
        mw = 0.0
        for atom in atoms:
            mw += atomic_weights.get(atom.element, 12.01)
        
        return mw


# ============ 靶点处理器 ============

class TargetProcessor:
    """靶点信息处理器"""
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 靶点名称映射
        self.target_mapping = {
            'EGFR-assembly1.cif.gz': 'EGFR',
            'VEGF:VEGFR-assembly1.cif.gz': 'VEGFR',
            'IDH1-assembly1.cif.gz': 'IDH1',
            'MGMT-assembly1.cif.gz': 'MGMT',
            'PD-1:PD-L1-assembly1.cif.gz': 'PD1_PDL1',
            'PI3K:AKT:mTOR-assembly1.cif.gz': 'PI3K_AKT_mTOR',
            'p53:MDM2.cif.gz': 'TP53_MDM2',
            'MDM2-assembly1.cif.gz': 'MDM2'
        }
    
    def process_all(self) -> Dict:
        """处理所有靶点文件"""
        results = {
            "processed_targets": [],
            "summary": {},
            "timestamp": str(Path(__file__).stat().st_mtime)
        }
        
        cif_files = list(self.input_dir.glob("*.cif.gz"))
        
        for cif_file in cif_files:
            print(f"处理靶点文件: {cif_file.name}")
            
            try:
                target_name = self.target_mapping.get(cif_file.name, cif_file.stem)
                extractor = TargetStructureExtractor(str(cif_file), target_name)
                target_struct = extractor.extract()
                
                # 保存详细JSON
                output_path = self.output_dir / f"{target_name}_structure.json"
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(target_struct.to_dict(), f, indent=2, ensure_ascii=False)
                
                # 保存简化版本用于快速查询
                summary_path = self.output_dir / f"{target_name}_summary.json"
                self._save_summary(target_struct, summary_path)
                
                results["processed_targets"].append({
                    "target_name": target_name,
                    "pdb_id": target_struct.pdb_id,
                    "sequence_length": target_struct.sequence_length,
                    "atoms_count": target_struct.atoms_count,
                    "binding_sites_count": len(target_struct.binding_sites),
                    "ligands_count": len(target_struct.ligands),
                    "metal_ions_count": len(target_struct.metal_ions),
                    "output_file": str(output_path)
                })
                
                print(f"  ✓ 完成: {target_name} (序列长度: {target_struct.sequence_length})")
                
            except Exception as e:
                print(f"  ✗ 失败: {cif_file.name} - {str(e)}")
                results["processed_targets"].append({
                    "target_name": cif_file.name,
                    "error": str(e)
                })
        
        # 生成汇总报告
        self._generate_summary_report(results)
        
        return results
    
    def _save_summary(self, target: TargetStructure, output_path: Path):
        """保存简化版本"""
        summary = {
            "target_name": target.target_name,
            "pdb_id": target.pdb_id,
            "metadata": {
                "resolution": target.resolution,
                "experimental_method": target.experimental_method,
                "title": target.title,
                "gene_name": target.gene_name,
                "organism": target.organism
            },
            "sequence": {
                "sequence": target.sequence,
                "length": target.sequence_length,
                "chains": target.chains
            },
            "binding_sites": [
                {
                    "site_id": site.site_id,
                    "site_name": site.site_name,
                    "center": list(site.center),
                    "radius": site.radius,
                    "properties": site.properties,
                    "key_residues": [
                        r["auth_comp_id"] + str(r["auth_seq_id"]) 
                        for r in site.residues[:15]
                    ]
                }
                for site in target.binding_sites
            ],
            "pharmacophore_features": target.to_dict().get("pharmacophore_features", {})
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    def _generate_summary_report(self, results: Dict):
        """生成汇总报告"""
        report = {
            "report_title": "GBM靶点结构信息提取报告",
            "processing_date": "2026-01-29",
            "total_targets": len(results["processed_targets"]),
            "successful": sum(1 for t in results["processed_targets"] if "error" not in t),
            "failed": sum(1 for t in results["processed_targets"] if "error" in t),
            "targets_summary": [
                {
                    "name": t.get("target_name", t.get("target_name")),
                    "pdb_id": t.get("pdb_id"),
                    "sequence_length": t.get("sequence_length"),
                    "binding_sites": t.get("binding_sites_count", 0),
                    "ligands": t.get("ligands_count", 0),
                    "status": "success" if "error" not in t else "failed"
                }
                for t in results["processed_targets"]
            ],
            "model_ready": True
        }
        
        report_path = self.output_dir / "targets_summary_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n汇总报告已保存: {report_path}")
    
    def generate_model_input(self, target_name: str) -> Dict:
        """为特定靶点生成模型可直接使用的输入格式"""
        summary_path = self.output_dir / f"{target_name}_summary.json"
        
        if not summary_path.exists():
            raise FileNotFoundError(f"靶点 {target_name} 的摘要文件不存在")
        
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        # 转换为模型输入格式
        model_input = {
            "target_id": target_name,
            "pdb_id": summary.get("pdb_id"),
            "sequence": summary["sequence"]["sequence"],
            "binding_pockets": [
                {
                    "pocket_id": site["site_id"],
                    "pocket_name": site["site_name"],
                    "center_3d": site["center"],
                    "radius": site["radius"],
                    "key_residues": site["key_residues"],
                    "properties": site["properties"]
                }
                for site in summary.get("binding_sites", [])
            ],
            "pharmacophore": summary.get("pharmacophore_features", {})
        }
        
        return model_input


# ============ 主函数 ============

def main():
    """主函数"""
    input_dir = "/root/Llamole-main/gbm_project/data/gbm_targets"
    output_dir = "/root/Llamole-main/gbm_project/data/processed_targets"
    
    print("=" * 60)
    print("GBM靶点结构信息提取与处理流程")
    print("=" * 60)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)
    
    processor = TargetProcessor(input_dir, output_dir)
    results = processor.process_all()
    
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    
    # 演示：为EGFR生成模型输入
    try:
        model_input = processor.generate_model_input("EGFR")
        print(f"\nEGFR模型输入预览:")
        print(f"  序列长度: {len(model_input['sequence'])}")
        print(f"  结合口袋数: {len(model_input['binding_pockets'])}")
        print(f"  药效团特征: {list(model_input['pharmacophore'].keys())}")
    except Exception as e:
        print(f"生成模型输入失败: {e}")
    
    return results


if __name__ == "__main__":
    main()

