#!/usr/bin/env python3
"""
改进的SMILES提取函数
支持多种SMILES格式和模型输出模式
"""

import re
from rdkit import Chem
from typing import Optional, List, Tuple


def extract_smiles_from_text(text: str) -> Optional[str]:
    """
    从文本中提取SMILES字符串

    支持的格式：
    1. SMILES: CCO
    2. smiles: CCO
    3. 分子式: CCO
    4. 直接的SMILES字符串 (没有标签)
    5. 包含在句子中的SMILES
    """

    # 方法1: 查找明确的标签
    label_patterns = [
        r'SMILES:\s*([A-Za-z0-9@+\-\[\]\(\)=#]+)',
        r'smiles:\s*([A-Za-z0-9@+\-\[\]\(\)=#]+)',
        r'分子式[:：]\s*([A-Za-z0-9@+\-\[\]\(\)=#]+)',
        r'分子结构[:：]\s*([A-Za-z0-9@+\-\[\]\(\)=#]+)',
        r'SMILES字符串[:：]\s*([A-Za-z0-9@+\-\[\]\(\)=#]+)',
    ]

    for pattern in label_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            for match in matches:
                cleaned_smiles = clean_smiles_string(match)
                if validate_smiles(cleaned_smiles):
                    return cleaned_smiles

    # 方法2: 查找可能的SMILES字符串（通用模式）
    # SMILES的基本特征：包含字母、数字、特殊字符，但不含空格
    general_patterns = [
        r'\b[A-Za-z][A-Za-z0-9@+\-\[\]\(\)=#]+\b',  # 以字母开头
        r'\b[0-9][A-Za-z0-9@+\-\[\]\(\)=#]+\b',     # 以数字开头
    ]

    candidates = []
    for pattern in general_patterns:
        matches = re.findall(pattern, text)
        candidates.extend(matches)

    # 过滤和验证候选SMILES
    for candidate in candidates:
        # 跳过太短或太长的字符串
        if len(candidate) < 3 or len(candidate) > 100:
            continue

        # 跳过包含中文或其他非SMILES字符的字符串
        if re.search(r'[^\x00-\x7F]', candidate):  # 非ASCII字符
            continue

        cleaned_smiles = clean_smiles_string(candidate)
        if validate_smiles(cleaned_smiles):
            return cleaned_smiles

    return None


def clean_smiles_string(smiles: str) -> str:
    """清理SMILES字符串"""
    if not smiles:
        return ""

    # 移除首尾空白字符
    smiles = smiles.strip()

    # 移除末尾的标点符号
    smiles = re.sub(r'[.,;:!?\s]+$', '', smiles)

    # 移除开头的标点符号
    smiles = re.sub(r'^[.,;:!?\s]+', '', smiles)

    return smiles


def validate_smiles(smiles: str) -> bool:
    """验证SMILES字符串的有效性"""
    if not smiles or len(smiles) < 2:
        return False

    try:
        # 使用RDKit验证
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False

        # 检查基本属性
        num_atoms = mol.GetNumAtoms()
        if num_atoms < 3 or num_atoms > 100:  # 合理的分子大小
            return False

        return True

    except Exception:
        return False


def extract_multiple_smiles(text: str, max_count: int = 5) -> List[str]:
    """从文本中提取多个SMILES字符串"""
    smiles_list = []

    # 先尝试提取带标签的SMILES
    label_patterns = [
        r'SMILES:\s*([A-Za-z0-9@+\-\[\]\(\)=#]+)',
        r'smiles:\s*([A-Za-z0-9@+\-\[\]\(\)=#]+)',
    ]

    for pattern in label_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            cleaned = clean_smiles_string(match)
            if validate_smiles(cleaned) and cleaned not in smiles_list:
                smiles_list.append(cleaned)
                if len(smiles_list) >= max_count:
                    return smiles_list

    # 如果没找到标签SMILES，尝试通用提取
    if not smiles_list:
        general_pattern = r'\b[A-Za-z0-9@+\-\[\]\(\)=#]{3,50}\b'
        matches = re.findall(general_pattern, text)

        for match in matches:
            cleaned = clean_smiles_string(match)
            if validate_smiles(cleaned) and cleaned not in smiles_list:
                smiles_list.append(cleaned)
                if len(smiles_list) >= max_count:
                    break

    return smiles_list


def debug_smiles_extraction(text: str) -> dict:
    """调试SMILES提取过程，返回详细信息"""
    result = {
        'original_text': text,
        'extracted_smiles': None,
        'all_candidates': [],
        'validation_results': [],
        'extraction_method': None
    }

    # 尝试标准提取
    smiles = extract_smiles_from_text(text)
    result['extracted_smiles'] = smiles

    # 收集所有候选
    patterns = [
        r'SMILES:\s*([A-Za-z0-9@+\-\[\]\(\)=#]+)',
        r'smiles:\s*([A-Za-z0-9@+\-\[\]\(\)=#]+)',
        r'\b[A-Za-z][A-Za-z0-9@+\-\[\]\(\)=#]+\b',
        r'\b[0-9][A-Za-z0-9@+\-\[\]\(\)=#]+\b',
    ]

    for i, pattern in enumerate(patterns):
        matches = re.findall(pattern, text)
        for match in matches:
            cleaned = clean_smiles_string(match)
            is_valid = validate_smiles(cleaned)

            candidate_info = {
                'pattern_id': i,
                'original_match': match,
                'cleaned_smiles': cleaned,
                'is_valid': is_valid
            }

            result['all_candidates'].append(candidate_info)
            result['validation_results'].append(is_valid)

            if is_valid and result['extracted_smiles'] is None:
                result['extraction_method'] = f'pattern_{i}'

    return result


# 测试函数
if __name__ == "__main__":
    # 测试样例
    test_texts = [
        "请设计分子 SMILES: CCO",
        "分子结构：CC(=O)O",
        "这是一个药物分子 C1CCCCC1",
        "SMILES字符串：CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "没有SMILES的文本",
        "多个分子：CCO 和 aspirin C1=CC=CC=C1C(=O)O"
    ]

    print("🧪 SMILES提取功能测试")
    print("=" * 50)

    for i, text in enumerate(test_texts, 1):
        print(f"\n测试 {i}: {text}")

        # 标准提取
        smiles = extract_smiles_from_text(text)
        print(f"提取的SMILES: {smiles}")

        # 多SMILES提取
        multiple_smiles = extract_multiple_smiles(text)
        if len(multiple_smiles) > 1:
            print(f"多个SMILES: {multiple_smiles}")

        # 验证
        if smiles:
            is_valid = validate_smiles(smiles)
            print(f"验证结果: {'✓ 有效' if is_valid else '✗ 无效'}")
