#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPUM Merkle树哈希验证脚本
功能：生成版本目录Merkle根、校验内容完整性
"""
import os
import hashlib
import json
from typing import List, Dict


def sha256_file(file_path: str) -> str:
    """计算单个文件的SHA256哈希"""
    hash_sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        print(f"文件哈希计算失败: {file_path}, 错误: {e}")
        return ""


def get_all_files(dir_path: str) -> List[str]:
    """递归获取目录下所有文件，按字典序排序"""
    file_list = []
    for root, dirs, files in os.walk(dir_path):
        # 按字典序排序，保证Merkle树生成稳定
        dirs.sort()
        files.sort()
        for file in files:
            if file == ".keep":
                continue
            file_path = os.path.join(root, file)
            file_list.append(file_path)
    return file_list


def build_merkle_tree(file_hashes: List[str]) -> str:
    """构建Merkle树，返回根哈希"""
    if not file_hashes:
        return hashlib.sha256(b"").hexdigest()

    # 叶子节点列表
    nodes = file_hashes.copy()

    # 逐级哈希，直到只剩根节点
    while len(nodes) > 1:
        # 奇数个节点时，复制最后一个节点
        if len(nodes) % 2 != 0:
            nodes.append(nodes[-1])
        # 两两配对哈希
        new_nodes = []
        for i in range(0, len(nodes), 2):
            combined = nodes[i] + nodes[i+1]
            new_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()
            new_nodes.append(new_hash)
        nodes = new_nodes

    return nodes[0]


def generate_merkle_root(dir_path: str) -> Dict[str, str]:
    """生成目录的Merkle根，返回文件哈希列表与根哈希"""
    if not os.path.isdir(dir_path):
        raise ValueError(f"路径{dir_path}不是有效目录")

    file_list = get_all_files(dir_path)
    file_hashes = []
    file_hash_map = {}

    for file_path in file_list:
        file_hash = sha256_file(file_path)
        file_hashes.append(file_hash)
        # 保存相对路径
        rel_path = os.path.relpath(file_path, dir_path)
        file_hash_map[rel_path] = file_hash

    merkle_root = build_merkle_tree(file_hashes)
    return {
        "merkle_root": merkle_root,
        "merkle_prefix": merkle_root[:12],
        "file_hash_map": file_hash_map
    }


def validate_directory(dir_path: str, expected_merkle_root: str) -> bool:
    """校验目录内容完整性"""
    result = generate_merkle_root(dir_path)
    actual_root = result["merkle_root"]
    is_valid = (actual_root == expected_merkle_root)

    if is_valid:
        print(f"✅ 内容完整性校验通过，Merkle根匹配: {actual_root}")
    else:
        print(f"❌ 内容完整性校验失败！")
        print(f"期望根哈希: {expected_merkle_root}")
        print(f"实际根哈希: {actual_root}")
    return is_valid


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法:")
        print("1. 生成Merkle根: python hash_validator.py generate <版本目录路径>")
        print("2. 校验完整性: python hash_validator.py validate <版本目录路径> <期望Merkle根>")
        sys.exit(1)

    action = sys.argv[1]

    if action == "generate":
        if len(sys.argv) < 3:
            print("请指定版本目录路径")
            sys.exit(1)
        dir_path = sys.argv[2]
        result = generate_merkle_root(dir_path)
        print(f"Merkle根: {result['merkle_root']}")
        print(f"Merkle前缀(版本号用): {result['merkle_prefix']}")
        # 写入merkle_result.json供CI读取
        with open("merkle_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    elif action == "validate":
        if len(sys.argv) < 4:
            print("请指定版本目录路径和期望Merkle根")
            sys.exit(1)
        dir_path = sys.argv[2]
        expected_root = sys.argv[3]
        validate_directory(dir_path, expected_root)

    else:
        print("无效操作，仅支持generate/validate")
        sys.exit(1)