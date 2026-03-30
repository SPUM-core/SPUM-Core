#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPUM 四维评分与版本号生成脚本
符合规范：SPUM版本号生成与审核规范 v4.0
版本号格式：v5.0
"""
import re
import json
import hashlib
import time
from typing import Dict, Tuple, List

# 固定配置
BASE_FINGERPRINT = "a7f3e2d8"  # 核心公设固定指纹
FORBIDDEN_ENTITIES = [
    "引力子", "绝对时空", "原粒子", "暗物质", "暗能量", "大爆炸奇点",
    "力", "场", "粒子", "能量", "熵", "波函数", "磁重联"
]
NATIVE_ENTITIES = [
    "关节", "连杆", "连接密度", "VSPT", "虚面粒子树", "拓扑守恒",
    "欧拉示性数", "创生事件", "湮灭事件", "空间粒子", "永恒粒子",
    "时元", "结构常数", "拓扑荷", "虚面粒子"
]


class SPUMScorer:
    def __init__(self, md_content: str):
        self.content = md_content
        self.lines = md_content.split("\n")
        self.score = {
            "A": 100,  # 架构纯净度
            "B": 100,  # 推导严密性
            "C": 100,  # 归约完备度
            "D": 100,  # 方法论透明度
            "LE": 0    # 梯子依赖指数
        }
        self.deduction_details = {
            "A": [], "B": [], "C": [], "D": []
        }

    def _count_keywords(self, keywords: List[str]) -> Dict[str, int]:
        """统计关键词出现次数"""
        count = {}
        content_lower = self.content.lower()
        for kw in keywords:
            count[kw] = len(re.findall(re.escape(kw), content_lower))
        return count

    def calc_A_score(self) -> int:
        """计算架构纯净度A分"""
        score = 100
        forbidden_count = self._count_keywords(FORBIDDEN_ENTITIES)
        native_count = self._count_keywords(NATIVE_ENTITIES)

        # 1. 禁止实体扣分
        for entity, cnt in forbidden_count.items():
            if cnt > 0:
                if entity in ["暗物质", "绝对时空", "引力子"]:
                    score -= 30
                    self.deduction_details["A"].append(f"引入禁止基底实体[{entity}]，扣30分")
                    if score < 30:
                        score = 29
                else:
                    score -= 10 * min(cnt, 3)
                    self.deduction_details["A"].append(f"引入未归约传统概念[{entity}]，扣{10*min(cnt,3)}分")

        # 2. 公理冗余/循环扣分
        axiom_pattern = re.compile(r"公设\s*\d+|公理\s*\d+")
        axioms = axiom_pattern.findall(self.content)
        if len(axioms) != len(set(axioms)):
            score -= 20
            self.deduction_details["A"].append("公理存在循环/冗余，扣20分")

        # 3. 数学工具未声明扣分
        if "黎曼流形" in self.content and "近似" not in self.content:
            score -= 10
            self.deduction_details["A"].append("使用数学工具未声明工具属性，扣10分")

        # 分数边界限制
        score = max(0, min(100, score))
        self.score["A"] = score
        return score

    def calc_B_score(self) -> int:
        """计算推导严密性B分"""
        score = 100

        # 1. 循环定义扣分
        cycle_pattern = re.compile(r"(\w+)\s*[=:是]\s*.*?\1")
        cycles = cycle_pattern.findall(self.content)
        for cycle in cycles:
            if len(cycle) > 2:
                score -= 20
                self.deduction_details["B"].append(f"存在循环定义[{cycle}]，扣20分")

        # 2. 模糊表述扣分
        vague_words = ["显然", "众所周知", "偶然", "自然", "应该"]
        for word in vague_words:
            cnt = self.content.count(word)
            if cnt > 0:
                score -= 5 * min(cnt, 4)
                self.deduction_details["B"].append(f"使用模糊词[{word}]代替推导，扣{5*min(cnt,4)}分")

        # 3. 逻辑跳跃/悬空节点扣分
        conclusion_pattern = re.compile(r"因此|所以|故|结论是")
        conclusions = conclusion_pattern.findall(self.content)
        derivation_pattern = re.compile(r"推导|证明|由|根据|因为")
        derivations = derivation_pattern.findall(self.content)
        if len(conclusions) > len(derivations) + 2:
            score -= 30
            self.deduction_details["B"].append("存在无推导的悬空结论，扣30分")

        # 分数边界限制
        score = max(0, min(100, score))
        self.score["B"] = score
        return score

    def calc_C_score(self) -> int:
        """计算归约完备度C分"""
        score = 100

        # 1. 现象无推导路径扣分
        phenomenon_words = ["现象", "实验", "观测", "预言"]
        for word in phenomenon_words:
            lines_with_phenomenon = [line for line in self.lines if word in line]
            for line in lines_with_phenomenon:
                if not any(kw in line for kw in ["推导", "归约", "机制", "源于", "来自"]):
                    score -= 15
                    self.deduction_details["C"].append(f"现象[{line[:30]}]无归约路径，扣15分")
                    break

        # 2. 黑箱步骤扣分
        blackbox_words = ["结构产生意识", "涌现", "自发形成"]
        for word in blackbox_words:
            cnt = self.content.count(word)
            if cnt > 0 and "机制" not in self.content:
                score -= 20
                self.deduction_details["C"].append(f"存在黑箱步骤[{word}]，扣20分")

        # 3. 未归约传统概念扣分
        if "力" in self.content and "拓扑应力" not in self.content:
            score -= 10
            self.deduction_details["C"].append("使用未归约传统概念[力]，扣10分")

        # 分数边界限制
        score = max(0, min(100, score))
        self.score["C"] = score
        return score

    def calc_D_score(self) -> int:
        """计算方法论透明度D分"""
        score = 100

        # 1. 数学量与物理实体混淆扣分
        math_entities = ["曲率", "联络", "黎曼流形", "度规", "哈密顿量"]
        for entity in math_entities:
            cnt = self.content.count(entity)
            if cnt > 0 and "数学工具" not in self.content and "近似" not in self.content:
                score -= 20
                self.deduction_details["D"].append(f"将数学量[{entity}]作为物理实体，扣20分")

        # 2. 离散-连续过渡未声明扣分
        if "连续" in self.content and "离散" in self.content and "极限" not in self.content:
            score -= 10
            self.deduction_details["D"].append("未声明离散-连续过渡的近似性质，扣10分")

        # 3. 复杂数学形式未说明扣分
        if "$$" in self.content and "定义" not in self.content:
            score -= 10
            self.deduction_details["D"].append("使用复杂数学形式未说明描述性质，扣10分")

        # 分数边界限制
        score = max(0, min(100, score))
        self.score["D"] = score
        return score

    def calc_LE_index(self) -> int:
        """计算梯子依赖指数LE"""
        # 移除所有传统术语，检查逻辑是否完整
        content_clean = self.content
        for entity in FORBIDDEN_ENTITIES:
            content_clean = content_clean.replace(entity, "")

        # 检查核心逻辑是否完整
        core_logic_keywords = ["关节", "连杆", "拓扑", "公设", "推导"]
        core_cnt = sum(1 for kw in core_logic_keywords if kw in content_clean)

        if core_cnt >= 4:
            le = 0
        elif core_cnt >= 3:
            le = 1
        elif core_cnt >= 2:
            le = 2
        else:
            le = 3

        self.score["LE"] = le
        return le

    def calc_all_scores(self) -> Dict[str, int]:
        """计算所有维度分数"""
        self.calc_A_score()
        self.calc_B_score()
        self.calc_C_score()
        self.calc_D_score()
        self.calc_LE_index()
        return self.score

    def generate_version_number(self, content_type: str, merkle_prefix: str, timestamp: int = None) -> str:
        """生成合规版本号"""
        if timestamp is None:
            timestamp = int(time.time())
        # 分数补零为两位数
        a_str = f"{self.score['A']:02d}"
        b_str = f"{self.score['B']:02d}"
        c_str = f"{self.score['C']:02d}"
        d_str = f"{self.score['D']:02d}"
        le_str = f"{self.score['LE']}"

        version = f"{BASE_FINGERPRINT}.{content_type}.{a_str}.{b_str}.{c_str}.{d_str}.{le_str}.{merkle_prefix}.{timestamp}"
        return version

    def get_deduction_report(self) -> str:
        """生成扣分详情报告"""
        report = "=== SPUM 评分扣分详情报告 ===\n"
        for dim, details in self.deduction_details.items():
            dim_name = {
                "A": "架构纯净度",
                "B": "推导严密性",
                "C": "归约完备度",
                "D": "方法论透明度"
            }[dim]
            report += f"\n【{dim_name}({self.score[dim]}/100)】\n"
            if not details:
                report += "  无扣分，满分\n"
            else:
                for detail in details:
                    report += f"  - {detail}\n"
        report += f"\n【梯子依赖指数LE】：{self.score['LE']}\n"
        return report


if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) < 3:
        print("用法: python score_calculator.py <md文件路径> <内容类型>")
        print("内容类型可选: theory, exp, data, dial, code")
        sys.exit(1)

    md_path = sys.argv[1]
    content_type = sys.argv[2]

    if not os.path.exists(md_path):
        print(f"错误: 文件{md_path}不存在")
        sys.exit(1)

    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    # 计算分数
    scorer = SPUMScorer(md_content)
    scores = scorer.calc_all_scores()
    report = scorer.get_deduction_report()

    # 输出结果
    print(report)
    print("\n=== 最终评分 ===")
    print(f"架构纯净度A: {scores['A']}/100")
    print(f"推导严密性B: {scores['B']}/100")
    print(f"归约完备度C: {scores['C']}/100")
    print(f"方法论透明度D: {scores['D']}/100")
    print(f"梯子依赖指数LE: {scores['LE']}")

    # 写入临时结果文件，供CI读取
    with open("score_result.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)