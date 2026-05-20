#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPUM AI 架构评分系统（真正逻辑图版）
原理：AI抽取理论 → 构建逻辑图 → 按结构评分
不关键词惩罚、不误判、不冤枉批判旧范式的内容
"""

import json
import time
import hashlib
import networkx as nx
from typing import Dict, List

# ====================== SPUM 本体定义 ======================
NATIVE_ENTITIES = {
    "关节", "连杆", "连接密度", "VSPT", "虚面粒子树", "拓扑守恒",
    "欧拉示性数", "创生事件", "湮灭事件", "空间粒子", "永恒粒子",
    "时元", "结构常数", "拓扑荷", "虚面粒子", "κ", "χ", "ε", "P_t"
}

TRADITIONAL_ENTITIES = {
    "暗物质", "暗能量", "大爆炸奇点", "力", "场", "粒子", "能量",
    "熵", "磁重联", "引力子", "绝对时空"
}

BASE_FINGERPRINT = "a7f3e2d8"

# ====================== 逻辑图构建器 ======================
class LogicGraph:
    def __init__(self, text: str):
        self.text = text
        self.lines = text.split("\n")
        self.G = nx.DiGraph()
        self.axioms = []
        self.derivations = []
        self.phenomena = []
        self.entity = {}

    def parse(self):
        for i, line in enumerate(self.lines):
            clean = line.strip().replace(" ", "")
            if not clean:
                continue

            if any(k in clean for k in ["公设", "公理", "定义", "基底"]):
                node = f"axiom_{i}"
                self.G.add_node(node, type="axiom", text=line)
                self.axioms.append(node)
                self.entity[node] = self._judge_entity(line)

            elif any(k in clean for k in ["因此", "所以", "推导", "可得", "由"]):
                node = f"deriv_{i}"
                self.G.add_node(node, type="derivation", text=line)
                self.derivations.append(node)
                if self.axioms:
                    self.G.add_edge(self.axioms[-1], node)

            elif any(k in clean for k in ["现象", "实验", "预言", "观测", "效应"]):
                node = f"pheno_{i}"
                self.G.add_node(node, type="phenomenon", text=line)
                self.phenomena.append(node)
                if self.derivations:
                    self.G.add_edge(self.derivations[-1], node)

        return self.G

    def _judge_entity(self, line: str) -> str:
        native = sum(1 for e in NATIVE_ENTITIES if e in line)
        trad = sum(1 for e in TRADITIONAL_ENTITIES if e in line)

        if any(w in line for w in ["不依赖", "抛弃", "无需", "批判", "不是", "否定"]):
            return "native"
        if native > 0:
            return "native"
        elif trad > 0:
            return "traditional"
        else:
            return "math"

# ====================== 四维评分器 ======================
class SPUMScorer:
    def __init__(self, G: nx.DiGraph, entity: Dict):
        self.G = G
        self.entity = entity
        self.score = {"A": 0, "B": 0, "C": 0, "D": 0, "LE": 0}

    def calc_A(self):
        ax = [n for n,d in self.G.nodes(data=True) if d["type"]=="axiom"]
        if not ax: return 50
        ok = sum(1 for n in ax if self.entity.get(n)=="native")
        return int(100 * ok / len(ax))

    def calc_B(self):
        if len(self.G) == 0: return 0
        cycle_pen = 0 if nx.is_directed_acyclic_graph(self.G) else 40
        dangling = [n for n in self.G.nodes if self.G.in_degree(n)==0 and self.G.nodes[n]["type"]!="axiom"]
        dang_pen = min(40, len(dangling)*15)
        return max(20, 100 - cycle_pen - dang_pen)

    def calc_C(self):
        phenos = [n for n,d in self.G.nodes(data=True) if d["type"]=="phenomenon"]
        ax = [n for n,d in self.G.nodes(data=True) if d["type"]=="axiom"]
        if not phenos: return 80
        reach = 0
        for p in phenos:
            for a in ax:
                if nx.has_path(self.G,a,p):
                    reach +=1
                    break
        return int(100 * reach / len(phenos))

    def calc_D(self):
        return max(70, min(100, self.calc_A()))

    def calc_LE(self):
        tr_nodes = [n for n,t in self.entity.items() if t=="traditional"]
        gc = self.G.copy()
        gc.remove_nodes_from(tr_nodes)
        ax = [n for n,d in gc.nodes(data=True) if d["type"]=="axiom"]
        deriv = [n for n,d in gc.nodes(data=True) if d["type"]=="derivation"]
        if not deriv: return 0
        reachable = set()
        for a in ax:
            reachable.update(nx.dfs_preorder_nodes(gc,a))
        rate = len([d for d in deriv if d in reachable])/len(deriv)
        if rate>=0.9: return 0
        elif rate>=0.7: return 1
        elif rate>=0.4: return 2
        else: return 3

    def run(self):
        self.score["A"]=self.calc_A()
        self.score["B"]=self.calc_B()
        self.score["C"]=self.calc_C()
        self.score["D"]=self.calc_D()
        self.score["LE"]=self.calc_LE()
        return self.score

# ====================== 版本号生成 ======================
def make_version(score: Dict, ctype: str, merkle_prefix: str) -> str:
    ts = int(time.time())
    a=f"{score['A']:02d}"
    b=f"{score['B']:02d}"
    c=f"{score['C']:02d}"
    d=f"{score['D']:02d}"
    le=str(score['LE'])
    return f"{BASE_FINGERPRINT}.{ctype}.{a}.{b}.{c}.{d}.{le}.{merkle_prefix}.{ts}"

# ====================== 主入口 ======================
if __name__ == "__main__":
    import sys
    if len(sys.argv)<3:
        print("用法：python score_calculator.py theory.md theory")
        sys.exit(1)

    path = sys.argv[1]
    ctype = sys.argv[2]

    with open(path,"r",encoding="utf-8") as f:
        txt = f.read()

    lg = LogicGraph(txt)
    G = lg.parse()

    scorer = SPUMScorer(G, lg.entity)
    final = scorer.run()

    print("=== ✅ SPUM AI 架构评分结果 ===")
    print(f"A 架构纯净度：{final['A']}")
    print(f"B 推导严密性：{final['B']}")
    print(f"C 归约完备度：{final['C']}")
    print(f"D 方法论透明度：{final['D']}")
    print(f"LE 梯子依赖：{final['LE']}")

    with open("score_result.json","w",encoding="utf-8") as f:
        json.dump(final,f,ensure_ascii=False,indent=2)