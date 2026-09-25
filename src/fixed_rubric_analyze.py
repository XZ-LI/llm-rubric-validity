#!/usr/bin/env python3
"""
fixed_rubric_analyze.py — RQ4：哪些维度把效度拖回零？

前提（由 RQ6 确立）：评分表在**多样性**这一维上与读者是一致的
（评分表↔多样性 +0.217，多样性↔收藏率 +0.180），但总体效度只有 0.136。
问题因此收紧为：它在唯一对齐的维度之外，是哪些维度在抵消？

要回答这个，维度分数必须跨品类可比。原先 37 个格子各自归纳评分表，
产生 208 个维度名、145 个只出现一次，维度身份与格子身份共线，无法分析。
fixed_rubric_run.py 用同一套十维重打了全部 825 本，本脚本分析其产出。

三块：
  1. 逐维度 × 读者行为的相关（品类内 z 标准化 + 逐格符号检验）
  2. 固定十维总分 vs 原品类专属评分表总分，谁更能预测读者行为
     —— 顺带回答「品类定制到底有没有用」
  3. 维度间相关矩阵 + 主成分
     —— 若十维塌缩成一两个因子，则「十维」是虚的，评分表的信息量
        比表面小得多。这对「机器归纳评分表」这一做法本身是另一层批评。

用法：
  python3 fixed_rubric_analyze.py
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from validate_rubric import pearson, spearman  # noqa: E402

BASE = Path("review_loop")

# 先验分组，与数据无关：按维度关注的是"写得好不好"还是"有没有新意"
CRAFT = {"画面感", "对话自然度", "节奏变化", "人物鲜活度", "世界观融合"}
NOVELTY = {"钩子强度", "核心矛盾", "悬念设计", "情感感染力", "戏剧性反转"}


def min_detectable(n: int) -> float:
    if n < 4:
        return float("nan")
    df = n - 2
    tt = 1.96 + 2.6 / max(df, 1)
    return tt / math.sqrt(tt ** 2 + df)


def num(v):
    if isinstance(v, bool) or v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def safe_div(a, b):
    a, b = num(a), num(b)
    return a / b if (a is not None and b and b > 0) else None


def z(v):
    s = statistics.pstdev(v)
    return [(x - statistics.mean(v)) / s for x in v] if s else []


def sign_p(per):
    pos = sum(1 for r in per if r > 0)
    N = len(per)
    if not N:
        return 0, 0, float("nan")
    return pos, N, sum(math.comb(N, i) * 0.5 ** N for i in range(pos, N + 1))


def pooled(cells, xk, yk):
    zx, zy, per = [], [], []
    for rows in cells.values():
        v = [(r.get(xk), r.get(yk)) for r in rows]
        v = [(a, b) for a, b in v
             if isinstance(a, (int, float)) and isinstance(b, (int, float))]
        if len(v) < 8:
            continue
        a = [p[0] for p in v]
        b = [p[1] for p in v]
        za, zb = z(a), z(b)
        if za and zb:
            zx += za
            zy += zb
            per.append(pearson(a, b))
    if len(zx) < 10:
        return float("nan"), len(zx), per
    return pearson(zx, zy), len(zx), per


def load():
    fx = json.loads((BASE / "fixed_rubric.json").read_text(encoding="utf-8"))
    eng = json.loads((BASE / "engagement.json").read_text(encoding="utf-8"))["books"]
    dims = [r["name"] for r in fx["rubric"]]
    cells = defaultdict(list)
    for key, b in fx["books"].items():
        e = eng.get(key)
        if not e:
            continue
        ar = num(e.get("read_count_all"))
        row = {"fixed": b["fixed_score"], "orig": b["orig_score"],
               "rating": b["rating"],
               "收藏率": safe_div(e.get("all_bookshelf_count"), ar),
               "留存比": safe_div(e.get("read_count"), ar)}
        for d in dims:
            row[d] = b["dims"].get(d)
        cells[b["cell"]].append(row)
    return {k: v for k, v in cells.items() if len(v) >= 8}, dims


def main() -> None:
    cells, dims = load()
    n = sum(len(v) for v in cells.values())
    print(f"\n{'=' * 80}\n  RQ4 逐维度分析（固定十维，跨品类可比）\n{'=' * 80}")
    print(f"  {n} 本，{len(cells)} 个格子，可检出 |r| ≥ {min_detectable(n):.3f}")

    print(f"\n{'─' * 80}\n  1. 逐维度 × 收藏率（最干净的行为效标）\n{'─' * 80}")
    print(f"  {'维度':<12}{'类别':<6}{'r 收藏率':>10}{'r 留存比':>10}{'r 平台评分':>11}"
          f"{'逐格为正':>10}{'p':>9}")
    rows = []
    for d in dims:
        r_c, m, per = pooled(cells, d, "收藏率")
        r_l, _, _ = pooled(cells, d, "留存比")
        r_r, _, _ = pooled(cells, d, "rating")
        pos, N, p = sign_p(per)
        kind = "工艺" if d in CRAFT else "新颖" if d in NOVELTY else "—"
        rows.append((d, kind, r_c, r_l, r_r, pos, N, p))
    for d, kind, rc, rl, rr, pos, N, p in sorted(rows, key=lambda x: -x[2]):
        print(f"  {d:<12}{kind:<6}{rc:>+10.3f}{rl:>+10.3f}{rr:>+11.3f}"
              f"{f'{pos}/{N}':>10}{p:>9.4f}")

    craft = [r[2] for r in rows if r[1] == "工艺"]
    novel = [r[2] for r in rows if r[1] == "新颖"]
    if craft and novel:
        print(f"\n  工艺类 {len(craft)} 维 均值 r = {statistics.mean(craft):+.3f}")
        print(f"  新颖类 {len(novel)} 维 均值 r = {statistics.mean(novel):+.3f}")
        print(f"  差 {statistics.mean(craft) - statistics.mean(novel):+.3f}")

    print(f"\n{'─' * 80}\n  2. 固定十维 vs 品类专属评分表\n{'─' * 80}")
    print(f"  {'总分来源':<22}{'r 收藏率':>10}{'r 留存比':>10}{'r 平台评分':>11}{'n':>7}")
    for key, lab in (("fixed", "固定十维（本次重打）"), ("orig", "品类专属（原评分表）")):
        rc, m, _ = pooled(cells, key, "收藏率")
        rl, _, _ = pooled(cells, key, "留存比")
        rr, _, _ = pooled(cells, key, "rating")
        print(f"  {lab:<22}{rc:>+10.3f}{rl:>+10.3f}{rr:>+11.3f}{m:>7}")
    r_ff, m_ff, _ = pooled(cells, "fixed", "orig")
    print(f"\n  两套总分彼此 r = {r_ff:+.3f}（n={m_ff}）")
    print("  判读：若品类专属并不更准，则『按品类定制评分表』这一步没有带来收益。")

    print(f"\n{'─' * 80}\n  3. 维度间结构：十维是真的十维吗\n{'─' * 80}")
    mat = []
    for rows_ in cells.values():
        usable = [r for r in rows_ if all(isinstance(r.get(d), (int, float)) for d in dims)]
        if len(usable) < 8:
            continue
        cols = [z([r[d] for r in usable]) for d in dims]
        if all(cols):
            mat.append(np.array(cols))
    if mat:
        X = np.hstack(mat)                      # 维度 × 样本
        C = np.corrcoef(X)
        iu = np.triu_indices(len(dims), 1)
        print(f"  维度间相关：中位 {np.median(C[iu]):.3f}　"
              f"区间 {C[iu].min():.3f}–{C[iu].max():.3f}")
        ev = np.linalg.eigvalsh(C)[::-1]
        tot = ev.sum()
        print(f"  主成分方差占比：PC1 {ev[0] / tot * 100:.1f}%　"
              f"PC1+PC2 {(ev[0] + ev[1]) / tot * 100:.1f}%　"
              f"特征值>1 的成分数 {int((ev > 1).sum())}")
        print("\n  判读：PC1 占比越高，十维越是同一个潜在因子的十种说法；")
        print("        特征值>1 的成分数才是有效维度数。")

    out = BASE / "fixed_rubric_analysis.json"
    out.write_text(json.dumps(
        {"n": n, "cells": len(cells),
         "dims": [{"name": d, "kind": k, "r_收藏率": rc, "r_留存比": rl,
                   "r_评分": rr, "pos": f"{pos}/{N}", "p": p}
                  for d, k, rc, rl, rr, pos, N, p in rows]},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  已保存：{out}")


if __name__ == "__main__":
    main()
