#!/usr/bin/env python3
"""
residual_criterion.py — 路径 D：把效标洗干净再测一次

问题：番茄读者评分里混了大量非质量成分——篇幅 r=+0.21、在读人数 r=+0.27、
完结比连载高 0.35 分。拿它当效标，只能证明"两把尺子不一致"，
不能证明"哪把量到了文本质量"。

做法：在每个品类内部，把评分对 log(章节数)、log(字数)、log(在读)、完结状态
做多元回归，取**残差**作为新效标。残差的含义是：
「这本书的评分，比它的篇幅与曝光所能解释的高出/低出多少」
——即剥掉规模与热度之后，读者额外给出的认可。

然后拿评分表分数去打这个更干净的靶子。

⚠ 过度控制的风险：如果文本质量本身也驱动了曝光（好书更多人看），
   残差化会把真实的质量信号一并剥掉。所以本脚本同时报告
   原始效标与残差效标两个结果，并报告协变量解释了多少方差。

用法：
  python3 residual_criterion.py
"""
from __future__ import annotations

import collections
import json
import math
import re
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from covariates import header_fields  # noqa: E402
from validate_rubric import pearson, spearman  # noqa: E402

BASE = Path("review_loop")


def min_detectable(n: int) -> float:
    if n < 4:
        return float("nan")
    df = n - 2
    tt = 1.96 + 2.6 / max(df, 1)
    return tt / math.sqrt(tt ** 2 + df)


def load_cells() -> dict[str, list[dict]]:
    cells: dict[str, list[dict]] = {}
    for p in sorted(BASE.glob("*/validation_pilot.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("reviewer_model") != "qwen3-max":
            continue
        src = d.get("source_dir")
        rows = []
        for r in d["results"]:
            f = Path(src) / d["category"] / r["file"]
            if not f.exists():
                continue
            h = header_fields(f)
            if not all(h.get(k) for k in ("chapters", "words", "readers")):
                continue
            rows.append({
                "title": r.get("title", ""), "rating": r["real_rating"],
                "score": r["rubric_score"], "dims": r.get("dimensions") or {},
                "chapters": h["chapters"], "words": h["words"],
                "readers": h["readers"], "status": h.get("status"),
            })
        if len(rows) >= 10:
            cells[p.parent.name] = rows
    return cells


def residualize(rows: list[dict]) -> tuple[list[float], float]:
    """把 rating 对协变量回归，返回残差与模型 R²。
    计数/曝光类变量取对数——在读人数跨三个数量级，线性尺度会被巨值主导。"""
    y = np.array([r["rating"] for r in rows], dtype=float)
    X = np.column_stack([
        np.ones(len(rows)),
        np.log([r["chapters"] for r in rows]),
        np.log([r["words"] for r in rows]),
        np.log([r["readers"] for r in rows]),
        [1.0 if r.get("status") == "完结" else 0.0 for r in rows],
    ])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    resid = y - pred
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float((resid ** 2).sum()) / ss_tot if ss_tot else float("nan")
    return resid.tolist(), r2


def z(v: list[float]) -> list[float]:
    s = statistics.pstdev(v)
    if not s:
        return []
    m = statistics.mean(v)
    return [(x - m) / s for x in v]


def main() -> None:
    cells = load_cells()
    print(f"\n{'=' * 72}\n  路径 D：残差效标\n{'=' * 72}")
    print(f"  {len(cells)} 个格子，{sum(len(v) for v in cells.values())} 本"
          f"（需同时有章节/字数/在读三项，故略少于 840）")

    r2s = []
    pool_raw_s, pool_raw_t, pool_res_s, pool_res_t = [], [], [], []
    per_cell = []
    for cell, rows in cells.items():
        resid, r2 = residualize(rows)
        r2s.append(r2)
        sc = [r["score"] for r in rows]
        rt = [r["rating"] for r in rows]
        zs, zt, zr = z(sc), z(rt), z(resid)
        if not (zs and zt and zr):
            continue
        pool_raw_s += zs; pool_raw_t += zt
        pool_res_s += zs; pool_res_t += zr
        per_cell.append((cell, len(rows), pearson(sc, rt), pearson(sc, resid), r2))

    print(f"\n  协变量（log 章节 + log 字数 + log 在读 + 完结）解释评分方差：")
    print(f"    中位 R² = {statistics.median(r2s) * 100:.1f}%   均值 {statistics.mean(r2s) * 100:.1f}%"
          f"   区间 {min(r2s) * 100:.1f}%–{max(r2s) * 100:.1f}%")

    n = len(pool_raw_s)
    print(f"\n{'─' * 72}\n  合并结果（品类内 z 标准化）  n={n}"
          f"   可检出 |r| ≥ {min_detectable(n):.3f}\n{'─' * 72}")
    print(f"  评分表 ↔ 原始评分     r = {pearson(pool_raw_s, pool_raw_t):+.4f}"
          f"   ρ = {spearman(pool_raw_s, pool_raw_t):+.4f}")
    print(f"  评分表 ↔ 残差评分     r = {pearson(pool_res_s, pool_res_t):+.4f}"
          f"   ρ = {spearman(pool_res_s, pool_res_t):+.4f}   ← 洗掉规模与热度后")

    per_cell.sort(key=lambda x: -x[3])
    print(f"\n{'─' * 72}\n  分格子（按残差相关排序，仅列两端）\n{'─' * 72}")
    print(f"  {'格子':<20}{'n':>4}{'原始r':>9}{'残差r':>9}{'协变量R²':>10}")
    for c, k, ra, rr, r2 in per_cell[:6]:
        print(f"  {c:<20}{k:>4}{ra:>+9.3f}{rr:>+9.3f}{r2 * 100:>9.0f}%")
    print(f"  {'…':<20}")
    for c, k, ra, rr, r2 in per_cell[-6:]:
        print(f"  {c:<20}{k:>4}{ra:>+9.3f}{rr:>+9.3f}{r2 * 100:>9.0f}%")

    pos = sum(1 for _, _, _, rr, _ in per_cell if rr > 0)
    N = len(per_cell)
    pv = sum(math.comb(N, i) * 0.5 ** N for i in range(pos, N + 1))
    print(f"\n  符号检验（残差效标）：{pos}/{N} 个格子为正，单尾 p = {pv:.4f}")

    print(f"\n{'─' * 72}\n  维度级：对残差效标\n{'─' * 72}")
    dim: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    for cell, rows in cells.items():
        resid, _ = residualize(rows)
        for r, e in zip(rows, resid):
            for k, v in (r["dims"] or {}).items():
                if isinstance(v, (int, float)):
                    dim[k].append((v, e))
    scored = []
    for k, pairs in dim.items():
        if len(pairs) >= 40:
            xs, ys = zip(*pairs)
            scored.append((pearson(list(xs), list(ys)), k, len(pairs)))
    scored.sort(reverse=True)
    for r_val, k, cnt in scored[:6]:
        print(f"    {k:<10} n={cnt:<5} r = {r_val:+.3f}  {'█' * max(0, round(abs(r_val) * 40))}")
    print(f"    {'…':<10}")
    for r_val, k, cnt in scored[-4:]:
        print(f"    {k:<10} n={cnt:<5} r = {r_val:+.3f}  {'█' * max(0, round(abs(r_val) * 40))}")

    print(f"\n{'=' * 72}")
    print("  注：残差化可能过度控制——若文本质量本身也驱动曝光，")
    print("      这一步会连真实质量信号一起剥掉。故上面同时给出原始与残差两个值；")
    print("      两者差异本身就是信息。")


if __name__ == "__main__":
    main()
