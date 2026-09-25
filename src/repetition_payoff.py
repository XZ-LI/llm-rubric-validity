#!/usr/bin/env python3
"""
repetition_payoff.py — RQ6：读者是否偏好重复，而评分表奖励多样？

由 RQ3 的发现引出：AI 产出的用词多样性比真人出版小说高 +0.165（七品类无一例外），
而评分表恰恰奖励这类"教科书意义上更好"的文本。若读者实际偏好的是重复
——连载网文赖以运转的口头禅、固定爽点节拍、熟悉句式——那评分表的近零效度
就有了机制解释，而不只是一个相关系数。

要检验的是一条链，两环都必须成立：

    ① 评分表分数  ↔ 文本多样性     预期为正（评分表奖励多样）
    ② 文本多样性  ↔ 收藏率等行为   预期为负（读者偏好重复）

两环方向相反，即构成"评分表所奖励的，正是读者所回避的"。
若 ② 不成立（多样性与行为无关或为正），则该解释被否定——同样是结论。

全部使用留出集里的**真人小说**，不掺 AI 文本：这检验的是读者偏好，
不是人机对比。样本为已取得行为数据的 797 部。

用法：
  python3 repetition_payoff.py --limit 120    # 先小样本看方向
  python3 repetition_payoff.py
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from diversity import SAMPLE_TOKENS, distinct_n, mattr, tokens  # noqa: E402
from validate_rubric import pearson, spearman  # noqa: E402

BASE = Path("review_loop")
CHARS = 14000     # 先截断再分词：2500 词约需 5000 汉字，留足余量


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


def measure(job: tuple) -> dict | None:
    """单本书的多样性度量。放在模块层，供多进程调用。"""
    cell, path_s, meta = job
    p = Path(path_s)
    if not p.exists():
        return None
    body = p.read_text(encoding="utf-8", errors="replace").split("=" * 20, 1)[-1]
    t = tokens(body[:CHARS])
    if len(t) < SAMPLE_TOKENS:
        return None
    t = t[:SAMPLE_TOKENS]
    return {"cell": cell, **meta,
            "d1": distinct_n(t, 1), "d2": distinct_n(t, 2), "mattr": mattr(t)}


def build_jobs(limit: int = 0) -> list[tuple]:
    eng = json.loads((BASE / "engagement.json").read_text(encoding="utf-8"))["books"]
    jobs = []
    for p in sorted(BASE.glob("*/validation_pilot.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("reviewer_model") != "qwen3-max":
            continue
        for r in d["results"]:
            e = eng.get(f"{p.parent.name}|{r['file']}")
            if not e:
                continue
            all_read = num(e.get("read_count_all"))
            jobs.append((p.parent.name,
                         str(Path(d["source_dir"]) / d["category"] / r["file"]),
                         {"score": r["rubric_score"], "rating": r["real_rating"],
                          "收藏率": safe_div(e.get("all_bookshelf_count"), all_read),
                          "留存比": safe_div(e.get("read_count"), all_read),
                          "听书率": safe_div(e.get("listen_count"), all_read),
                          "words": num(e.get("word_number"))}))
    return jobs[:limit] if limit else jobs


def z(v):
    s = statistics.pstdev(v)
    return [(x - statistics.mean(v)) / s for x in v] if s else []


def pooled_corr(cells: dict, xk: str, yk: str) -> tuple[float, float, int, list[float]]:
    zx, zy, per = [], [], []
    for rows in cells.values():
        v = [(r[xk], r[yk]) for r in rows
             if isinstance(r.get(xk), (int, float)) and isinstance(r.get(yk), (int, float))]
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
        return float("nan"), float("nan"), len(zx), per
    return pearson(zx, zy), spearman(zx, zy), len(zx), per


def sign_p(per: list[float]) -> tuple[int, int, float]:
    pos = sum(1 for r in per if r > 0)
    N = len(per)
    if not N:
        return 0, 0, float("nan")
    p = sum(math.comb(N, i) * 0.5 ** N for i in range(pos, N + 1))
    return pos, N, p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--procs", type=int, default=4)
    args = ap.parse_args()

    jobs = build_jobs(args.limit)
    print(f"\n{'=' * 76}\n  RQ6 重复的回报：读者偏好重复，而评分表奖励多样？\n{'=' * 76}")
    print(f"  真人小说 {len(jobs)} 部（已有行为数据），每部取前 {SAMPLE_TOKENS} 词")
    print(f"  分词中（{args.procs} 进程）…", flush=True)

    with Pool(args.procs) as pool:
        got = [r for r in pool.map(measure, jobs, chunksize=8) if r]
    cells = defaultdict(list)
    for r in got:
        cells[r["cell"]].append(r)
    cells = {k: v for k, v in cells.items() if len(v) >= 8}
    n = sum(len(v) for v in cells.values())
    print(f"  完成 {n} 部，覆盖 {len(cells)} 个格子\n")

    print(f"{'─' * 76}\n  第 ① 环：评分表分数 ↔ 文本多样性（预期为正）\n{'─' * 76}")
    print(f"  {'多样性度量':<14}{'n':>6}{'r':>9}{'ρ':>9}{'可检出':>9}{'逐格为正':>11}{'p':>10}")
    link1 = {}
    for key, lab in (("d1", "distinct-1"), ("d2", "distinct-2"), ("mattr", "MATTR")):
        r, rho, m, per = pooled_corr(cells, "score", key)
        pos, N, p = sign_p(per)
        link1[key] = r
        print(f"  {lab:<14}{m:>6}{r:>+9.3f}{rho:>+9.3f}{min_detectable(m):>9.3f}"
              f"{f'{pos}/{N}':>11}{p:>10.4f}")

    print(f"\n{'─' * 76}\n  第 ② 环：文本多样性 ↔ 读者行为（预期为负）\n{'─' * 76}")
    print(f"  {'多样性':<12}{'行为效标':<10}{'n':>6}{'r':>9}{'ρ':>9}{'可检出':>9}{'逐格为正':>11}{'p':>10}")
    link2 = {}
    for key, lab in (("d1", "distinct-1"), ("mattr", "MATTR")):
        for crit in ("收藏率", "留存比", "听书率"):
            r, rho, m, per = pooled_corr(cells, key, crit)
            pos, N, p = sign_p(per)
            link2[(key, crit)] = r
            print(f"  {lab:<12}{crit:<10}{m:>6}{r:>+9.3f}{rho:>+9.3f}"
                  f"{min_detectable(m):>9.3f}{f'{pos}/{N}':>11}{p:>10.4f}")

    print(f"\n{'─' * 76}\n  对照：多样性 ↔ 平台评分（评分已被证明与行为脱节）\n{'─' * 76}")
    for key, lab in (("d1", "distinct-1"), ("mattr", "MATTR")):
        r, rho, m, per = pooled_corr(cells, key, "rating")
        pos, N, p = sign_p(per)
        print(f"  {lab:<12}{'评分':<10}{m:>6}{r:>+9.3f}{rho:>+9.3f}"
              f"{min_detectable(m):>9.3f}{f'{pos}/{N}':>11}{p:>10.4f}")

    print(f"\n{'=' * 76}\n  判读\n{'=' * 76}")
    a = link1.get("d1", float("nan"))
    b = link2.get(("d1", "收藏率"), float("nan"))
    floor = min_detectable(n)
    ok1 = a == a and a > floor
    ok2_neg = b == b and b < -floor        # 假设方向：多样性↑ → 行为↓
    ok2_pos = b == b and b > floor         # 反向：多样性↑ → 行为↑
    if ok1 and ok2_neg:
        print("  两环都成立：评分表奖励多样，读者偏好重复，方向相反。")
        print("  → 近零效度有了机制解释，不再只是一个相关系数。")
    elif ok2_pos:
        # 原判读漏了这一支：② 不仅不成立，而且**显著反向**。
        # 这比单纯的零结果更有信息量，必须单独报出来。
        print(f"  ② 显著**反向**（{b:+.3f}）：用词越丰富的真人小说，收藏率越高。")
        print("  → 「读者偏好重复、AI 的多样性是负担」这一假设被证伪，而非仅仅未获支持。")
        if ok1:
            print("    且 ① 成立——评分表确实奖励多样。两环同向，")
            print("    因此多样性无法解释评分表的近零效度：它是双方共同看重的东西。")
        else:
            print("    ① 未检出——评分表对多样性并无显著偏好。")
        print("    多样性这条解释线应从讨论中撤下，另寻机制。")
    elif ok1:
        print("  ① 成立、② 未检出：评分表奖励多样，但多样性与读者行为无关。")
        print("  → 该解释未获支持；失效另有原因。")
    else:
        print("  两环均未检出：整条机制链的前提不存在，该解释应从讨论中撤下。")
    print(f"\n  （判定用可检出下限 {floor:.3f}；|r| 未超过它的一律记为未检出）")

    out = BASE / "repetition_payoff.json"
    out.write_text(json.dumps({"n": n, "cells": len(cells),
                               "link1": link1,
                               "link2": {f"{k[0]}|{k[1]}": v for k, v in link2.items()}},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  已保存：{out}")


if __name__ == "__main__":
    main()
