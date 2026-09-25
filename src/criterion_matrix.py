#!/usr/bin/env python3
"""
criterion_matrix.py — 多效标收敛分析

设计背景：只用平台读者反馈做效标（不做人工标注）。单一指标都不干净——
番茄评分里约 28% 的方差由篇幅与曝光解释。对策是用**一组污染方式互不相同**
的读者信号：如果评分表对它们全部失效，结论就不依赖任何一个指标是否干净。

六个效标：
  评分        自选样本的主观打分        怕热度与篇幅
  残差评分    评分对规模/曝光回归后的残差  怕过度控制
  收藏率      收藏 / 累计阅读           分母即曝光，热度被结构性消除
  留存比      在读 / 累计阅读           弃书率的反面，受连载时长影响
  听书率      听书 / 累计阅读           另一种消费形态，离文本工艺更远
  书架留存    收藏 / 累计加书架         字段语义未经核实，仅作探索

关键输出有三块：
  1. 评分表 × 六效标（品类内 z 标准化后合并 + 逐格符号检验）
  2. 六效标**彼此**的相关矩阵——若评分与收藏行为本身就脱节，
     那么前面所有围绕评分的分析都要重新定位
  3. 缺失偏倚检查——搜不到的书约占 4%，且多半是被下架/改名的作品，
     缺失可能非随机。须比较缺失组与在册组的评分分布。

用法：
  python3 criterion_matrix.py
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
from covariates import header_fields  # noqa: E402
from validate_rubric import pearson, spearman  # noqa: E402

BASE = Path("review_loop")
ENG = BASE / "engagement.json"


def min_detectable(n: int) -> float:
    if n < 4:
        return float("nan")
    df = n - 2
    tt = 1.96 + 2.6 / max(df, 1)
    return tt / math.sqrt(tt ** 2 + df)


def num(v):
    """平台 API 把计数字段返回成字符串（read_count_all = '13321942'），
    而 slim() 因为它是短字符串就原样留下了。直接喂 np.log 会炸，
    且真值比较（if v）对 '0' 这种字符串还会判真。一律先过这里。"""
    if isinstance(v, bool) or v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def safe_div(a, b):
    a, b = num(a), num(b)
    if a is None or b is None:
        return None
    return a / b if b > 0 else None


def z(v: list[float]) -> list[float]:
    s = statistics.pstdev(v)
    if not s:
        return []
    m = statistics.mean(v)
    return [(x - m) / s for x in v]


def load() -> tuple[dict[str, list[dict]], list[dict]]:
    """→ (按格子分组的在册书, 缺失书)。缺失书只带评分，用于偏倚检查。"""
    eng = json.loads(ENG.read_text(encoding="utf-8")) if ENG.exists() else {"books": {}}
    books = eng.get("books", {})

    cells: dict[str, list[dict]] = defaultdict(list)
    missing: list[dict] = []
    for p in sorted(BASE.glob("*/validation_pilot.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("reviewer_model") != "qwen3-max":
            continue
        for r in d["results"]:
            key = f"{p.parent.name}|{r['file']}"
            f = Path(d["source_dir"]) / d["category"] / r["file"]
            e = books.get(key)
            if not e:
                missing.append({"cell": p.parent.name, "rating": r["real_rating"],
                                "score": r["rubric_score"]})
                continue
            h = header_fields(f) if f.exists() else {}
            all_read = num(e.get("read_count_all"))
            cells[p.parent.name].append({
                "title": e.get("title", ""),
                "score": r["rubric_score"],
                "rating": r["real_rating"],
                "收藏率": safe_div(e.get("all_bookshelf_count"), all_read),
                "留存比": safe_div(e.get("read_count"), all_read),
                "听书率": safe_div(e.get("listen_count"), all_read),
                "书架留存": safe_div(e.get("all_bookshelf_count"), e.get("shelf_cnt_history")),
                "chapters": h.get("chapters"), "words": h.get("words"),
                "readers": all_read,   # 已转数值
                "fin": 1.0 if h.get("status") == "完结" else 0.0,
            })
    return cells, missing


def add_residual(rows: list[dict]) -> None:
    """评分对 log(章节)+log(字数)+log(累计阅读)+完结 回归，残差写回。"""
    use = [r for r in rows if all(r.get(k) for k in ("chapters", "words", "readers"))]
    if len(use) < 8:
        for r in rows:
            r["残差评分"] = None
        return
    y = np.array([r["rating"] for r in use], dtype=float)
    X = np.column_stack([np.ones(len(use)),
                         np.log([r["chapters"] for r in use]),
                         np.log([r["words"] for r in use]),
                         np.log([r["readers"] for r in use]),
                         [r["fin"] for r in use]])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = (y - X @ beta).tolist()
    for r in rows:
        r["残差评分"] = None
    for r, v in zip(use, res):
        r["残差评分"] = v


CRITERIA = ["评分", "残差评分", "收藏率", "留存比", "听书率", "书架留存"]
FIELD = {"评分": "rating", "残差评分": "残差评分", "收藏率": "收藏率",
         "留存比": "留存比", "听书率": "听书率", "书架留存": "书架留存"}


def main() -> None:
    cells, missing = load()
    for rows in cells.values():
        add_residual(rows)
    n_total = sum(len(v) for v in cells.values()) + len(missing)

    print(f"\n{'=' * 74}\n  多效标收敛分析\n{'=' * 74}")
    print(f"  留出集 {n_total} 本；取得行为数据 {n_total - len(missing)} 本，"
          f"缺失 {len(missing)} 本（{len(missing) / max(1, n_total) * 100:.1f}%）")
    print(f"  覆盖格子 {len(cells)} 个")

    # ── 1. 评分表 × 六效标 ──
    print(f"\n{'─' * 74}\n  1. 评分表分数 × 各效标（品类内 z 标准化后合并）\n{'─' * 74}")
    print(f"  {'效标':<10}{'n':>6}{'r':>9}{'ρ':>9}{'可检出':>9}{'逐格为正':>11}{'符号检验p':>11}")
    for name in CRITERIA:
        fld = FIELD[name]
        zs, zc, per_cell = [], [], []
        for cell, rows in cells.items():
            v = [(r["score"], r.get(fld)) for r in rows
                 if isinstance(r.get(fld), (int, float))]
            if len(v) < 8:
                continue
            a = [x[0] for x in v]
            b = [x[1] for x in v]
            za, zb = z(a), z(b)
            if not (za and zb):
                continue
            zs += za
            zc += zb
            per_cell.append(pearson(a, b))
        if len(zs) < 10:
            print(f"  {name:<10}{'—':>6}  数据不足")
            continue
        pos = sum(1 for r in per_cell if r > 0)
        N = len(per_cell)
        pv = sum(math.comb(N, i) * 0.5 ** N for i in range(pos, N + 1))
        print(f"  {name:<10}{len(zs):>6}{pearson(zs, zc):>+9.3f}{spearman(zs, zc):>+9.3f}"
              f"{min_detectable(len(zs)):>9.3f}{f'{pos}/{N}':>11}{pv:>11.4f}")

    # ── 2. 效标彼此 ──
    print(f"\n{'─' * 74}\n  2. 六个效标彼此的相关（品类内 z 标准化）\n{'─' * 74}")
    pooled: dict[str, list[float]] = defaultdict(list)
    for cell, rows in cells.items():
        usable = [r for r in rows
                  if all(isinstance(r.get(FIELD[c]), (int, float)) for c in CRITERIA)]
        if len(usable) < 8:
            continue
        for c in CRITERIA:
            pooled[c] += z([r[FIELD[c]] for r in usable])
    if pooled and len(next(iter(pooled.values()))) >= 10:
        m = len(next(iter(pooled.values())))
        print(f"  共同样本 n={m}（六项齐全的书）")
        head = "".join(f"{c:>10}" for c in CRITERIA)
        print(f"  {'':<10}{head}")
        for a in CRITERIA:
            row = "".join(f"{pearson(pooled[a], pooled[b]):>+10.3f}"
                          if a != b else f"{'—':>10}" for b in CRITERIA)
            print(f"  {a:<10}{row}")
    else:
        print("  六项齐全的样本不足，跳过。")

    # ── 3. 缺失偏倚 ──
    print(f"\n{'─' * 74}\n  3. 缺失偏倚检查\n{'─' * 74}")
    if missing:
        got_r = [r["rating"] for rows in cells.values() for r in rows]
        mis_r = [r["rating"] for r in missing]
        got_s = [r["score"] for rows in cells.values() for r in rows]
        mis_s = [r["score"] for r in missing]
        print(f"  在册 {len(got_r)} 本：读者评分 {statistics.mean(got_r):.2f}"
              f"（SD {statistics.pstdev(got_r):.2f}）　评分表 {statistics.mean(got_s):.2f}")
        print(f"  缺失 {len(mis_r)} 本：读者评分 {statistics.mean(mis_r):.2f}"
              f"（SD {statistics.pstdev(mis_r):.2f}）　评分表 {statistics.mean(mis_s):.2f}")
        d = statistics.mean(mis_r) - statistics.mean(got_r)
        sp = math.sqrt((statistics.pvariance(got_r) + statistics.pvariance(mis_r)) / 2)
        print(f"  评分差 {d:+.2f}（Cohen's d = {d / sp if sp else float('nan'):+.2f}）")
        print("\n  判读：|d| < 0.2 可视为缺失近似随机；更大则须在论文中报告")
        print("        缺失非随机，并说明搜不到的多为下架/改名作品。")
    else:
        print("  无缺失。")

    print(f"\n{'=' * 74}")
    print("  注：『书架留存』依赖 shelf_cnt_history 的字段语义，未经平台文档核实，")
    print("      仅作探索性指标，不应单独用于下结论。")


if __name__ == "__main__":
    main()
