#!/usr/bin/env python3
"""
pool_analysis.py — 把各评审、各品类的结果合并成一份可写进论文的分析

补上 validate_rubric.py 缺的那一步：它只按品类分别算相关、分别存文件，
没有合并。而单品类 n≈20 只能检出 |ρ|≥0.43，跑出低相关也无法解读
——分不清"真没关系"还是"样本不够"。

本脚本做四件事：

  1. 品类内标准化后合并
     直接把各品类堆一起会把"品类间的均值差"误算成"评分表有效"
     （古言脑洞整体 8 分、动漫衍生整体 7 分）。先在每个品类内部把
     真实评分与评分表分数各自转成 z 分，再合并，品类差异即被抵消。

  2. 范围限制校正（Thorndike Case II）
     排除污染要砍掉最高 5 本和最低 2 本，代价是砍掉 37–70% 的方差。
     方差被砍，相关会被结构性压低。用全集 SD 作为未截断参照做校正，
     同时报告校正前后两个值。

  3. 评审间一致性
     若各评审彼此高度相关、却都与真实读者零相关，说明它们共享同一套
     偏见——这比任何单个评审的失效都更有力。

  4. 维度级分析
     十个维度里哪些与真实读者最脱钩。预期技法维度（画面感、对话自然度）
     相关较高，新颖性维度（戏剧性反转、章节差异度）接近零。

用法：
  python3 pool_analysis.py
"""
from __future__ import annotations

import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from validate_rubric import pearson, spearman  # noqa: E402

BASE = Path("review_loop")
TOP_EXCL, BOT_EXCL = 5, 2


def discover_cells() -> dict[str, str]:
    """从结果文件反查 品类→语料目录。
    早期版本这里硬编码了试水的两个品类，全量跑完后只吃到 80 本而非 840。
    改为扫描实际产出，格子增减都不用再动代码。
    同名品类（男女频各一套）的结果目录带 __Nanpin / __Nvpin 后缀，
    这里用目录名作键，保证两套不会互相覆盖。"""
    out = {}
    for p in sorted(BASE.glob("*/validation_pilot.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        src = d.get("source_dir")
        if src:
            out[p.parent.name] = src
    return out


CELL_SRC = discover_cells()


def min_detectable_r(n: int, t: float = 1.96) -> float:
    """α=.05 双尾下，样本量 n 能检出的最小 |r|。"""
    if n < 4:
        return float("nan")
    df = n - 2
    tt = 1.96 + 2.6 / max(df, 1)          # 对小样本 t 分位的粗略修正
    return tt / math.sqrt(tt ** 2 + df)


def full_category_sd(cell: str) -> float | None:
    """未排除污染前的全集离散度，作为范围限制校正的参照。
    参数是结果目录名；同名品类带 __Nanpin/__Nvpin 后缀，须剥掉才是语料目录名。"""
    src = CELL_SRC.get(cell)
    if not src:
        return None
    category = cell.split("__")[0]
    rs = []
    for f in (Path(src) / category).glob("*.txt"):
        m = re.search(r"^评分[：:]\s*(.+)$",
                      f.read_text(encoding="utf-8", errors="replace")[:2000], re.M)
        if m:
            try:
                v = float(m.group(1))
                if v > 0:
                    rs.append(v)
            except ValueError:
                pass
    return statistics.pstdev(rs) if len(rs) > 2 else None


def correct_range_restriction(r: float, sd_restricted: float, sd_full: float) -> float:
    """Thorndike Case II：已知未截断变量的 SD，校正相关。"""
    if not r or not sd_restricted or not sd_full or sd_restricted == 0:
        return float("nan")
    u = sd_full / sd_restricted
    return r * u / math.sqrt(1 + r ** 2 * (u ** 2 - 1))


def zscore(vals: list[float]) -> list[float]:
    s = statistics.pstdev(vals)
    if not s:
        return []
    m = statistics.mean(vals)
    return [(v - m) / s for v in vals]


def load_judges() -> dict[str, dict[str, list[dict]]]:
    """→ {评审名: {品类: [ {file, real_rating, score, dimensions}, ... ]}}"""
    judges: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))

    # 当前结果 + 被改名留档的旧结果。留档的是别的评审打的（试水用 o3），
    # 不读回来就会丢掉整个 o3 臂——它是三评审对比的基线。
    for cell in CELL_SRC:
        for p in [BASE / cell / "validation_pilot.json",
                  *sorted((BASE / cell).glob("validation_pilot_prev_*.json"))]:
            if not p.exists():
                continue
            d = json.loads(p.read_text(encoding="utf-8"))
            name = d.get("reviewer_model") or "o3（试水·无元数据）"
            for r in d["results"]:
                judges[name][cell].append({
                    # 用书名做连接键：交叉评审那边只存了书名，用文件名会连不上
                    "file": r.get("title") or r["file"], "real_rating": r["real_rating"],
                    "score": r["rubric_score"], "reps": r.get("rubric_scores") or [],
                    "dimensions": r.get("dimensions") or {}, "kind": "真人",
                })

    for p in sorted(BASE.glob("crossjudge_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        name = d.get("model", p.stem.replace("crossjudge_", ""))
        for r in d["rows"]:
            judges[name][r["category"]].append({
                "file": r.get("label"), "real_rating": r.get("real_rating"),
                "score": r["new_score"], "reps": [], "dimensions": {},
                "kind": r.get("kind", "真人"),
            })

    # 去重：同一个评审可能在两处评过同一本书——全量跑存进 validation_pilot.json，
    # 交叉评审又存进 crossjudge_*.json（民国言情、动漫衍生两格）。
    # 不去重会把 840 虚增成 878，把相关观测当独立样本，可检出下限被假性压低。
    # 保留先入的那条（全量跑的口径统一，且带 run 元数据）。
    dropped = 0
    for name, cells in judges.items():
        for cell, rows in cells.items():
            seen, keep = set(), []
            for r in rows:
                key = (r["kind"], r["file"])
                if key in seen:
                    dropped += 1
                    continue
                seen.add(key)
                keep.append(r)
            cells[cell] = keep
    if dropped:
        print(f"  [去重] 剔除 {dropped} 条同评审重复评分\n")
    return judges


def main() -> None:
    judges = load_judges()
    if not judges:
        raise SystemExit("没有找到任何结果文件。先跑 validate_pilot.py / crossjudge.py。")

    print(f"\n{'=' * 74}\n  1. 各评审 × 各品类（仅真人小说）\n{'=' * 74}")
    print(f"  {'评审':<20}{'品类':<10}{'n':>4}{'r':>9}{'ρ':>9}{'范围校正r':>11}{'可检出下限':>11}")
    print("  " + "-" * 72)

    for judge, cats in judges.items():
        for cat, rows in cats.items():
            real_rows = [r for r in rows if r["kind"] == "真人" and r["real_rating"]]
            if len(real_rows) < 4:
                continue
            rr = [r["real_rating"] for r in real_rows]
            sc = [r["score"] for r in real_rows]
            r_p, r_s = pearson(sc, rr), spearman(sc, rr)
            sd_full = full_category_sd(cat)
            corr = correct_range_restriction(r_p, statistics.pstdev(rr), sd_full) if sd_full else float("nan")
            print(f"  {judge:<20}{cat:<10}{len(real_rows):>4}{r_p:>+9.3f}{r_s:>+9.3f}"
                  f"{corr:>+11.3f}{min_detectable_r(len(real_rows)):>11.2f}")

    print(f"\n{'=' * 74}\n  2. 品类内标准化后合并\n{'=' * 74}")
    for judge, cats in judges.items():
        zr, zs = [], []
        for cat, rows in cats.items():
            real_rows = [r for r in rows if r["kind"] == "真人" and r["real_rating"]]
            if len(real_rows) < 4:
                continue
            a = zscore([r["real_rating"] for r in real_rows])
            b = zscore([r["score"] for r in real_rows])
            if a and b:
                zr += a
                zs += b
        if len(zr) >= 4:
            print(f"  {judge:<20} n={len(zr):<4} r = {pearson(zs, zr):+.3f}   "
                  f"ρ = {spearman(zs, zr):+.3f}   （可检出下限 {min_detectable_r(len(zr)):.2f}）")

    print(f"\n{'=' * 74}\n  3. 评审间一致性（同一批真人小说）\n{'=' * 74}")
    names = list(judges)
    common: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for cat in CELL_SRC:
                ra = {r["file"]: r["score"] for r in judges[a].get(cat, []) if r["kind"] == "真人"}
                rb = {r["file"]: r["score"] for r in judges[b].get(cat, []) if r["kind"] == "真人"}
                for k in set(ra) & set(rb):
                    common[(a, b)].append((ra[k], rb[k]))
    for (a, b), pairs in common.items():
        if len(pairs) >= 4:
            xs, ys = zip(*pairs)
            print(f"  {a} ↔ {b}   n={len(pairs)}   r = {pearson(list(xs), list(ys)):+.3f}")

    print(f"\n{'=' * 74}\n  4. AI 产出 vs 真人小说（动漫衍生）\n{'=' * 74}")
    print(f"  {'评审':<20}{'AI':>8}{'真人':>9}{'差距':>9}{'真人≥AI最低分':>15}")
    print("  " + "-" * 72)
    for judge, cats in judges.items():
        rows = cats.get("动漫衍生", [])
        ai = [r["score"] for r in rows if r["kind"] == "AI"]
        hu = [r["score"] for r in rows if r["kind"] == "真人"]
        if ai and hu:
            above = sum(1 for h in hu if h >= min(ai))
            print(f"  {judge:<20}{statistics.mean(ai):>8.2f}{statistics.mean(hu):>9.2f}"
                  f"{statistics.mean(ai) - statistics.mean(hu):>+9.2f}{f'{above}/{len(hu)}':>15}")

    print(f"\n{'=' * 74}\n  5. 维度级：哪些维度与真实读者最脱钩\n{'=' * 74}")
    for judge, cats in judges.items():
        dim_pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for cat, rows in cats.items():
            for r in rows:
                if r["kind"] != "真人" or not r["real_rating"]:
                    continue
                for name, v in (r["dimensions"] or {}).items():
                    if isinstance(v, (int, float)):
                        dim_pairs[name].append((v, r["real_rating"]))
        if not dim_pairs:
            continue
        print(f"\n  评审 {judge}")
        scored = []
        for name, pairs in dim_pairs.items():
            if len(pairs) >= 8:
                xs, ys = zip(*pairs)
                scored.append((pearson(list(xs), list(ys)), name, len(pairs)))
        for r_val, name, n in sorted(scored, reverse=True):
            bar = "█" * max(0, round(abs(r_val) * 20))
            print(f"    {name:<10} n={n:<4} r = {r_val:+.3f}  {bar}")
        if not scored:
            print("    （维度分数只在 validate_pilot 的结果里保存，交叉评审未留存）")

    print(f"\n{'=' * 74}\n  说明\n{'=' * 74}")
    print("  · 范围校正列用 Thorndike Case II，参照为该品类未排除污染前的全集 SD。")
    print("  · 「可检出下限」是 α=.05 双尾下该样本量能检出的最小 |r|；")
    print("    实测 |r| 低于它时，只能报「未检出关系」，不能报「无关系」。")
    print("  · 合并用品类内 z 分，已消除品类间均值差。")


if __name__ == "__main__":
    main()
