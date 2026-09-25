#!/usr/bin/env python3
"""
covariates.py — 读者评分作为效标，够不够干净？

背景：人类配对比较推迟到修改阶段后，番茄读者评分成为唯一效标。
它于是必须独自承受全部质疑。本脚本检验针对它的三个竞争解释，
全部用 txt 头部已有的元数据，零 API 成本。

  1. 构念错配   评分是整本书的（中位 454 章），评审只看前 3 章。
                检验：短书里三章占比大得多，评分表是否更准？
                若短书相关明显更高 → 错配是零相关的原因，主结论需收回。

  2. 篇幅混淆   若读者偏爱长书、而评分表给长书低分，两者方向相反
                会人为制造负相关。须把篇幅作为协变量偏相关掉。

  3. 热度混淆   「在读」人数是曝光度。若评分主要跟人数走，
                那它测的是热度而非质量，作为效标就不合格。

用法：
  python3 covariates.py                    # 用现有结果（试水 or 全量）
  python3 covariates.py --pattern validation_pilot.json
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from validate_rubric import pearson, spearman  # noqa: E402

BASE = Path("review_loop")
SOURCES = ("Nanpin", "Nvpin")


def parse_readers(s: str) -> float | None:
    """『2.4万人在读』→ 24000；『8231人在读』→ 8231。"""
    m = re.search(r"([\d.]+)\s*(万)?", s or "")
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    return v * 10000 if m.group(2) else v


def header_fields(path: Path) -> dict:
    head = path.read_text(encoding="utf-8", errors="replace")[:4000]

    def f(label):
        m = re.search(rf"^{label}[：:]\s*(.+)$", head, re.MULTILINE)
        return m.group(1).strip() if m else ""

    def num(label):
        try:
            return float(re.sub(r"[^\d.]", "", f(label)) or 0) or None
        except ValueError:
            return None

    # 番茄的实际取值是「连载 / 完结 / 未知」。早先这里写成 == "完本"，
    # 永远匹配不上，于是 840 本全被判成连载——完全错误的描述。
    # 「未知」是缺失值，不能当成连载，故用三态而非布尔。
    st = f("状态").strip()
    status = st if st in ("连载", "完结") else None
    return {"words": num("字数"), "chapters": num("章节"),
            "readers": parse_readers(f("在读")), "status": status}


def partial(xs, ys, zs):
    """控制 z 之后 x 与 y 的偏相关。"""
    rxy, rxz, ryz = pearson(xs, ys), pearson(xs, zs), pearson(ys, zs)
    den = math.sqrt(max(1e-12, (1 - rxz ** 2) * (1 - ryz ** 2)))
    return (rxy - rxz * ryz) / den


def min_detectable(n):
    if n < 4:
        return float("nan")
    df = n - 2
    tt = 1.96 + 2.6 / max(df, 1)
    return tt / math.sqrt(tt ** 2 + df)


def standardize_within(rows: list[dict], keys=("rating", "score")) -> list[dict]:
    """按品类把各列转 z 分后再合并。

    不做这一步，品类间的均值差会被算成相关：本脚本最初直接堆 34 个品类的
    原始分，得到 r=+0.146，而品类内标准化后只有 +0.067——一半以上是假的。
    协变量（篇幅、热度）同样按品类标准化，否则偏相关也建立在被污染的尺度上。
    """
    by_cat = collections.defaultdict(list)
    for r in rows:
        by_cat[r["cat"]].append(r)
    out = []
    for cat, grp in by_cat.items():
        z = {}
        for k in list(keys) + ["chapters", "words", "readers"]:
            vals = [r[k] for r in grp if r.get(k) is not None]
            if len(vals) < 3:
                continue
            m, s = statistics.mean(vals), statistics.pstdev(vals)
            if s:
                z[k] = (m, s)
        for r in grp:
            nr = dict(r)
            # 原值另存：描述统计（中位数、区间）必须用原值，否则会印出
            # 「章节数中位 -0，区间 -2–4」这种把 z 分当章节数的胡话。
            for k in ("chapters", "words", "readers", "rating", "score"):
                if r.get(k) is not None:
                    nr["raw_" + k] = r[k]
            for k, (m, s_) in z.items():
                if r.get(k) is not None:
                    nr[k] = (r[k] - m) / s_
            out.append(nr)
    return out


def load(pattern: str) -> list[dict]:
    rows = []
    for p in sorted(BASE.glob(f"*/{pattern}")):
        d = json.loads(p.read_text(encoding="utf-8"))
        cat = d["category"]
        src = d.get("source_dir") or next(
            (s for s in SOURCES if (Path(s) / cat).exists()), None)
        if not src:
            continue
        for r in d["results"]:
            f = Path(src) / cat / r["file"]
            if not f.exists():
                continue
            # 用结果目录名而非 category 字段做分组键：男女频同名品类
            # （悬疑脑洞 / 游戏体育 / 科幻末世）的 category 相同，
            # 用它会把两个独立格子并成一组标准化。
            rows.append({"cat": p.parent.name, "title": r.get("title", ""),
                         "rating": r["real_rating"], "score": r["rubric_score"],
                         **header_fields(f)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="validation_pilot.json")
    args = ap.parse_args()

    rows = standardize_within(load(args.pattern))
    if len(rows) < 8:
        raise SystemExit(f"样本不足（{len(rows)}）。先跑 validate_pilot.py。")

    ok = lambda k: [r for r in rows if r.get(k) is not None]
    n = len(rows)
    print(f"\n{'=' * 70}\n  效标体检：番茄读者评分够不够干净\n{'=' * 70}")
    print(f"  样本 {n} 部，来自 {len(set(r['cat'] for r in rows))} 个品类"
          f"（可检出 |r| ≥ {min_detectable(n):.2f}）")

    ch = ok("chapters")
    if ch:
        c = [r["raw_chapters"] for r in ch if r.get("raw_chapters")]
        print(f"  章节数 中位 {statistics.median(c):.0f}  区间 {min(c):.0f}–{max(c):.0f}"
              f"   → 评审看 3 章 ≈ 中位书的 {3 / statistics.median(c) * 100:.1f}%")
    st = collections.Counter(r.get("status") or "未知" for r in rows)
    print("  状态： " + "　".join(f"{k} {v}" for k, v in st.most_common()))

    print(f"\n{'─' * 70}\n  质疑 1：评分是整本书的，评审只看前三章\n{'─' * 70}")
    if len(ch) >= 16:
        ch.sort(key=lambda r: r["raw_chapters"])
        half = len(ch) // 2
        for lab, grp in (("短书（章节少的一半）", ch[:half]), ("长书（多的一半）", ch[half:])):
            a = [r["score"] for r in grp]
            b = [r["rating"] for r in grp]
            print(f"  {lab:<20} n={len(grp):<4} 章节 {grp[0]['raw_chapters']:.0f}–{grp[-1]['raw_chapters']:.0f}"
                  f"   r = {pearson(a, b):+.3f}   ρ = {spearman(a, b):+.3f}")
        print("\n  判读：若短书相关明显更高 → 错配成立，主结论须收回；")
        print("        若短书同样趋零 → 错配救不了评分表，主结论加固。")

    print(f"\n{'─' * 70}\n  质疑 2：篇幅混淆\n{'─' * 70}")
    for key, lab in (("chapters", "章节数"), ("words", "字数")):
        g = ok(key)
        if len(g) < 8:
            continue
        z = [r[key] for r in g]
        rt = [r["rating"] for r in g]
        sc = [r["score"] for r in g]
        print(f"  读者评分 ↔ {lab:<5} r = {pearson(z, rt):+.3f}     "
              f"评分表分 ↔ {lab:<5} r = {pearson(z, sc):+.3f}")
        print(f"    控制{lab}后，评分表 ↔ 读者评分 偏相关 r = {partial(sc, rt, z):+.3f}"
              f"   （未控制 {pearson(sc, rt):+.3f}）")

    print(f"\n{'─' * 70}\n  质疑 3：热度混淆——评分测的是质量还是曝光\n{'─' * 70}")
    g = ok("readers")
    if len(g) >= 8:
        z = [r["readers"] for r in g]
        rt = [r["rating"] for r in g]
        sc = [r["score"] for r in g]
        rw = [r["raw_readers"] for r in g if r.get("raw_readers")]
        print(f"  样本 {len(g)} 部，在读人数 中位 {statistics.median(rw):,.0f}"
              f"  区间 {min(rw):,.0f}–{max(rw):,.0f}")
        print(f"  读者评分 ↔ 在读人数   r = {pearson(z, rt):+.3f}   ρ = {spearman(z, rt):+.3f}")
        print(f"  评分表分 ↔ 在读人数   r = {pearson(z, sc):+.3f}")
        print(f"    控制热度后，评分表 ↔ 读者评分 偏相关 r = {partial(sc, rt, z):+.3f}"
              f"   （未控制 {pearson(sc, rt):+.3f}）")
        print("\n  判读：若『评分↔在读』很高，说明评分很大程度是热度的代理，")
        print("        用它当质量效标须在论文里明确限定；这不推翻零相关，")
        print("        但会改变零相关的解释——那时是『评分表测不出热度』。")
    else:
        print("  在读人数字段缺失过多，跳过。")

    print(f"\n{'─' * 70}\n  质疑 4：连载中的评分是临时的\n{'─' * 70}")
    fin = [r for r in rows if r.get("status") == "完结"]
    ser = [r for r in rows if r.get("status") == "连载"]
    if len(fin) >= 12 and len(ser) >= 12:
        for lab, grp in (("完结（终局评分）", fin), ("连载（临时评分）", ser)):
            a = [r["score"] for r in grp]
            b = [r["rating"] for r in grp]
            print(f"  {lab:<18} n={len(grp):<5} r = {pearson(a, b):+.3f}   ρ = {spearman(a, b):+.3f}"
                  f"   （可检出 {min_detectable(len(grp)):.2f}）")
        rawf = [r["raw_rating"] for r in fin if r.get("raw_rating") is not None]
        raws = [r["raw_rating"] for r in ser if r.get("raw_rating") is not None]
        if rawf and raws:
            print(f"\n  原始评分均值：完结 {statistics.mean(rawf):.2f}　连载 {statistics.mean(raws):.2f}")
        print("\n  判读：完结书的评分已定型，是更干净的效标。若完结组相关明显更高，")
        print("        说明连载书的临时评分在稀释信号，主分析应以完结组为准。")
    else:
        print(f"  分组过小（完结 {len(fin)} / 连载 {len(ser)}），跳过。")

    print(f"\n{'=' * 70}")
    print("  注：所有子组 n 都小，单个系数多半不显著。全量 840 本跑完后重跑本脚本：")
    print("      python3 covariates.py --pattern validation_pilot.json")


if __name__ == "__main__":
    main()
