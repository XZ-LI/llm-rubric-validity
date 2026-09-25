#!/usr/bin/env python3
"""
diversity.py — RQ3：迭代过程中，输出的多样性怎么变？

为什么这条要优先做：三个评审对同一批文本给出三个方向（持平/上升/下降），
说明任何以评审分数为因变量的结论都不稳。而多样性度量**完全不依赖评审模型**
——它直接从文本算，是目前唯一不受"谁来打分"影响的因变量。零 API 成本。

假设（提示空间的模型坍缩）：
  自我改进循环里，编辑模型每轮把上一轮的"毛病"写成禁令，约束越积越多。
  若多样性随轮次单调下降而评分持平，即为坍缩证据——
  系统不是在变好，是在收窄。

四个度量：
  distinct-n   每轮文本内部的词汇多样性（定长采样，消除长度效应）
  MATTR        移动窗口类符形符比，对长度不敏感
  轮间收敛      相邻轮次的用词重合度；上升 = 各轮越写越像
  真人基线      同品类真人小说前三章的同一批度量，作为绝对参照

长度归一化是必须的：各轮 1 万–1.3 万字不等，不归一化测的是长度不是多样性。

用法：
  python3 diversity.py
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

import jieba

sys.path.insert(0, str(Path(__file__).parent))
from crossjudge import extract_ai_body  # noqa: E402
from validate_rubric import pearson, spearman  # noqa: E402

BASE = Path("review_loop")
SAMPLE_TOKENS = 2500      # 定长采样：取每份文本的前 N 个词，消除长度差异
MATTR_WINDOW = 500

jieba.setLogLevel(60)


def tokens(text: str) -> list[str]:
    """分词并去掉标点与空白——标点在各轮之间高度一致，留着会稀释差异。"""
    return [w for w in jieba.lcut(text)
            if w.strip() and not re.fullmatch(r"[\W_]+", w, re.UNICODE)]


def distinct_n(toks: list[str], n: int) -> float:
    if len(toks) < n:
        return float("nan")
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return len(set(grams)) / len(grams)


def mattr(toks: list[str], w: int = MATTR_WINDOW) -> float:
    """移动窗口 TTR。普通 TTR 随文本变长必然下降，无法跨长度比较。"""
    if len(toks) < w:
        return len(set(toks)) / len(toks) if toks else float("nan")
    vals = [len(set(toks[i:i + w])) / w for i in range(0, len(toks) - w + 1, 50)]
    return statistics.mean(vals)


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else float("nan")


def load_iters(cell: str) -> list[tuple[int, list[str]]]:
    out = []
    for d in sorted((BASE / cell).glob("iter_*"), key=lambda p: int(p.name.split("_")[1])):
        nv = d / "novel.txt"
        if not nv.exists():
            continue
        t = tokens(extract_ai_body(nv))
        if len(t) >= SAMPLE_TOKENS:
            out.append((int(d.name.split("_")[1]), t[:SAMPLE_TOKENS]))
    return out


def load_scores(cell: str) -> dict[int, float]:
    out = {}
    for d in (BASE / cell).glob("iter_*"):
        ev = d / "evaluation.json"
        if ev.exists():
            try:
                v = json.loads(ev.read_text(encoding="utf-8")).get("weighted_total")
            except json.JSONDecodeError:
                continue
            if v is not None:
                out[int(d.name.split("_")[1])] = float(v)
    return out


def human_baseline(cell: str, k: int = 12) -> dict | None:
    """同品类真人小说前三章，同样定长采样，作为绝对参照。"""
    p = BASE / cell / "validation_pilot.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    vals = {"d1": [], "d2": [], "d3": [], "mattr": []}
    for r in d["results"][:k]:
        f = Path(d["source_dir"]) / d["category"] / r["file"]
        if not f.exists():
            continue
        body = f.read_text(encoding="utf-8", errors="replace").split("=" * 20, 1)[-1]
        # 必须先截断再分词：真人小说动辄几十万字，对全文分词会慢到跑不完。
        # 2500 词大约对应 4000–5000 汉字，取 20000 字符足够且留有余量。
        t = tokens(body[:20000])
        if len(t) < SAMPLE_TOKENS:
            continue
        t = t[:SAMPLE_TOKENS]
        vals["d1"].append(distinct_n(t, 1))
        vals["d2"].append(distinct_n(t, 2))
        vals["d3"].append(distinct_n(t, 3))
        vals["mattr"].append(mattr(t))
    if not vals["d1"]:
        return None
    return {k2: statistics.mean(v) for k2, v in vals.items()} | {"n": len(vals["d1"])}


def main() -> None:
    cells = [p.name for p in sorted(BASE.iterdir())
             if p.is_dir() and list(p.glob("iter_*/novel.txt"))]
    print(f"\n{'=' * 76}\n  RQ3 多样性：迭代过程中输出是否收窄\n{'=' * 76}")
    print(f"  {len(cells)} 个品类有迭代数据；每份文本取前 {SAMPLE_TOKENS} 词（定长，消除长度效应）")

    rows = []
    for cell in cells:
        its = load_iters(cell)
        if len(its) < 6:
            continue
        scores = load_scores(cell)
        d1 = [distinct_n(t, 1) for _, t in its]
        d2 = [distinct_n(t, 2) for _, t in its]
        mt = [mattr(t) for _, t in its]
        idx = [i for i, _ in its]
        sc = [scores.get(i) for i in idx]

        # 轮次 × 多样性 的趋势
        tr_d1 = spearman(idx, d1)
        tr_d2 = spearman(idx, d2)
        tr_mt = spearman(idx, mt)
        # 分数趋势，作对照
        pair = [(i, s) for i, s in zip(idx, sc) if s is not None]
        tr_sc = spearman([p[0] for p in pair], [p[1] for p in pair]) if len(pair) >= 4 else float("nan")

        # 相邻轮次用词重合：上升 = 各轮越写越像
        sets = [set(t) for _, t in its]
        adj = [jaccard(sets[i], sets[i + 1]) for i in range(len(sets) - 1)]
        tr_adj = spearman(list(range(len(adj))), adj)

        rows.append({"cell": cell, "n": len(its), "d1": d1, "d2": d2, "mattr": mt,
                     "tr_d1": tr_d1, "tr_d2": tr_d2, "tr_mt": tr_mt,
                     "tr_sc": tr_sc, "adj": adj, "tr_adj": tr_adj,
                     "base": human_baseline(cell)})

    print(f"\n{'─' * 76}\n  1. 多样性随轮次的趋势（Spearman ρ，负 = 收窄）\n{'─' * 76}")
    print(f"  {'品类':<12}{'轮':>4}{'distinct-1':>12}{'distinct-2':>12}{'MATTR':>9}"
          f"{'轮间重合':>10}{'分数趋势':>10}")
    for r in rows:
        print(f"  {r['cell']:<12}{r['n']:>4}{r['tr_d1']:>+12.3f}{r['tr_d2']:>+12.3f}"
              f"{r['tr_mt']:>+9.3f}{r['tr_adj']:>+10.3f}{r['tr_sc']:>+10.3f}")
    for key, lab in (("tr_d1", "distinct-1"), ("tr_d2", "distinct-2"),
                     ("tr_mt", "MATTR"), ("tr_adj", "轮间重合"), ("tr_sc", "分数")):
        v = [r[key] for r in rows if r[key] == r[key]]
        if v:
            neg = sum(1 for x in v if x < 0)
            print(f"\n  {lab:<10} 中位 ρ = {statistics.median(v):+.3f}"
                  f"　为负的品类 {neg}/{len(v)}")

    print(f"\n{'─' * 76}\n  2. 绝对水平：AI 各轮 vs 同品类真人小说\n{'─' * 76}")
    print(f"  {'品类':<12}{'AI d-1':>9}{'真人 d-1':>10}{'AI MATTR':>11}{'真人 MATTR':>12}{'真人n':>7}")
    ai_d1, hu_d1, ai_mt, hu_mt = [], [], [], []
    for r in rows:
        b = r["base"]
        if not b:
            continue
        a1, a2 = statistics.mean(r["d1"]), statistics.mean(r["mattr"])
        print(f"  {r['cell']:<12}{a1:>9.3f}{b['d1']:>10.3f}{a2:>11.3f}{b['mattr']:>12.3f}{b['n']:>7}")
        ai_d1.append(a1); hu_d1.append(b["d1"])
        ai_mt.append(a2); hu_mt.append(b["mattr"])
    if ai_d1:
        print(f"\n  均值：AI distinct-1 {statistics.mean(ai_d1):.3f} vs 真人 {statistics.mean(hu_d1):.3f}"
              f"　（差 {statistics.mean(ai_d1) - statistics.mean(hu_d1):+.3f}）")
        print(f"  均值：AI MATTR      {statistics.mean(ai_mt):.3f} vs 真人 {statistics.mean(hu_mt):.3f}"
              f"　（差 {statistics.mean(ai_mt) - statistics.mean(hu_mt):+.3f}）")

    out = BASE / "diversity.json"
    out.write_text(json.dumps(
        [{k: v for k, v in r.items() if k != "base"} | {"baseline": r["base"]} for r in rows],
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{'=' * 76}")
    print("  判读：多样性单调下降 + 分数持平 = 提示空间坍缩（系统在收窄而非变好）；")
    print("        多样性无趋势 = 坍缩假设不成立，平台期另有原因。")
    print(f"  已保存：{out}")


if __name__ == "__main__":
    main()
