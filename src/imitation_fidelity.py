#!/usr/bin/env python3
"""
imitation_fidelity.py — 模型对哪些品类模仿得更像？

为什么不用评审分数：四个评审对同一批文本给出跨度 3.62 分的判断，
用分数衡量"像不像"会把评审的脾气当成文本的性质。本脚本只从文本算。

方法：Burrows's Delta（作者归属研究的标准量度）
  1. 取高频词表（跨全语料的前 N 个词，以功能词与常用词为主——
     它们的分布是作者/文类指纹，不受题材内容左右）
  2. 每个品类内，把各文本的词频 z 标准化
  3. Delta = 待测文本与该品类参照轮廓的 z 分平均绝对差；越小越像

关键在基准：Delta 的绝对值没有意义，必须和"真人小说彼此有多远"比。
  · 真人基准 = 留出集内每部书对该品类轮廓的 Delta（留一法，避免自证）
  · AI 的 Delta 若落在真人分布之内 → 文体上已不可区分
  · 若明显在外 → 模仿未达该品类的常态

输出的核心是 **AI 的 Delta 在真人分布里的百分位**：
0% 表示比所有真人小说都更贴近该品类轮廓，100% 表示比谁都远。

用法：
  python3 imitation_fidelity.py
"""
from __future__ import annotations

import collections
import json
import math
import statistics
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from contamination import excluded_files  # noqa: E402
from crossjudge import extract_ai_body  # noqa: E402
from diversity import tokens  # noqa: E402

BASE = Path("review_loop")
SAMPLE = 2500        # 每份文本取前 N 词，定长消除长度效应
TOP_WORDS = 300      # 高频词表规模
CHARS = 14000        # 先截断再分词


def tok_job(job):
    kind, label, path_s, is_ai = job
    p = Path(path_s)
    if not p.exists():
        return None
    raw = extract_ai_body(p) if is_ai else \
        p.read_text(encoding="utf-8", errors="replace").split("=" * 20, 1)[-1]
    t = tokens(raw[:CHARS])
    return (kind, label, t[:SAMPLE]) if len(t) >= SAMPLE else None


def build_jobs():
    jobs = []
    for p in sorted(BASE.glob("*/validation_pilot.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("reviewer_model") != "qwen3-max":
            continue
        cell = p.parent.name
        if not list((BASE / cell).glob("iter_*/novel.txt")):
            continue            # 只看跑过生成循环的品类
        drop = excluded_files(cell, d["category"], d["source_dir"])
        for r in d["results"]:
            if r["file"] in drop:
                continue
            jobs.append((cell, f"真人:{r.get('title', r['file'])[:18]}",
                         str(Path(d["source_dir"]) / d["category"] / r["file"]), False))
        for it in sorted((BASE / cell).glob("iter_*/novel.txt"),
                         key=lambda q: int(q.parent.name.split("_")[1])):
            jobs.append((cell, f"AI:{it.parent.name}", str(it), True))
    return jobs


def delta_profile(docs: list[list[str]], vocab: list[str],
                  min_docs: int = 3):
    """→ (每篇的 z 分向量, 保留的词表, 均值, 标准差)。z 标准化在品类内做。

    必须按品类筛词：高频词表取自全语料，但某个词可能在**这个品类**的
    真人文本里从不出现，品类内方差为 0。早先用 `or 1e-12` 兜底，结果
    AI 文本里该词一出现，z 分就炸到千万量级（西方奇幻曾算出 Delta=2e6），
    并静默污染其余品类。标准做法是直接剔除零方差特征。
    """
    freqs = []
    for t in docs:
        c = collections.Counter(t)
        n = len(t)
        freqs.append([c.get(w, 0) / n for w in vocab])
    cols = list(zip(*freqs))
    keep = [i for i, col in enumerate(cols)
            if statistics.pstdev(col) > 0 and sum(1 for x in col if x > 0) >= min_docs]
    vocab2 = [vocab[i] for i in keep]
    mu = [statistics.mean(cols[i]) for i in keep]
    sd = [statistics.pstdev(cols[i]) for i in keep]
    zs = [[(row[i] - m) / s for i, m, s in zip(keep, mu, sd)] for row in freqs]
    return zs, vocab2, mu, sd


def main() -> None:
    jobs = build_jobs()
    print(f"\n{'=' * 78}\n  模仿保真度：Burrows's Delta\n{'=' * 78}")
    print(f"  {len(jobs)} 份文本，分词中…", flush=True)
    with Pool(6) as pool:
        got = [r for r in pool.map(tok_job, jobs, chunksize=6) if r]

    by = collections.defaultdict(lambda: {"真人": [], "AI": []})
    for cell, label, t in got:
        by[cell]["AI" if label.startswith("AI:") else "真人"].append((label, t))

    # 高频词表取自全部真人文本，避免被 AI 的用词习惯带偏
    allc = collections.Counter()
    for v in by.values():
        for _, t in v["真人"]:
            allc.update(t)
    vocab = [w for w, _ in allc.most_common(TOP_WORDS)]
    print(f"  高频词表 {len(vocab)} 词（取自真人文本）\n")

    print(f"  {'品类':<10}{'真人n':>6}{'AI n':>6}{'真人Delta中位':>14}"
          f"{'AI Delta中位':>13}{'AI百分位':>10}{'判读':>12}")
    rows = []
    for cell in sorted(by):
        hu = by[cell]["真人"]
        ai = by[cell]["AI"]
        if len(hu) < 8 or not ai:
            continue
        docs = [t for _, t in hu]
        zs, vocab2, mu, sd = delta_profile(docs, vocab)

        # 真人基准：留一法，避免自己参与构成轮廓
        hu_delta = []
        for i in range(len(zs)):
            others = [zs[j] for j in range(len(zs)) if j != i]
            prof = [statistics.mean(col) for col in zip(*others)]
            hu_delta.append(statistics.mean(abs(a - b) for a, b in zip(zs[i], prof)))
        prof_all = [statistics.mean(col) for col in zip(*zs)]

        ai_delta = []
        for _, t in ai:
            c = collections.Counter(t)
            n = len(t)
            zv = [((c.get(w, 0) / n) - m) / s for w, m, s in zip(vocab2, mu, sd)]
            ai_delta.append(statistics.mean(abs(a - b) for a, b in zip(zv, prof_all)))

        med_ai = statistics.median(ai_delta)
        pct = sum(1 for d in hu_delta if d < med_ai) / len(hu_delta) * 100
        verdict = ("落在真人分布内" if pct <= 90 else
                   "略超出" if pct <= 99 else "明显在外")
        rows.append((cell, len(hu), len(ai), statistics.median(hu_delta),
                     med_ai, pct, verdict, ai_delta, hu_delta))
    for cell, nh, na, mh, ma, pct, v, _, _ in sorted(rows, key=lambda r: r[5]):
        print(f"  {cell:<10}{nh:>6}{na:>6}{mh:>14.3f}{ma:>13.3f}{pct:>9.0f}%{v:>12}")

    print(f"\n{'─' * 78}")
    print("  百分位 = AI 的 Delta 中位数在真人 Delta 分布里的位置。")
    print("  50% 表示 AI 与该品类轮廓的距离和典型真人小说相当；")
    print("  100% 表示比该品类里最离群的真人小说还要远。")

    out = BASE / "imitation_fidelity.json"
    out.write_text(json.dumps(
        [{"cell": c, "n_human": nh, "n_ai": na,
          "human_delta_median": mh, "ai_delta_median": ma,
          "ai_percentile": pct, "verdict": v,
          "ai_deltas": ad, "human_deltas": hd}
         for c, nh, na, mh, ma, pct, v, ad, hd in rows],
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  已保存：{out}")


if __name__ == "__main__":
    main()
