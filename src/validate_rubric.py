#!/usr/bin/env python3
"""
validate_rubric.py — 检验「机器归纳出的评分表」是否真的能衡量小说质量。

review_loop.py 从高分源小说导出评分表，再用它给自己生成的文本打分。
但这个评分表从未被验证过：它能区分真实的好书和差书吗？

本脚本用**留出的真实小说**做检验：
  1. 载入该品类已缓存的评分表与评审 prompt（与 review_loop 完全一致）
  2. 排除塑造过评分表的书 —— 用于导出评分表的 top-5、以及评审 prompt 里的
     校准锚点（高分 top-3 / 低分 bottom-2）。留下的才是干净的测试集。
  3. 用同一套评审 prompt 给每部留出小说的前 3 章打分
  4. 计算「评分表分数」与「番茄真实读者评分」的相关性

判读：
  相关性高 → 评分表是有效代理，review_loop 的分数可以当质量读。
  相关性低 → 评分表与真实读者反馈无关。那么迭代中的「7.5 分平台期」
             并非创造力天花板，而是自指涉评估的产物 —— 这本身是核心发现。

用法：
  export OPENAI_API_KEY=sk-...
  python3 validate_rubric.py --category 西方奇幻 --dry-run   # 先看测试集与调用量
  python3 validate_rubric.py --category 西方奇幻
  python3 validate_rubric.py --category 西方奇幻 --repeat 3  # 同文本重复评分，测评审信度
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from review_loop import (  # noqa: E402
    CALIBRATION_HIGH_N,
    CALIBRATION_LOW_N,
    REVIEWER_MODEL,
    SOURCE_DIR,
    TOP_SOURCE_NOVELS,
    evaluate_chapters,
    format_calibration_block,
    load_calibration_anchors,
    parse_novel_txt,
)

OUTPUT_BASE = Path("review_loop")
CHAPTERS_TO_SCORE = 3          # 与 review_loop 的 CHAPTERS_PER_ITER 一致
CHAPTER_RE = re.compile(r"^第\s*[〇零一二三四五六七八九十百千\d]+\s*[章节]\s*(.*)$", re.MULTILINE)


def split_chapters(body: str, n: int) -> list[str]:
    """把正文切成前 n 章。评审端本来就只看每章前 2000 字。"""
    marks = list(CHAPTER_RE.finditer(body))
    if not marks:
        return [body[:2000]] if body.strip() else []
    out = []
    for i, m in enumerate(marks[:n]):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
        out.append(body[m.start():end].strip())
    return out


def load_category(category: str, source_dir: Path) -> list[dict]:
    cat_dir = source_dir / category
    if not cat_dir.exists():
        sys.exit(f"[ERROR] 找不到品类目录：{cat_dir}")
    novels = []
    for f in sorted(cat_dir.glob("*.txt")):
        try:
            d = parse_novel_txt(f)
            d["_path"] = f
            d["_rating"] = float(d.get("rating") or 0)
            novels.append(d)
        except Exception as e:
            print(f"  [跳过] {f.name}: {e}")
    return [n for n in novels if n["_rating"] > 0]


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else float("nan")


def spearman(xs: list[float], ys: list[float]) -> float:
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):            # 并列取平均秩
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return pearson(rank(xs), rank(ys))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", default="西方奇幻")
    ap.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    ap.add_argument("--limit", type=int, default=0, help="只评前 N 部（省钱）")
    ap.add_argument("--repeat", type=int, default=1, help="同一部重复评 N 次，测评审信度")
    ap.add_argument("--dry-run", action="store_true", help="只列测试集与调用量，不调 API")
    args = ap.parse_args()

    cat = args.category
    out_dir = OUTPUT_BASE / cat
    rubrics_path = out_dir / "rubrics.json"
    if not rubrics_path.exists():
        sys.exit(f"[ERROR] 没有该品类的评分表：{rubrics_path}\n先跑 review_loop.py 生成。")
    rubrics = json.loads(rubrics_path.read_text(encoding="utf-8"))

    novels = load_category(cat, args.source_dir)
    novels.sort(key=lambda n: n["_rating"], reverse=True)

    # 排除塑造过评分表 / 评审 prompt 的书
    contaminated = {n["_path"].name for n in novels[:max(TOP_SOURCE_NOVELS, CALIBRATION_HIGH_N)]}
    contaminated |= {n["_path"].name for n in novels[-CALIBRATION_LOW_N:]}
    held_out = [n for n in novels if n["_path"].name not in contaminated]
    if args.limit:
        # 保留评分跨度：按评分排序后均匀抽样
        step = max(1, len(held_out) // args.limit)
        held_out = held_out[::step][:args.limit]

    anchors = load_calibration_anchors(cat, source_dir=args.source_dir)
    calibration_block = format_calibration_block(anchors)

    print(f"\n{'='*66}")
    print(f"  评分表外部效度检验：{cat}")
    print(f"{'='*66}")
    print(f"  评分表维度 {len(rubrics)} 条，权重合计 {sum(r.get('weight',1) for r in rubrics)}")
    print(f"  该品类共 {len(novels)} 部有评分")
    print(f"  排除（污染）{len(contaminated)} 部：评分表来源 top-{TOP_SOURCE_NOVELS} + "
          f"校准锚点 高{CALIBRATION_HIGH_N}/低{CALIBRATION_LOW_N}")
    print(f"  留出测试集 {len(held_out)} 部，真实评分 "
          f"{min(n['_rating'] for n in held_out):.1f}–{max(n['_rating'] for n in held_out):.1f}")
    print(f"  评审模型 {REVIEWER_MODEL}，每部评 {CHAPTERS_TO_SCORE} 章 × {args.repeat} 次")
    print(f"  预计 API 调用：{len(held_out) * args.repeat} 次\n")

    for n in held_out:
        print(f"    ★{n['_rating']:<4} {n.get('title','')[:34]}")

    if args.dry_run:
        print("\n[dry-run] 未调用 API。去掉 --dry-run 正式运行。")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("[ERROR] 未设置 OPENAI_API_KEY")
    from openai import OpenAI
    client = OpenAI()

    results = []
    for i, n in enumerate(held_out, 1):
        body = n["_path"].read_text(encoding="utf-8", errors="replace")
        body = body.split("=" * 20, 1)[-1]
        chapters = split_chapters(body, CHAPTERS_TO_SCORE)
        if not chapters:
            print(f"  [{i}/{len(held_out)}] 跳过（切不出章节）：{n.get('title','')}")
            continue

        scores = []
        for rep in range(args.repeat):
            tag = f"{i}/{len(held_out)}" + (f" rep{rep+1}" if args.repeat > 1 else "")
            print(f"  [{tag}] ★{n['_rating']} {n.get('title','')[:26]} ", end="", flush=True)
            try:
                ev = evaluate_chapters(client, chapters, rubrics, cat, calibration_block)
                s = float(ev.get("weighted_total") or 0)
                scores.append(s)
                print(f"→ {s}")
            except Exception as e:
                print(f"→ 失败 {e}")
        if scores:
            results.append({
                "title": n.get("title", ""),
                "file": n["_path"].name,
                "real_rating": n["_rating"],
                "rubric_scores": scores,
                "rubric_mean": statistics.mean(scores),
            })

    if len(results) < 3:
        sys.exit("\n[ERROR] 有效样本不足 3 个，无法计算相关性。")

    real = [r["real_rating"] for r in results]
    got  = [r["rubric_mean"] for r in results]
    r_p, r_s = pearson(got, real), spearman(got, real)

    print(f"\n{'='*66}")
    print("  结果")
    print(f"{'='*66}")
    print(f"  样本 {len(results)} 部")
    print(f"  真实评分   均值 {statistics.mean(real):.2f}  标准差 {statistics.pstdev(real):.2f}"
          f"  跨度 {min(real):.1f}–{max(real):.1f}")
    print(f"  评分表分数 均值 {statistics.mean(got):.2f}  标准差 {statistics.pstdev(got):.2f}"
          f"  跨度 {min(got):.1f}–{max(got):.1f}")
    print(f"\n  Pearson  r = {r_p:+.3f}")
    print(f"  Spearman ρ = {r_s:+.3f}")

    if args.repeat > 1:
        sds = [statistics.pstdev(r["rubric_scores"]) for r in results if len(r["rubric_scores"]) > 1]
        if sds:
            print(f"\n  评审信度：同一文本重复评分的标准差 中位数 {statistics.median(sds):.3f}"
                  f"  最大 {max(sds):.3f}")
            print("  （这是噪声下限：小于它的迭代『改进』都不可解读）")

    ranked = sorted(results, key=lambda r: r["real_rating"])
    k = max(1, len(ranked) // 3)
    lo = statistics.mean(r["rubric_mean"] for r in ranked[:k])
    hi = statistics.mean(r["rubric_mean"] for r in ranked[-k:])
    print(f"\n  真实低分组(n={k}) 评分表均分 {lo:.2f}")
    print(f"  真实高分组(n={k}) 评分表均分 {hi:.2f}")
    print(f"  区分度 Δ = {hi-lo:+.2f}")

    print("\n  判读：")
    if not math.isnan(r_s) and abs(r_s) >= 0.5:
        print("    ρ ≥ 0.5 —— 评分表与真实读者评分中等以上相关，可作为质量代理。")
    elif not math.isnan(r_s) and abs(r_s) >= 0.3:
        print("    0.3 ≤ ρ < 0.5 —— 弱相关，结论需谨慎，建议扩大样本。")
    else:
        print("    ρ < 0.3 —— 评分表基本无法区分真实的好书与差书。")
        print("    则迭代中的分数平台期不能解读为『创造力天花板』，")
        print("    而是自指涉评估体系的产物 —— 这正是论文要论证的。")

    out = out_dir / "validation.json"
    out.write_text(json.dumps({
        "category": cat,
        "reviewer_model": REVIEWER_MODEL,
        "n": len(results),
        "excluded_contaminated": sorted(contaminated),
        "pearson_r": r_p,
        "spearman_rho": r_s,
        "discrimination_delta": hi - lo,
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  已保存：{out}")


if __name__ == "__main__":
    main()
