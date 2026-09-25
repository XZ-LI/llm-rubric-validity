#!/usr/bin/env python3
"""
reasoning_effect.py — RQ5：评审的推理强度如何影响它的判断？

线索：三个评审的推理量与它们偏爱 AI 的程度呈单调关系——

    qwen3-max（不思考）  推理/可见 = 0.00   →  AI 比真人高 +2.30
    o3                   推理/可见 = 1.19   →  +1.50
    deepseek-v4-pro      推理/可见 = 4.96   →  −0.02

想得越多，越不吃 AI 那套。但三个点不能下结论，且跨模型比较混淆了模型身份：
差异可能来自谱系、语料、对齐方式，而非推理本身。

本实验把混淆去掉：**同一个模型、同一批文本、只改思考开关**。
qwen-turbo 支持 enable_thinking，实测关=0 推理 token、开=284，开关有效。

若开思考后差距显著缩小 → 推理强度是因，单调关系成立；
若差距不变 → 那三个点的差异来自模型身份，与推理无关。

成本：32 份文本 × 2 组 = 64 次调用，约 ¥0.5。

用法：
  python3 reasoning_effect.py --dry-run
  python3 reasoning_effect.py
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from crossjudge import extract_ai_body  # noqa: E402
from review_loop import format_calibration_block, load_calibration_anchors  # noqa: E402
from validate_pilot import Ledger, evaluate_lean  # noqa: E402
from validate_rubric import CHAPTERS_TO_SCORE, pearson, split_chapters  # noqa: E402

BASE = Path("review_loop")
CELL, SRC = "动漫衍生", "Nanpin"
MODEL = "qwen-turbo"
CONDS = [("思考关", {"enable_thinking": False}),
         ("思考开", {"enable_thinking": True})]


def load_texts() -> list[dict]:
    """10 份 AI 产出 + 该品类的真人留出集。与交叉评审用的是同一批文本。"""
    items = []
    for i in range(1, 11):
        nv = BASE / CELL / f"iter_{i}" / "novel.txt"
        if not nv.exists():
            continue
        chs = split_chapters(extract_ai_body(nv), CHAPTERS_TO_SCORE)
        if chs:
            items.append({"kind": "AI", "label": f"iter_{i}", "chapters": chs,
                          "rating": None})

    d = json.loads((BASE / CELL / "validation_pilot.json").read_text(encoding="utf-8"))
    for r in d["results"]:
        f = Path(SRC) / CELL / r["file"]
        if not f.exists():
            continue
        body = f.read_text(encoding="utf-8", errors="replace").split("=" * 20, 1)[-1]
        chs = split_chapters(body, CHAPTERS_TO_SCORE)
        if chs:
            items.append({"kind": "真人", "label": r["title"], "chapters": chs,
                          "rating": r["real_rating"]})
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    items = load_texts()
    n_ai = sum(1 for i in items if i["kind"] == "AI")
    print(f"\n{'=' * 70}\n  RQ5 推理强度对照：{MODEL}\n{'=' * 70}")
    print(f"  文本 {len(items)} 份（AI {n_ai} + 真人 {len(items) - n_ai}）× {len(CONDS)} 组"
          f" = {len(items) * len(CONDS)} 次调用")
    if args.dry_run:
        print("  [dry-run] 未调 API。")
        return

    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise SystemExit("[ERROR] 未设置 DASHSCOPE_API_KEY")
    from openai import OpenAI
    client = OpenAI(api_key=key,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    rubrics = json.loads((BASE / CELL / "rubrics.json").read_text(encoding="utf-8"))
    calib = format_calibration_block(load_calibration_anchors(CELL, source_dir=Path(SRC)))

    results: dict[str, list[dict]] = {}
    ledgers: dict[str, Ledger] = {}
    for cname, extra in CONDS:
        led = Ledger(MODEL)
        ledgers[cname] = led
        print(f"\n  ── {cname} ──")

        def work(it):
            try:
                ev = evaluate_lean(client, it["chapters"], rubrics, CELL, calib, led,
                                   model=MODEL, extra_body=extra)
                return it, float(ev.get("weighted_total") or 0)
            except Exception as e:
                print(f"    {it['label'][:20]} → 失败 {type(e).__name__}")
                return it, 0.0

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            got = list(pool.map(work, items))
        rows = [{"kind": it["kind"], "label": it["label"], "rating": it["rating"],
                 "score": s} for it, s in got if s]
        results[cname] = rows
        ai = [r["score"] for r in rows if r["kind"] == "AI"]
        hu = [r["score"] for r in rows if r["kind"] == "真人"]
        print(f"    完成 {len(rows)}/{len(items)}　AI {statistics.mean(ai):.2f}"
              f"　真人 {statistics.mean(hu):.2f}")

    print(f"\n{'=' * 70}\n  结果\n{'=' * 70}")
    print(f"  {'条件':<8}{'推理/可见':>10}{'AI':>8}{'真人':>8}{'差距':>9}{'真人≥AI最低':>13}")
    gaps = {}
    for cname, _ in CONDS:
        rows = results[cname]
        led = ledgers[cname]
        ai = [r["score"] for r in rows if r["kind"] == "AI"]
        hu = [r["score"] for r in rows if r["kind"] == "真人"]
        vis = led.tout - led.treason
        ratio = led.treason / vis if vis > 0 else 0.0
        gap = statistics.mean(ai) - statistics.mean(hu)
        gaps[cname] = gap
        above = sum(1 for h in hu if h >= min(ai))
        print(f"  {cname:<8}{ratio:>10.2f}{statistics.mean(ai):>8.2f}"
              f"{statistics.mean(hu):>8.2f}{gap:>+9.2f}{f'{above}/{len(hu)}':>13}")

    off, on = gaps[CONDS[0][0]], gaps[CONDS[1][0]]
    delta = on - off
    print(f"\n  差距变化：{off:+.2f} → {on:+.2f}　（{delta:+.2f}）")
    print("\n  判读：")
    # 原判读用 on < off * 0.5 判断"缩小"，在 off 为负时逻辑反转
    # （-1.16 < -0.66 恒真），会把"几乎没变"误报成"明显缩小"。
    # 改为比较**绝对值**，并单独看方向是否符合假设。
    shrunk = abs(on) < abs(off) * 0.5
    if abs(delta) < 0.25:
        print(f"    差距几乎未变（{delta:+.2f}）—— 同模型内加大推理**没有**复现跨模型的单调关系。")
        print("    那三个点的差异更可能来自模型身份（谱系/语料/对齐），而非推理强度。")
        print("    原假设不成立，这同样是可写的结论。")
    elif shrunk and abs(on) < abs(off):
        print("    开思考后差距绝对值明显缩小 —— 推理强度是因，单调关系在同模型内复现。")
    else:
        print(f"    差距有变化但未缩小（|{off:+.2f}| → |{on:+.2f}|）—— 推理不是主因。")
    print(f"\n  ⚠ 操纵幅度有限：本实验推理/可见只到 {max(l.treason / max(1, l.tout - l.treason) for l in ledgers.values()):.2f}，"
          f"\n    远低于跨模型对照里 deepseek 的 4.96。窄幅操纵下的零结果不能完全排除强推理的影响。")

    # 与真人读者评分的相关，两组各算一次
    print(f"\n  与读者评分的相关（仅真人书）：")
    for cname, _ in CONDS:
        real = [r for r in results[cname] if r["kind"] == "真人" and r["rating"]]
        if len(real) >= 8:
            print(f"    {cname}  n={len(real)}  r = "
                  f"{pearson([r['score'] for r in real], [r['rating'] for r in real]):+.3f}")

    out = BASE / "reasoning_effect.json"
    out.write_text(json.dumps({"model": MODEL, "cell": CELL, "results": results,
                               "gaps": gaps}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n  已保存：{out}")
    for cname, _ in CONDS:
        print(f"\n  [{cname}] {ledgers[cname].report()}")


if __name__ == "__main__":
    main()
