#!/usr/bin/env python3
"""
fixed_rubric_run.py — RQ4：用**同一套十维**重打全部留出书

为什么必须重跑：37 个格子原先各自归纳评分表，产生 208 个不同维度名，
其中 145 个只出现在一个格子。维度身份与格子身份几乎完全共线——
「反差营造 +0.432」实际只反映「都市日常这个格子相关高」。
在那种数据上做维度级分析，测的是格子不是维度。

为什么现在非做不可：RQ6 发现评分表与读者在**多样性**上是一致的
（评分表↔多样性 +0.217，多样性↔收藏率 +0.180），可总体效度只有 0.136。
问题因此变成：**评分表在它唯一对齐的维度之外，是哪些维度把它拖回零？**
只有跨格可比的维度分数才能回答。

设计：
  · 维度取 review_loop.SEED_RUBRICS 的原始十维——它是整个循环的起点，
    体裁中立，工艺类与新颖性类都覆盖。
  · 评分说明由人写定、全局统一、权重一律为 1。不让机器各自命名或加权，
    以消除"维度定义随格子漂移"这个自由度。
  · 评审与主实验一致（qwen3-max），口径一致（同一精简 prompt）。
  · 结果写入 fixed_rubric.json，**不触碰** validation_pilot.json。

成本：840 次调用 × ¥0.0309 ≈ ¥26。

用法：
  python3 fixed_rubric_run.py --dry-run
  python3 fixed_rubric_run.py --limit 20      # 先小样本
  python3 fixed_rubric_run.py                 # 全量（自动续跑）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from review_loop import format_calibration_block, load_calibration_anchors  # noqa: E402
from validate_pilot import Ledger, evaluate_lean  # noqa: E402
from validate_rubric import CHAPTERS_TO_SCORE, split_chapters  # noqa: E402

BASE = Path("review_loop")
OUT = BASE / "fixed_rubric.json"
MODEL = "qwen3-max"
SAVE_EVERY = 40

# 全局固定评分表。名称取自 review_loop.SEED_RUBRICS，说明由人写定。
# 权重一律为 1：加权会把"哪一维重要"这个判断偷偷塞进结果，
# 而本实验要问的恰恰是各维度各自与读者行为的关系。
FIXED_RUBRIC = [
    {"name": "钩子强度", "weight": 1,
     "description": "开篇能否在前几句内抛出使人想继续读下去的问题、危机或反常。"},
    {"name": "核心矛盾", "weight": 1,
     "description": "是否存在清晰、具体、会升级的对立；冲突是否有真实的利害关系。"},
    {"name": "节奏变化", "weight": 1,
     "description": "推进快慢是否有张弛；句长与场景密度是否有变化而非匀速。"},
    {"name": "人物鲜活度", "weight": 1,
     "description": "角色是否有可辨识的声音与反应方式；行为是否由性格而非情节需要驱动。"},
    {"name": "画面感", "weight": 1,
     "description": "描写能否让读者形成具体的感官画面；细节是否可感而非抽象概括。"},
    {"name": "对话自然度", "weight": 1,
     "description": "对白是否像真人在这个处境下会说的话；是否承担推进与塑形的功能。"},
    {"name": "悬念设计", "weight": 1,
     "description": "是否留下有效的未解问题；章节收束是否促使读者继续。"},
    {"name": "情感感染力", "weight": 1,
     "description": "是否让读者产生具体的情绪反应，而非仅被告知角色有情绪。"},
    {"name": "世界观融合", "weight": 1,
     "description": "设定是否随情节自然交代；有无信息倾倒或设定与情节脱节。"},
    {"name": "戏剧性反转", "weight": 1,
     "description": "是否出现改变读者理解的意外；反转是否有铺垫而非凭空。"},
]

_lock = threading.Lock()


def targets() -> list[dict]:
    out = []
    for p in sorted(BASE.glob("*/validation_pilot.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("reviewer_model") != MODEL:
            continue
        for r in d["results"]:
            f = Path(d["source_dir"]) / d["category"] / r["file"]
            if not f.exists():
                continue
            out.append({"key": f"{p.parent.name}|{r['file']}",
                        "cell": p.parent.name, "category": d["category"],
                        "source_dir": d["source_dir"], "file": r["file"],
                        "rating": r["real_rating"], "orig_score": r["rubric_score"]})
    return out


def load_done() -> dict:
    if not OUT.exists():
        return {}
    try:
        return json.loads(OUT.read_text(encoding="utf-8")).get("books", {})
    except json.JSONDecodeError:
        return {}


def save(books: dict, total: int) -> None:
    OUT.write_text(json.dumps({
        "model": MODEL, "rubric": FIXED_RUBRIC,
        "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n": len(books), "target_total": total, "books": books,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tg = targets()
    done = load_done()
    todo = [t for t in tg if t["key"] not in done]
    if args.limit:
        todo = todo[:args.limit]

    print(f"\n{'=' * 70}\n  RQ4 固定十维重打\n{'=' * 70}")
    print(f"  目标 {len(tg)} 本；已完成 {len(done)}；本次待跑 {len(todo)}")
    print(f"  维度：{'、'.join(r['name'] for r in FIXED_RUBRIC)}")
    print(f"  评审 {MODEL}，权重一律为 1，预计 ¥{len(todo) * 0.0309:.2f}")
    if args.dry_run:
        print("  [dry-run] 未调 API。")
        return
    if not todo:
        print("  没有待跑的书。")
        return

    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise SystemExit("[ERROR] 未设置 DASHSCOPE_API_KEY")
    from openai import OpenAI
    client = OpenAI(api_key=key,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    ledger = Ledger(MODEL)

    # 校准锚点仍按品类取——锚点的作用是把分数尺度对齐到该品类的真实评分，
    # 与维度定义无关，换成全局锚点反而会让跨品类的分数尺度失真。
    calib_cache: dict[str, str] = {}

    def work(t: dict):
        f = Path(t["source_dir"]) / t["category"] / t["file"]
        body = f.read_text(encoding="utf-8", errors="replace").split("=" * 20, 1)[-1]
        chs = split_chapters(body, CHAPTERS_TO_SCORE)
        if not chs:
            return t, None
        with _lock:
            if t["cell"] not in calib_cache:
                calib_cache[t["cell"]] = format_calibration_block(
                    load_calibration_anchors(t["category"], source_dir=Path(t["source_dir"])))
            calib = calib_cache[t["cell"]]
        ev = evaluate_lean(client, chs, FIXED_RUBRIC, t["category"], calib,
                           ledger, model=MODEL)
        return t, ev

    t0 = time.time()
    n_ok = n_bad = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(work, t): t for t in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            t = futs[fut]
            try:
                t, ev = fut.result()
            except Exception as e:
                print(f"  [{i}/{len(todo)}] {t['file'][:22]} → 失败 {type(e).__name__}")
                n_bad += 1
                continue
            total = float((ev or {}).get("weighted_total") or 0)
            dims = {k: v.get("score") for k, v in (ev or {}).get("scores", {}).items()
                    if isinstance(v, dict)}
            if not total or len(dims) < 6:
                n_bad += 1
                continue
            with _lock:
                done[t["key"]] = {"cell": t["cell"], "file": t["file"],
                                  "rating": t["rating"], "orig_score": t["orig_score"],
                                  "fixed_score": total, "dims": dims}
                n_ok += 1
                if i % SAVE_EVERY == 0:
                    save(done, len(tg))
                    print(f"  {i}/{len(todo)}  ok={n_ok} 失败={n_bad}"
                          f"  {(time.time() - t0) / 60:.1f} 分", flush=True)

    save(done, len(tg))
    print(f"\n  完成 {n_ok}，失败 {n_bad}；累计 {len(done)}/{len(tg)}")
    print(f"  耗时 {(time.time() - t0) / 60:.1f} 分\n")
    print(ledger.report())
    print(f"\n  已保存：{OUT}")
    print("  下一步：python3 fixed_rubric_analyze.py")


if __name__ == "__main__":
    main()
