#!/usr/bin/env python3
"""
crossjudge.py — 换一个评审模型，重打同一批文本。

要回答的问题（试水实验留下的悬念）：
  o3 给 AI 自己的产出打 7.03，给真人已出版小说打 5.72，差 +1.31 分。
  这 +1.31 有两种可能的来源，目前分不开：

    (1) 自我偏好 —— o3 认得出自己的文风并偏爱它
    (2) 评分表形状 —— 评分表从"第一章开头"归纳，天然奖励密集钩子和干净句子；
        AI 最擅长堆这些，而真人连载第一章常常慢热

  换一个完全不同谱系的评审（DeepSeek / 通义，中文母语、不同预训练与对齐）
  重打同一批文本：
    差距缩小  → 是 (1) 自我偏好
    差距还在  → 是 (2) 评分表本身的形状问题

  两种结果都成立，都是可写的结论。

评分口径与 validate_pilot.py 完全一致（同一份 rubrics、同一段校准锚点、
同一个精简 prompt），唯一变量是评审模型本身。

用法：
  python3 crossjudge.py --dry-run                    # 先验证文本抽取，不调 API
  python3 crossjudge.py --provider deepseek
  python3 crossjudge.py --provider qwen --model qwen3-max
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from review_loop import format_calibration_block, load_calibration_anchors  # noqa: E402
from validate_pilot import Ledger, evaluate_lean  # noqa: E402
from validate_rubric import CHAPTERS_TO_SCORE, pearson, spearman, split_chapters  # noqa: E402

PROVIDERS = {
    "openai":   ("https://api.openai.com/v1",                         "OPENAI_API_KEY",    "o3"),
    "deepseek": ("https://api.deepseek.com",                          "DEEPSEEK_API_KEY",  "deepseek-v4-pro"),
    "qwen":     ("https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY", "qwen3-max"),
}

CELLS = [("民国言情", "Nvpin"), ("动漫衍生", "Nanpin")]
AI_CELL = "动漫衍生"        # 唯一同时有 AI 十轮产出和真人留出集的品类
AI_ITERS = range(1, 11)

# AI 产出文件里，正文前有「投稿信息头 + 逐章大纲」，大纲行同样长得像"第N章"。
# 直接套 split_chapters 会把大纲误当正文，只喂给评审几十个字。
# 正文前有一条长分隔线（≥40 个 =），取最后一条之后的内容才是真正文。
LONG_SEP = re.compile(r"^={40,}\s*$", re.MULTILINE)


def extract_ai_body(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    seps = list(LONG_SEP.finditer(text))
    return text[seps[-1].end():] if seps else text


def load_real(category: str, source_dir: Path) -> list[dict]:
    """复用试水实验已确定的留出集，保证两次评的是同一批书。"""
    pilot = Path("review_loop") / category / "validation_pilot.json"
    if not pilot.exists():
        sys.exit(f"[ERROR] 缺少 {pilot}，先跑 validate_pilot.py")
    out = []
    for r in json.loads(pilot.read_text(encoding="utf-8"))["results"]:
        p = source_dir / category / r["file"]
        if not p.exists():
            continue
        body = p.read_text(encoding="utf-8", errors="replace").split("=" * 20, 1)[-1]
        chs = split_chapters(body, CHAPTERS_TO_SCORE)
        if chs:
            out.append({"kind": "真人", "label": r["title"], "chapters": chs,
                        "real_rating": r["real_rating"], "o3_score": r["rubric_score"]})
    return out


def load_ai() -> list[dict]:
    out = []
    for i in AI_ITERS:
        d = Path("review_loop") / AI_CELL / f"iter_{i}"
        nv, ev = d / "novel.txt", d / "evaluation.json"
        if not nv.exists():
            continue
        chs = split_chapters(extract_ai_body(nv), CHAPTERS_TO_SCORE)
        if not chs:
            continue
        o3 = json.loads(ev.read_text(encoding="utf-8")).get("weighted_total") if ev.exists() else None
        out.append({"kind": "AI", "label": f"iter_{i}", "chapters": chs,
                    "real_rating": None, "o3_score": o3})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="deepseek", choices=list(PROVIDERS))
    ap.add_argument("--model", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base, env_key, default_model = PROVIDERS[args.provider]
    model = args.model or default_model

    tasks = {}
    for cat, src in CELLS:
        items = load_real(cat, Path(src))
        if cat == AI_CELL:
            items = load_ai() + items
        tasks[(cat, src)] = items

    if args.dry_run:
        print(f"\n{'=' * 70}\n  抽取检查（未调 API）\n{'=' * 70}")
        for (cat, _), items in tasks.items():
            n_ai = sum(1 for i in items if i["kind"] == "AI")
            print(f"\n  {cat}：AI {n_ai} 份 + 真人 {len(items) - n_ai} 部 = {len(items)} 次调用")
            for it in items[:2] + ([items[n_ai]] if n_ai and len(items) > n_ai else []):
                chars = [len(c) for c in it["chapters"]]
                head = it["chapters"][0][:52].replace("\n", " ")
                print(f"    [{it['kind']}] {it['label'][:22]:<24} 章节字数 {chars}")
                print(f"          首句：{head}…")
        print(f"\n  合计 {sum(len(v) for v in tasks.values())} 次调用")
        print("  ↑ 请确认 AI 与真人两边的章节字数量级相当（都应是几千字，而非几十字）")
        return

    key = os.environ.get(env_key)
    if not key:
        sys.exit(f"[ERROR] 未设置 {env_key}")
    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=base)

    ledger = Ledger(model)
    print(f"\n{'=' * 70}\n  交叉评审：{args.provider} / {model}\n{'=' * 70}")

    all_rows = []
    for (cat, src), items in tasks.items():
        rubrics = json.loads((Path("review_loop") / cat / "rubrics.json").read_text(encoding="utf-8"))
        calib = format_calibration_block(load_calibration_anchors(cat, source_dir=Path(src)))
        print(f"\n  ── {cat} ──")
        for i, it in enumerate(items, 1):
            print(f"  [{i}/{len(items)}] {it['kind']:<3} {it['label'][:24]:<26}", end="", flush=True)
            try:
                ev = evaluate_lean(client, it["chapters"], rubrics, cat, calib, ledger, model=model)
                score = float(ev.get("weighted_total") or 0)
            except Exception as e:
                print(f" → 失败 {type(e).__name__} {str(e)[:60]}")
                continue
            if not score:
                print(" → 解析失败")
                continue
            print(f" → {score}")
            all_rows.append({**{k: v for k, v in it.items() if k != "chapters"},
                             "category": cat, "new_score": score})

    out = Path("review_loop") / f"crossjudge_{args.provider}.json"
    out.write_text(json.dumps({"provider": args.provider, "model": model, "rows": all_rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'=' * 70}\n  结果\n{'=' * 70}")

    # 1) 与真实读者评分的相关（只用真人书）
    for cat, _ in CELLS:
        real = [r for r in all_rows if r["category"] == cat and r["kind"] == "真人"]
        if len(real) < 3:
            continue
        rr = [r["real_rating"] for r in real]
        new = [r["new_score"] for r in real]
        old = [r["o3_score"] for r in real]
        print(f"\n  {cat}  n={len(real)}")
        print(f"    {model:<18} vs 读者评分   r = {pearson(new, rr):+.3f}   ρ = {spearman(new, rr):+.3f}")
        print(f"    {'o3（已有）':<18} vs 读者评分   r = {pearson(old, rr):+.3f}   ρ = {spearman(old, rr):+.3f}")
        print(f"    两个评审彼此的一致性                    r = {pearson(new, old):+.3f}")

    # 2) 核心诊断：AI 产出 vs 真人小说的差距
    ai = [r["new_score"] for r in all_rows if r["kind"] == "AI"]
    hu = [r["new_score"] for r in all_rows if r["kind"] == "真人" and r["category"] == AI_CELL]
    if ai and hu:
        ai_o3 = [r["o3_score"] for r in all_rows if r["kind"] == "AI" and r["o3_score"]]
        hu_o3 = [r["o3_score"] for r in all_rows if r["kind"] == "真人" and r["category"] == AI_CELL]
        gap_new = statistics.mean(ai) - statistics.mean(hu)
        gap_o3 = statistics.mean(ai_o3) - statistics.mean(hu_o3)
        print(f"\n{'─' * 70}\n  核心诊断（{AI_CELL}）：评审是否偏爱 AI 的产出\n{'─' * 70}")
        print(f"    {'评审':<20}{'AI 十轮':>10}{'真人 22 部':>12}{'差距':>10}")
        print(f"    {'o3（原实验）':<20}{statistics.mean(ai_o3):>10.2f}{statistics.mean(hu_o3):>12.2f}{gap_o3:>+10.2f}")
        print(f"    {model:<20}{statistics.mean(ai):>10.2f}{statistics.mean(hu):>12.2f}{gap_new:>+10.2f}")
        above = sum(1 for h in hu if h >= min(ai))
        print(f"\n    给分 ≥ AI 最差一轮({min(ai)}) 的真人小说：{above}/{len(hu)} 部（o3 那边是 0/22）")
        print("\n    判读：")
        if gap_o3 and gap_new < gap_o3 * 0.4:
            print(f"      差距从 {gap_o3:+.2f} 收窄到 {gap_new:+.2f} —— 主要是 o3 的**自我偏好**。")
        elif gap_new >= gap_o3 * 0.8:
            print(f"      差距基本不变（{gap_o3:+.2f} → {gap_new:+.2f}）—— 不是自我偏好，")
            print("      是**评分表本身的形状**在系统性奖励 AI 式文本。这是更强的结论。")
        else:
            print(f"      差距部分收窄（{gap_o3:+.2f} → {gap_new:+.2f}）—— 两种机制都有份，需第三个评审定量。")

    print(f"\n{'=' * 70}\n  用量\n{'=' * 70}")
    print(ledger.report())
    print(f"\n  已保存：{out}")


if __name__ == "__main__":
    main()
