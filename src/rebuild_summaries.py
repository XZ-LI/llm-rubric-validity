#!/usr/bin/env python3
"""
rebuild_summaries.py — 校对并重建各品类的 summary.json

问题来源：
  西方奇幻的 summary.json 记着 5 轮（7.5, 7.3, 7.5, 7.5, 7.5），
  但 iter_1..10/evaluation.json 记着 10 轮（7.0, 7.3, 7.4, 7.0, 7.4, …）。
  另外三个品类逐轮完全吻合。判断：summary.json 是一次 5 轮旧跑的残留，
  iter_* 目录已被后来的 10 轮跑静默覆盖。

口径裁定：
  以 iter_N/evaluation.json 为准。理由是它和该轮的 system_prompt.txt、
  novel.txt 存在同一个目录里，三者互相印证；summary.json 只是聚合结果，
  没有任何东西能证明它对应哪一次跑。

本脚本做三件事：
  1. 逐品类比对 summary.json 与 iter 目录，报出所有不一致
  2. 把有冲突的旧 summary.json 改名留档（不删除）
  3. 从 iter 目录重建 summary.json，并写入 run 元数据说明重建原因

用法：
  python3 rebuild_summaries.py            # 只检查，不改动
  python3 rebuild_summaries.py --apply    # 执行重建
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

BASE = Path("review_loop")


def read_iters(cat_dir: Path) -> list[dict]:
    out = []
    for d in sorted(cat_dir.glob("iter_*"), key=lambda p: int(p.name.split("_")[1])):
        ev = d / "evaluation.json"
        if not ev.exists():
            continue
        try:
            total = json.loads(ev.read_text(encoding="utf-8")).get("weighted_total")
        except json.JSONDecodeError:
            print(f"    [警告] {ev} 无法解析")
            continue
        if total is None:
            continue
        sp = d / "system_prompt.txt"
        out.append({
            "iteration": int(d.name.split("_")[1]),
            "score": total,
            "system_prompt_bytes": sp.stat().st_size if sp.exists() else None,
            "has_novel": (d / "novel.txt").exists(),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="执行重建；不加则只检查")
    args = ap.parse_args()

    conflicts, clean, rebuilt = [], [], []

    for cat_dir in sorted(p for p in BASE.iterdir() if p.is_dir()):
        iters = read_iters(cat_dir)
        if not iters:
            continue
        cat = cat_dir.name
        from_iters = [it["score"] for it in iters]

        sm_path = cat_dir / "summary.json"
        from_summary = None
        if sm_path.exists():
            try:
                sm = json.loads(sm_path.read_text(encoding="utf-8"))
                from_summary = [s.get("score") for s in sm.get("scores", [])]
            except json.JSONDecodeError:
                pass

        match = from_summary is not None and from_summary == from_iters
        if from_summary is None:
            status = "无 summary"
        elif match:
            status = "一致"
            clean.append(cat)
        else:
            status = "冲突"
            conflicts.append(cat)

        print(f"\n  {cat}  [{status}]  iter 目录 {len(iters)} 轮")
        if status == "冲突":
            print(f"    summary.json : {from_summary}")
            print(f"    iter 目录     : {from_iters}")

        if args.apply and status in ("冲突", "无 summary"):
            now = datetime.now(timezone.utc)
            if sm_path.exists():
                archived = cat_dir / f"summary_stale_{now.strftime('%Y%m%dT%H%M%SZ')}.json"
                sm_path.rename(archived)
                print(f"    旧文件留档 → {archived.name}")
            payload = {
                "category": cat,
                "rebuilt_utc": now.isoformat(timespec="seconds"),
                "rebuilt_from": "iter_*/evaluation.json",
                "rebuilt_reason": ("原 summary.json 与 iter 目录不一致，判定为旧跑残留；"
                                   "以 iter 目录为准重建" if status == "冲突" else "原先缺少 summary.json"),
                "total_rounds": len(iters),
                "scores": [{"iteration": it["iteration"], "score": it["score"]} for it in iters],
                "score_trend": from_iters,
                "system_prompt_bytes": [it["system_prompt_bytes"] for it in iters],
                "best_iteration": max(iters, key=lambda it: it["score"])["iteration"],
                "best_score": max(from_iters),
                "goal_reached": max(from_iters) >= 9.0,
            }
            sm_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            rebuilt.append(cat)
            print("    已重建 summary.json（含重建原因与提示体量）")

    print(f"\n{'=' * 62}")
    print(f"  一致 {len(clean)} 个：{'、'.join(clean) or '—'}")
    print(f"  冲突 {len(conflicts)} 个：{'、'.join(conflicts) or '—'}")
    if args.apply:
        print(f"  已重建 {len(rebuilt)} 个：{'、'.join(rebuilt) or '—'}")
    else:
        print("\n  这是检查模式。加 --apply 执行重建（旧文件会改名留档，不删除）。")


if __name__ == "__main__":
    main()
