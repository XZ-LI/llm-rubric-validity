#!/usr/bin/env python3
"""
contamination.py — 按格计算「被流程碰过」的书，作为唯一的污染排除口径

问题：排除集原先是硬编码的 top-5 ∪ bottom-2，但管线实际接触的书不止这些：

    环节                      取样        进入哪一端
    ───────────────────────────────────────────────
    归纳评分表                 top 5       评审端
    评审校准锚点               top 3 / 末 2  评审端
    风格标杆（写作提示）        top 3       生成端
    脑洞机制指南               top 12      生成端  ← 泄漏点

  derive_creativity_guide 用 top 12，而排除集只有 top 5，于是 top 6–12
  既喂了写手又留在留出集里。该函数有 BRAINSTORM_CATEGORIES 门控，
  跑过循环的七个品类中只有古言脑洞命中，故泄漏实际范围为一格。

为什么不能"所有格子一律排 top-12"：
  那是过度校正。多排高分书等于砍掉效标上端方差，会**机械地**压低相关
  （实测 top-5→top-12→top-15 时 r 从 .136 降到 .093 再到 .088，
  而 n 从 780 掉到 435、可检出下限从 .070 抬到 .094）。
  污染与范围限制就此混在一起，无法分辨。

正确做法：逐格取**该格实际被碰过的并集**。没跑过循环的 30 个格子
从未有写手介入，其排除集就是 top-5 ∪ bottom-2；跑过且属脑洞类的格子
才追加 top-12。

用法：
  from contamination import excluded_files
  drop = excluded_files(cell_dir_name, category, source_dir)
"""
from __future__ import annotations

import re
from pathlib import Path

BASE = Path("review_loop")

TOP_RUBRIC = 5      # review_loop.TOP_SOURCE_NOVELS
ANCHOR_LOW = 2      # review_loop.CALIBRATION_LOW_N
TOP_CREATIVITY = 12  # review_loop.CREATIVITY_SOURCE_N


def rated_desc(category: str, source_dir: str | Path) -> list[tuple[float, str]]:
    """该品类所有有效评分的书，按评分降序。评分为 0 视为缺失。"""
    out = []
    for f in sorted((Path(source_dir) / category).glob("*.txt")):
        m = re.search(r"^评分[：:]\s*(.+)$",
                      f.read_text(encoding="utf-8", errors="replace")[:2000],
                      re.MULTILINE)
        if not m:
            continue
        try:
            v = float(m.group(1))
        except ValueError:
            continue
        if v > 0:
            out.append((v, f.name))
    out.sort(reverse=True)
    return out


def touched_creativity(cell: str) -> bool:
    """该格是否真的生成过创意指南。以文件存在为准，而非按品类名推断——
    门控逻辑将来可能改，但产物文件是既成事实。"""
    return (BASE / cell / "creativity_guide.md").exists()


def excluded_files(cell: str, category: str, source_dir: str | Path) -> set[str]:
    """该格应排除的文件名集合。"""
    rated = rated_desc(category, source_dir)
    drop = {n for _, n in rated[:TOP_RUBRIC]}          # 评分表来源
    drop |= {n for _, n in rated[-ANCHOR_LOW:]}        # 低分锚点
    if touched_creativity(cell):                       # 仅此类格子追加
        drop |= {n for _, n in rated[:TOP_CREATIVITY]}
    return drop


def report() -> None:
    import json
    print(f"\n{'=' * 68}\n  按格污染排除口径\n{'=' * 68}")
    print(f"  {'格子':<20}{'有评分':>7}{'原排除':>8}{'应排除':>8}{'追加':>6}{'创意指南':>10}")
    tot_old = tot_new = 0
    for p in sorted(BASE.glob("*/validation_pilot.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("reviewer_model") != "qwen3-max":
            continue
        cell = p.parent.name
        rated = rated_desc(d["category"], d["source_dir"])
        old = {n for _, n in rated[:TOP_RUBRIC]} | {n for _, n in rated[-ANCHOR_LOW:]}
        new = excluded_files(cell, d["category"], d["source_dir"])
        tot_old += len(old)
        tot_new += len(new)
        if new != old:
            print(f"  {cell:<20}{len(rated):>7}{len(old):>8}{len(new):>8}"
                  f"{len(new - old):>6}{'有' if touched_creativity(cell) else '':>10}")
    print(f"\n  仅上列格子需要调整；其余格子排除集不变。")
    print(f"  全体排除总数 {tot_old} → {tot_new}（+{tot_new - tot_old}）")


if __name__ == "__main__":
    report()
