#!/usr/bin/env python3
"""
make_public.py — 生成可公开发布的数据副本（剥离受版权保护的正文）

背景：结果文件看起来只是分数，但 evidence 字段里存着评审打分时直接引用的
小说原文——全部 37 格合计 25,233 条片段、约 77.3 万字。这些不能上传。

本脚本产出 public/ 目录，只保留研究所需的派生数据：
  · 分数、维度分、真实读者评分、平台行为指标
  · 书名 / 作者 / book_id（事实性元数据，非表达性内容）
  · 引文**长度**而非引文本身（保留"评审举证了多少"这一信息）

不生成、也不复制：语料正文、标注工具、任何含正文的中间产物。

复现路径：公开仓库给出 book_id 清单与抓取脚本，他人可自行获取语料，
而不是由本仓库转发作品原文。

用法：
  python3 make_public.py
  python3 make_public.py --check     # 只扫描，报告哪些文件含正文
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

BASE = Path("review_loop")
OUT = Path("public")

# 逐字正文可能藏身的字段。
# ⚠ 这份名单是黑名单，必然不完整——首次运行时就漏掉了 rubrics.json 里的
#   high_score_example / low_score_example（引用小说正文并带书名），
#   820 处、险些随仓库公开。名单之外还有 verify() 做内容层兜底。
TEXT_FIELDS = {"evidence", "failures", "success", "text", "left", "right",
               "chapters", "ch1_text", "opening", "abstract", "premise",
               "real_desc", "high_score_example", "low_score_example",
               "rewrite_example", "strengths", "top_issues", "suggestion",
               "raw", "problem", "prompt_fix"}

# 允许保留长中文的字段：它们是研究产物本身（评分标准的定义、说明），
# 不是作品原文。除此之外，任何 ≥ PROSE_CHARS 汉字的串一律视为疑似正文。
ALLOW_LONG = {"description", "note", "verified", "rebuilt_reason", "measured"}
PROSE_CHARS = 40

# 整份都含正文、一律不发布
NEVER = ("annotation/", "Nanpin/", "Nvpin/", "novels/", "generated/",
         "prompts/", "converted_txt/", "pipeline")


def strip(obj):
    """递归剥离正文字段；对引文只保留长度。"""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "evidence" and isinstance(v, dict):
                # 保留"每个维度举了几条证、各多长"，丢掉证据原文
                out["evidence_shape"] = {
                    dim: {"n_failures": len(e.get("failures") or []),
                          "failure_chars": [len(str(x)) for x in (e.get("failures") or [])],
                          "success_chars": len(str(e.get("success") or ""))}
                    for dim, e in v.items() if isinstance(e, dict)}
            elif k in TEXT_FIELDS:
                continue
            else:
                out[k] = strip(v)
        return out
    if isinstance(obj, list):
        return [strip(x) for x in obj]
    return obj


def scan_for_prose(path: Path) -> tuple[int, int]:
    """→ (含正文字段数, 正文总字数)。用于 --check。"""
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0, 0
    n = chars = 0

    def walk(o):
        nonlocal n, chars
        if isinstance(o, dict):
            for k, v in o.items():
                if k in TEXT_FIELDS:
                    n += 1
                    chars += len(json.dumps(v, ensure_ascii=False))
                else:
                    walk(v)
        elif isinstance(o, list):
            for x in o:
                walk(x)
    walk(d)
    return n, chars


def verify(root: Path) -> list[tuple]:
    """内容层兜底：扫出任何仍含长中文自由文本的字段。

    不依赖字段名——名单漏一个就会把作品原文发出去。这里按内容判断，
    发现即中止，宁可不发布也不误发。"""
    hits = []

    def walk(o, path, f):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ALLOW_LONG:
                    continue
                walk(v, f"{path}.{k}", f)
        elif isinstance(o, list):
            for i, x in enumerate(o):
                walk(x, f"{path}[{i}]", f)
        elif isinstance(o, str):
            cn = sum(1 for c in o if "\u4e00" <= c <= "\u9fff")
            if cn >= PROSE_CHARS:
                hits.append((f.name, path, cn))

    for p in root.rglob("*.json"):
        try:
            walk(json.loads(p.read_text(encoding="utf-8")), p.stem, p)
        except Exception:
            pass
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    targets = sorted(BASE.rglob("*.json"))
    if args.check:
        print(f"\n{'=' * 66}\n  扫描：哪些 JSON 含逐字正文\n{'=' * 66}")
        tot_n = tot_c = 0
        for p in targets:
            n, c = scan_for_prose(p)
            if n:
                print(f"  {str(p):<52}{n:>7} 处 {c/10000:>7.1f} 万字")
                tot_n += n
                tot_c += c
        print(f"\n  合计 {tot_n:,} 处，约 {tot_c/10000:.1f} 万字 —— 这些必须剥离后才能发布。")
        print("  另有整目录不可发布：" + "、".join(NEVER))
        return

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "results").mkdir(parents=True)

    kept = stripped = 0
    for p in targets:
        if any(s in str(p) for s in NEVER):
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        clean = strip(d)
        rel = p.relative_to(BASE)
        dst = OUT / "results" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
        kept += 1
        before, after = len(p.read_bytes()), len(dst.read_bytes())
        stripped += before - after

    # 语料清单：只给 book_id 与事实性元数据，供他人自行抓取
    manifest = []
    for p in sorted(BASE.glob("*/validation_pilot.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("reviewer_model") != "qwen3-max":
            continue
        for r in d["results"]:
            f = Path(d["source_dir"]) / d["category"] / r["file"]
            bid = ""
            if f.exists():
                m = re.search(r"book_id=(\d+)",
                              f.read_text(encoding="utf-8", errors="replace")[:1500])
                bid = m.group(1) if m else ""
            manifest.append({"cell": p.parent.name, "category": d["category"],
                             "channel": d["source_dir"], "book_id": bid,
                             "title": r.get("title", ""), "rating": r["real_rating"]})
    (OUT / "corpus_manifest.json").write_text(
        json.dumps({"n": len(manifest),
                    "note": "书目清单，供复现者自行获取语料；本仓库不分发作品正文。",
                    "books": manifest}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'=' * 66}\n  已生成 public/\n{'=' * 66}")
    print(f"  结果文件 {kept} 份，剥离正文约 {stripped/1048576:.1f} MB")
    print(f"  语料清单 {len(manifest)} 条（book_id + 事实性元数据）")
    hits = verify(OUT)
    if hits:
        print(f"\n  ✗ 中止：public/ 仍有 {len(hits)} 处疑似正文")
        for f, path, cn in hits[:6]:
            print(f"      {f} · {path} · {cn} 字")
        shutil.rmtree(OUT)
        raise SystemExit("  已删除 public/。把上列字段加入 TEXT_FIELDS 后重跑。")
    print(f"\n  ✓ 内容层核验通过：无 ≥{PROSE_CHARS} 汉字的自由文本")
    print(f"  发布前请再跑一次核验：python3 make_public.py --check")
    print(f"  并确认 .gitignore 覆盖：" + "、".join(NEVER))


if __name__ == "__main__":
    main()
