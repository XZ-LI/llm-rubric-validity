#!/usr/bin/env python3
"""
build_annotation.py — 生成本地盲标工具（A1：真人 vs 真人强制二选一）

检验什么：评分表在它**最有把握**的地方，能不能预测人的阅读选择。
若连高十分位 vs 低十分位都赢不了 50%，评分表的效度问题就彻底定案。

抽样为什么不能随机：840 本里随便抽两本配对，多数对子评分表分差很小，
标注者选谁都近乎抛硬币，最后得到一个漂亮的 50/50 而什么都没学到。
本脚本只在**同品类内**取评分表分数的高/低十分位配对。

盲化措施（全部内建，标注者无法绕过）：
  · 不显示书名、作者、评分、在读人数、评分表分数
  · 左右随机，高分侧不固定
  · 品类打乱交错，避免品类内锚定
  · 混入注意力检验题（一侧句序打乱），用于事后剔除低质量批次
  · 所有判断录完前不揭晓任何一侧的身份

输出为单个自包含 HTML，双击即可用；进度存 localStorage，可分次完成。
**不发布到网络**：正文为已出版作品，仅作本地研究使用。

用法：
  python3 build_annotation.py                 # 默认 120 对 + 6 道检验题
  python3 build_annotation.py --pairs 80
"""
from __future__ import annotations

import argparse
import html
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from contamination import excluded_files  # noqa: E402

BASE = Path("review_loop")
OUT = Path("annotation/annotate.html")
EXCERPT_CHARS = 2000        # 与评审看到的窗口一致
CHAPTER_RE = re.compile(r"^第\s*[〇零一二三四五六七八九十百千\d]+\s*[章节]\s*(.*)$", re.M)


def first_chapter(path: Path) -> str | None:
    """取第 1 章正文前 N 字。取不到章节标记就退回正文开头。"""
    body = path.read_text(encoding="utf-8", errors="replace").split("=" * 20, 1)[-1]
    marks = list(CHAPTER_RE.finditer(body))
    seg = body[marks[0].end():marks[1].start()] if len(marks) > 1 else body
    seg = re.sub(r"\n{3,}", "\n\n", seg.strip())
    return seg[:EXCERPT_CHARS] if len(seg) >= 600 else None


def scramble(text: str) -> str:
    """注意力检验题：打乱句序。语义不通但字词分布不变，
    正常阅读的人一眼可辨，走神的人会选错。"""
    sents = [s for s in re.split(r"(?<=[。！？…])", text) if s.strip()]
    if len(sents) < 6:
        return text
    mid = sents[1:-1]
    random.shuffle(mid)
    return "".join([sents[0]] + mid + [sents[-1]])


def collect() -> dict[str, list[dict]]:
    """按格收集可用文本，附评分表分数（用于配对，不展示）。"""
    cells: dict[str, list[dict]] = {}
    for p in sorted(BASE.glob("*/validation_pilot.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("reviewer_model") != "qwen3-max":
            continue
        cell = p.parent.name
        drop = excluded_files(cell, d["category"], d["source_dir"])
        items = []
        for r in d["results"]:
            if r["file"] in drop:
                continue
            f = Path(d["source_dir"]) / d["category"] / r["file"]
            if not f.exists():
                continue
            ex = first_chapter(f)
            if ex:
                items.append({"file": r["file"], "score": r["rubric_score"],
                              "rating": r["real_rating"], "text": ex})
        if len(items) >= 12:
            cells[cell] = items
    return cells


def make_pairs(cells: dict, n_pairs: int, n_checks: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    random.seed(seed)
    per = max(1, n_pairs // max(1, len(cells)))
    pairs = []
    for cell, items in cells.items():
        items = sorted(items, key=lambda x: x["score"])
        k = max(2, len(items) // 4)          # 各取四分之一作为高/低十分位近似
        low, high = items[:k], items[-k:]
        rng.shuffle(low)
        rng.shuffle(high)
        for i in range(min(per, len(low), len(high))):
            a, b = high[i], low[i]
            if a["score"] - b["score"] < 0.3:   # 分差过小的对子没有判别力
                continue
            flip = rng.random() < 0.5           # 左右随机
            pairs.append({
                "kind": "real", "cell": cell,
                "left": (b if flip else a)["text"],
                "right": (a if flip else b)["text"],
                "hi_side": "right" if flip else "left",   # 评分表偏好的一侧
                "hi_file": a["file"], "lo_file": b["file"],
                "gap": round(a["score"] - b["score"], 3),
            })
    rng.shuffle(pairs)
    pairs = pairs[:n_pairs]

    # 注意力检验题：同一段文本，一侧打乱句序
    pool = [it for items in cells.values() for it in items]
    rng.shuffle(pool)
    checks = []
    for it in pool[:n_checks]:
        flip = rng.random() < 0.5
        bad = scramble(it["text"])
        checks.append({
            "kind": "check", "cell": "—",
            "left": (bad if flip else it["text"]),
            "right": (it["text"] if flip else bad),
            "good_side": "right" if flip else "left",
        })

    allp = pairs + checks
    rng.shuffle(allp)                        # 检验题均匀混入
    for i, p in enumerate(allp):
        p["idx"] = i
    return allp


TEMPLATE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>盲标：你更想接着读哪一个</title>
<style>
:root{--bg:#F3F4F6;--card:#fff;--ink:#16181C;--ink2:#4C525B;--muted:#858B94;
--line:#DCE0E5;--pick:#0E6B5C;--pick2:#CFE6E0;--warn:#8A5A12}
@media(prefers-color-scheme:dark){:root{--bg:#0F1114;--card:#171A1F;--ink:#E8EAED;
--ink2:#AEB4BC;--muted:#7D848D;--line:#2A2F36;--pick:#5FB9A4;--pick2:#17332D;--warn:#D6AE5B}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);padding:0 16px 60px;
font:16px/1.85 "PingFang SC","Hiragino Sans GB","Microsoft YaHei",system-ui,sans-serif}
.wrap{max-width:1080px;margin:0 auto}
header{position:sticky;top:0;background:var(--bg);padding:14px 0 10px;z-index:5;
border-bottom:1px solid var(--line)}
.bar{height:5px;background:var(--line);border-radius:3px;overflow:hidden;margin-top:9px}
.bar i{display:block;height:100%;background:var(--pick);width:0;transition:width .2s}
.meta{display:flex;justify-content:space-between;font-size:.82rem;color:var(--muted);
font-variant-numeric:tabular-nums}
.q{text-align:center;font-size:1.12rem;font-weight:600;margin:26px 0 16px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:820px){.cols{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:9px;
padding:20px 22px;white-space:pre-wrap;font-size:15.5px;max-height:62vh;overflow-y:auto}
.btns{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:14px}
@media(max-width:820px){.btns{grid-template-columns:1fr}}
button{font:inherit;font-weight:600;padding:13px;border-radius:8px;cursor:pointer;
border:1px solid var(--pick);background:var(--card);color:var(--pick)}
button:hover{background:var(--pick2)}
button:focus-visible{outline:3px solid var(--pick);outline-offset:2px}
.note{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--warn);
border-radius:0 8px 8px 0;padding:14px 16px;margin:18px 0;font-size:.9rem;color:var(--ink2)}
.done{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:26px}
textarea{width:100%;font:inherit;padding:10px;border:1px solid var(--line);
border-radius:7px;background:var(--bg);color:var(--ink);min-height:74px}
kbd{font-family:ui-monospace,monospace;background:var(--bg);border:1px solid var(--line);
border-radius:4px;padding:1px 6px;font-size:.85em}
</style></head><body><div class="wrap">
<header><div class="meta"><span id="pos"></span><span id="save">进度自动保存</span></div>
<div class="bar"><i id="fill"></i></div></header>
<div id="app"></div></div>
<script>
const PAIRS = __PAIRS__;
const KEY = "annot_v1";
let S = JSON.parse(localStorage.getItem(KEY) || '{"i":0,"ans":[],"prereg":null,"t0":null}');
const $ = id => document.getElementById(id);
const save = () => localStorage.setItem(KEY, JSON.stringify(S));

function prereg(){
  $("app").innerHTML = `<div class="done">
    <h2 style="margin-top:0">开始前：先写下你的预期</h2>
    <p style="color:var(--ink2)">这是单人标注唯一能部分抵消实验者偏差的办法。写完封存，标完再对照。</p>
    <p style="color:var(--ink2);font-size:.92rem">评分表给一侧的分明显更高。你预计自己会有多大比例选中<strong>评分表偏好的那一侧</strong>？（50% = 完全无关）</p>
    <textarea id="pr" placeholder="例：我预计 55% 左右，因为……"></textarea>
    <div style="margin-top:14px"><button onclick="startNow()">封存并开始</button></div>
    <p style="color:var(--muted);font-size:.85rem;margin-top:16px">
      共 ${PAIRS.length} 对，其中混有注意力检验题。快捷键 <kbd>F</kbd> 选左、<kbd>J</kbd> 选右。</p>
  </div>`;
}
window.startNow = () => { S.prereg = $("pr").value.trim(); S.t0 = Date.now(); save(); render(); };

function render(){
  if (S.prereg === null) return prereg();
  if (S.i >= PAIRS.length) return finish();
  const p = PAIRS[S.i];
  $("pos").textContent = `第 ${S.i+1} / ${PAIRS.length} 对`;
  $("fill").style.width = (S.i / PAIRS.length * 100) + "%";
  $("app").innerHTML = `<div class="q">这两个开头，你更想接着读哪一个？</div>
    <div class="cols">
      <div class="card">${esc(p.left)}</div>
      <div class="card">${esc(p.right)}</div>
    </div>
    <div class="btns">
      <button onclick="pick('left')">选左边　<kbd>F</kbd></button>
      <button onclick="pick('right')">选右边　<kbd>J</kbd></button>
    </div>
    <div class="note">只凭这两段判断。不必找“更好”的那个，找<strong>你更想继续读</strong>的那个。</div>`;
  window.scrollTo(0,0);
}
const esc = s => s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

window.pick = side => {
  const p = PAIRS[S.i];
  S.ans.push({idx:p.idx, kind:p.kind, cell:p.cell, choice:side,
    correct: p.kind==="check" ? (side===p.good_side) : null,
    hit: p.kind==="real" ? (side===p.hi_side) : null,
    gap: p.gap ?? null, ms: Date.now() - (S.tLast || Date.now())});
  S.i++; S.tLast = Date.now(); save(); render();
};
document.addEventListener("keydown", e => {
  if (S.prereg===null || S.i>=PAIRS.length) return;
  if (e.key==="f"||e.key==="F") pick("left");
  if (e.key==="j"||e.key==="J") pick("right");
});

function finish(){
  const real = S.ans.filter(a=>a.kind==="real");
  const chk  = S.ans.filter(a=>a.kind==="check");
  const hits = real.filter(a=>a.hit).length;
  const rate = real.length ? hits/real.length : 0;
  // 二项检验（双尾，正态近似）
  const se = Math.sqrt(0.25/Math.max(1,real.length));
  const zs = (rate-0.5)/se;
  const pv = 2*(1-0.5*(1+erf(Math.abs(zs)/Math.SQRT2)));
  const passed = chk.filter(a=>a.correct).length;
  $("pos").textContent = "已完成"; $("fill").style.width="100%";
  $("app").innerHTML = `<div class="done">
    <h2 style="margin-top:0">标注完成</h2>
    <p style="font-size:1.05rem">评分表偏好的一侧被选中：<strong>${hits} / ${real.length}
      （${(rate*100).toFixed(1)}%）</strong></p>
    <p style="color:var(--ink2)">二项检验 p = ${pv.toFixed(4)}
      ${pv<0.05 ? (rate>0.5?"显著高于 50%":"显著低于 50%") : "与 50% 无显著差异"}</p>
    <p style="color:var(--ink2)">注意力检验题：通过 ${passed} / ${chk.length}
      ${chk.length&&passed/chk.length<0.8 ? "　⚠ 通过率偏低，该批次质量存疑" : ""}</p>
    <hr style="border:none;border-top:1px solid var(--line);margin:18px 0">
    <p style="color:var(--muted);font-size:.9rem">你标注前的预期：</p>
    <p style="color:var(--ink2);white-space:pre-wrap">${esc(S.prereg||"（未填写）")}</p>
    <div style="margin-top:18px"><button onclick="dl()">导出 JSON</button></div>
    <p style="color:var(--muted);font-size:.84rem;margin-top:14px">
      导出后交给 analyze 脚本可做分品类拆分与逐格检验。清空重来：在控制台执行
      <kbd>localStorage.removeItem("${KEY}")</kbd></p>
  </div>`;
}
function erf(x){const t=1/(1+0.3275911*x);
  const y=1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-x*x);
  return y;}
window.dl = () => {
  const blob = new Blob([JSON.stringify({prereg:S.prereg, n:S.ans.length,
    answers:S.ans}, null, 2)], {type:"application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "annotation_results.json"; a.click();
};
render();
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=120)
    ap.add_argument("--checks", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260921)
    args = ap.parse_args()

    cells = collect()
    if not cells:
        raise SystemExit("[ERROR] 没有可用文本。先确认 validation_pilot.json 存在。")
    items = make_pairs(cells, args.pairs, args.checks, args.seed)
    real = [p for p in items if p["kind"] == "real"]

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(TEMPLATE.replace("__PAIRS__", json.dumps(items, ensure_ascii=False)),
                   encoding="utf-8")

    gaps = sorted(p["gap"] for p in real)
    print(f"\n{'=' * 62}\n  盲标工具已生成\n{'=' * 62}")
    print(f"  正式对子 {len(real)} 对，覆盖 {len({p['cell'] for p in real})} 个品类")
    print(f"  注意力检验题 {len(items) - len(real)} 道，已随机混入")
    print(f"  评分表分差：中位 {gaps[len(gaps)//2]:.2f}　区间 {gaps[0]:.2f}–{gaps[-1]:.2f}")
    print(f"  每段 {EXCERPT_CHARS} 字，与评审看到的窗口一致")
    print(f"  文件大小 {OUT.stat().st_size/1024/1024:.1f} MB")
    print(f"\n  打开：open {OUT}")
    print(f"  预计 {len(items)} 对 × 约 3 分钟 ≈ {len(items)*3/60:.1f} 小时，可分次完成")
    print(f"\n  ⚠ 本文件含已出版作品正文，仅供本地研究使用，勿上传或发布。")


if __name__ == "__main__":
    main()
