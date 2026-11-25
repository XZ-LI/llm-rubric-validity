#!/usr/bin/env python3
"""
build_annotation_round2.py — 生成第二轮盲标工具（双条件：阳性对照 + 效度检验）

第一轮（build_annotation.py / annotate.html）的缺口：
  它只有一个条件——评分表高分侧 vs 低分侧。若命中率落在 50%，无法区分
    (a) 评分表确实没有效度，
    (b) 标注者从 2000 字读不出任何差别，这台"仪器"本身分辨率为零。
  混在里面的 6 道注意力检验题（打乱句序）只能证明标注者没走神，
  不能证明他有分辨力——清醒但读不出差别的人照样 6 道全对、命中率 50%。

本轮用两个条件交错解决这件事。两个条件在**同一场次**内交错出现，
标注者无法分辨自己正在做哪一个，因此两个命中率可直接相减比较，
不受"隔天状态不同"的批次效应影响：

  条件 R（阳性对照 rating-contrast）
      读者评分差 ≥ --rating-gap，评分表分差 ≤ --rubric-match
      两本书评分表给分几乎一样，但读者行为差距明显。
      → 测的是：标注者能否从这个窗口读出**效标**信号。
      这是"仪器有没有分辨率"的已知真值对照。

  条件 S（效度检验 rubric-contrast）
      评分表分差 ≥ --rubric-gap，读者评分差 ≤ --rating-match
      两本书读者评分几乎一样，但评分表给分差距明显。
      → 测的是：评分表偏好的那一侧，是否对应人的阅读选择。
      比第一轮更干净：第一轮未控制读者评分，本轮把它配平掉了。

解读表（analyze_annotation.py 会直接算出来）：
      R 高于 50% 且 S 落在 50%  → 最强结果：同一个人、同一个窗口，
                                  读得出效标信号，读不出评分表信号
      R 和 S 都落在 50%        → 仪器分辨率为零，人工标注这一项整体不可用，
                                  不应写进结论
      R 和 S 都高于 50%        → 评分表确实携带了部分效标信号，
                                  主分析的零结果需要重新讨论

盲化措施与第一轮完全一致（不显示书名/作者/评分/分数、左右随机、品类打乱、
混入注意力检验题），另加两项：
  · 每对可勾选"这一对我基本是猜的"，记为 hard，事后按把握度分层
  · 本轮用到的正文默认**避开第一轮已出现过的 222 篇**，避免熟悉度带来的携带效应

节选窗口、章节切分、打乱句序三处逻辑直接复用 build_annotation.py，
保证两轮之间可比。

输出 annotation/annotate_round2.html，自包含，进度存 localStorage（独立 key，
不影响第一轮进度）。**不发布到网络**：正文为已出版作品，仅作本地研究使用。

用法：
  python3 build_annotation_round2.py                    # 默认 50 + 50 + 6
  python3 build_annotation_round2.py --rating-pairs 30 --rubric-pairs 30
  python3 build_annotation_round2.py --supply           # 只看供给量和功效，不出文件
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from build_annotation import EXCERPT_CHARS, first_chapter, scramble
from contamination import excluded_files

BASE = Path("review_loop")
ROUND1 = Path("annotation/annotate.html")
OUT = Path("annotation/annotate_round2.html")
KEYOUT = Path("annotation/round2_key.json")
REVIEWER = "qwen3-max"
MAX_PER_CELL = 4        # 单格取太多会让结果被一两个品类主导（--max-per-cell 可改）


def round1_texts() -> set[tuple[str, str]]:
    """第一轮已经让标注者读过的正文，(cell, file)。读不到就返回空集。"""
    if not ROUND1.exists():
        return set()
    t = ROUND1.read_text(encoding="utf-8")
    m = re.search(r"const PAIRS = (\[.*?\]);\nconst KEY", t, re.S)
    if not m:
        return set()
    return {(p["cell"], f) for p in json.loads(m.group(1)) if p["kind"] == "real"
            for f in (p["hi_file"], p["lo_file"])}


def collect(seen: set[tuple[str, str]]) -> dict[str, list[dict]]:
    """按格收集可用文本，附评分表分数与读者评分（两者都只用于配对，不展示）。"""
    cells: dict[str, list[dict]] = {}
    for p in sorted(BASE.glob("*/validation_pilot.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("reviewer_model") != REVIEWER:
            continue
        cell = p.parent.name
        drop = excluded_files(cell, d["category"], d["source_dir"])
        items = []
        for r in d["results"]:
            if r["file"] in drop or (cell, r["file"]) in seen:
                continue
            f = Path(d["source_dir"]) / d["category"] / r["file"]
            if not f.exists():
                continue
            ex = first_chapter(f)
            if ex:
                items.append({"file": r["file"], "score": r["rubric_score"],
                              "rating": r["real_rating"], "text": ex})
        if len(items) >= 4:
            cells[cell] = items
    return cells


def candidates(items: list[dict], cond: str, gap: float, match: float) -> list[tuple]:
    """该格内满足条件的所有对子，按对比度从大到小。(对比度, 高侧, 低侧)"""
    out = []
    for a, b in itertools.combinations(items, 2):
        dr = a["rating"] - b["rating"]
        ds = a["score"] - b["score"]
        if cond == "rating":
            if abs(dr) >= gap and abs(ds) <= match:
                hi, lo = (a, b) if dr > 0 else (b, a)
                out.append((abs(dr), hi, lo))
        else:
            if abs(ds) >= gap and abs(dr) <= match:
                hi, lo = (a, b) if ds > 0 else (b, a)
                out.append((abs(ds), hi, lo))
    out.sort(key=lambda x: -x[0])
    return out


def pick(cells: dict, cond: str, gap: float, match: float, n_want: int,
         taken: set[tuple[str, str]], rng: random.Random,
         cap: int = MAX_PER_CELL) -> list[dict]:
    """轮转各格挑互不重叠的对子：先每格 1 对，再第 2 对……避免被单格吃满。

    taken 跨条件共享，所以同一篇正文在整轮里最多出现一次。
    """
    pool = {c: candidates(it, cond, gap, match) for c, it in cells.items()}
    order = sorted(pool)
    rng.shuffle(order)
    got: list[dict] = []
    per: defaultdict[str, int] = defaultdict(int)
    for _ in range(cap):
        for cell in order:
            if len(got) >= n_want or per[cell] >= cap:
                continue
            for contrast, hi, lo in pool[cell]:
                if (cell, hi["file"]) in taken or (cell, lo["file"]) in taken:
                    continue
                taken.add((cell, hi["file"]))
                taken.add((cell, lo["file"]))
                per[cell] += 1
                flip = rng.random() < 0.5        # 左右随机，高侧不固定
                got.append({
                    "kind": cond, "cell": cell,
                    "left": (lo if flip else hi)["text"],
                    "right": (hi if flip else lo)["text"],
                    "hi_side": "right" if flip else "left",
                    "hi_file": hi["file"], "lo_file": lo["file"],
                    "contrast": round(contrast, 3),
                    "rating_gap": round(hi["rating"] - lo["rating"], 3),
                    "rubric_gap": round(hi["score"] - lo["score"], 3),
                })
                break
        if len(got) >= n_want:
            break
    return got[:n_want]


def build_checks(cells: dict, n: int, taken: set, rng: random.Random) -> list[dict]:
    """注意力检验题：同一段文本，一侧打乱句序。只验是否在读，不验分辨力。"""
    pool = [(c, it) for c, items in cells.items() for it in items
            if (c, it["file"]) not in taken]
    rng.shuffle(pool)
    out = []
    for cell, it in pool[:n]:
        flip = rng.random() < 0.5
        bad = scramble(it["text"])
        if bad == it["text"]:          # 句子太少，打乱无效，跳过
            continue
        out.append({"kind": "check", "cell": "—",
                    "left": (bad if flip else it["text"]),
                    "right": (it["text"] if flip else bad),
                    "good_side": "right" if flip else "left"})
    return out


def wilson(k: int, n: int) -> tuple[float, float]:
    if not n:
        return (0.0, 1.0)
    z, p = 1.959964, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def power(n: int, true_rate: float) -> float:
    """双尾 α=.05 下，真实命中率为 true_rate 时能测出显著偏离 50% 的概率。"""
    if n <= 0:
        return 0.0
    se0, se1 = math.sqrt(0.25 / n), math.sqrt(true_rate * (1 - true_rate) / n)
    crit = 1.959964 * se0
    z = (abs(true_rate - 0.5) - crit) / se1
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


TEMPLATE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>盲标第二轮：你更想接着读哪一个</title>
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
.hard{display:flex;align-items:center;gap:9px;justify-content:center;margin-top:16px;
padding:11px;border:1px dashed var(--line);border-radius:8px;cursor:pointer;
font-size:.92rem;color:var(--ink2);user-select:none}
.hard.on{border-style:solid;border-color:var(--warn);color:var(--warn);font-weight:600}
.hard input{width:17px;height:17px;accent-color:var(--warn);pointer-events:none}
.note{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--warn);
border-radius:0 8px 8px 0;padding:14px 16px;margin:18px 0;font-size:.9rem;color:var(--ink2)}
.done{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:26px}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:.94rem}
th,td{text-align:left;padding:9px 10px;border-bottom:1px solid var(--line);
font-variant-numeric:tabular-nums}
th{color:var(--muted);font-weight:600;font-size:.85rem}
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
const KEY = "annot_r2_v1";
let S = JSON.parse(localStorage.getItem(KEY) || '{"i":0,"ans":[],"prereg":null,"t0":null}');
const $ = id => document.getElementById(id);
const save = () => localStorage.setItem(KEY, JSON.stringify(S));
let hard = false;

function prereg(){
  $("app").innerHTML = `<div class="done">
    <h2 style="margin-top:0">开始前：先写下你的预期</h2>
    <p style="color:var(--ink2)">单人标注唯一能部分抵消实验者偏差的办法。写完封存，标完再对照。</p>
    <p style="color:var(--ink2);font-size:.92rem">本轮混了两类对子，你在标注时<strong>无法分辨</strong>当前这对属于哪一类：
    一类两本书读者评分差距明显而评分表给分接近，一类反过来。
    请分别预计你在这两类上会有多大比例选中"高的那一侧"？（50% = 完全无关）</p>
    <textarea id="pr" placeholder="例：读者评分那类我预计 60%，评分表那类 52%，因为……"></textarea>
    <p style="color:var(--ink2);font-size:.92rem;margin-top:16px">
      你的名字或代号（几个人标同一份时用来区分，会写进导出文件）：</p>
    <input id="who" placeholder="例：李雪昭 / A / 小王" style="width:260px;font:inherit;
      padding:8px 10px;border:1px solid var(--line);border-radius:7px;
      background:var(--bg);color:var(--ink)">
    <div style="margin-top:14px"><button onclick="startNow()">封存并开始</button></div>
    <p style="color:var(--muted);font-size:.85rem;margin-top:16px">
      共 ${PAIRS.length} 对，其中混有注意力检验题。
      <kbd>F</kbd> 选左、<kbd>J</kbd> 选右、<kbd>空格</kbd> 标记"基本是猜的"。</p>
  </div>`;
}
window.startNow = () => { S.prereg = $("pr").value.trim();
  S.who = $("who").value.trim() || "标注者1";
  S.t0 = Date.now(); S.tLast = Date.now(); save(); render(); };

function render(){
  if (S.prereg === null) return prereg();
  if (S.i >= PAIRS.length) return finish();
  const p = PAIRS[S.i];
  hard = false;
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
    <div class="hard" id="hardbox" onclick="toggleHard()">
      <input type="checkbox" id="hardck" tabindex="-1">
      <span>这一对我基本是猜的　<kbd>空格</kbd></span></div>
    <div class="note">只凭这两段判断。不必找"更好"的那个，找<strong>你更想继续读</strong>的那个。
    拿不准就勾上然后照直觉选——勾了不扣分，这一栏是用来做分层分析的，
    犹豫时不勾反而会把结果稀释掉。</div>`;
  window.scrollTo(0,0);
}
const esc = s => s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

window.toggleHard = () => { hard = !hard;
  $("hardck").checked = hard; $("hardbox").classList.toggle("on", hard); };

window.pick = side => {
  // 页面里没有答案，只记录"选了哪边"。判分由 analyze_annotation.py 对答案文件完成。
  S.ans.push({idx: PAIRS[S.i].idx, choice: side, hard: hard,
              ms: Date.now() - (S.tLast || Date.now())});
  S.i++; S.tLast = Date.now(); save(); render();
};
document.addEventListener("keydown", e => {
  if (S.prereg===null || S.i>=PAIRS.length) return;
  if (e.key===" ") { e.preventDefault(); toggleHard(); }
  if (e.key==="f"||e.key==="F") pick("left");
  if (e.key==="j"||e.key==="J") pick("right");
});

function finish(){
  const n = S.ans.length, h = S.ans.filter(a=>a.hard).length;
  $("pos").textContent = "已完成"; $("fill").style.width="100%";
  $("app").innerHTML = `<div class="done">
    <h2 style="margin-top:0">第二轮标注完成</h2>
    <p style="font-size:1.05rem">共 ${n} 对，其中 ${h} 对标为"基本是猜的"。</p>
    <p style="color:var(--ink2)">本页<strong>不显示成绩</strong>——答案不在这个文件里，
    页面也无从判分。这是刻意的：标注中途若能看到自己命中多少，
    后面的判断就会被它带着走，盲标也就白做了。</p>
    <div style="margin-top:18px"><button onclick="dl()">导出 JSON</button></div>
    <p style="color:var(--ink2);margin-top:16px">导出后跑：<br>
      <kbd>python3 analyze_annotation.py</kbd></p>
    <hr style="border:none;border-top:1px solid var(--line);margin:18px 0">
    <p style="color:var(--muted);font-size:.9rem">你标注前的预期：</p>
    <p style="color:var(--ink2);white-space:pre-wrap">${esc(S.prereg||"（未填写）")}</p>
    <p style="color:var(--muted);font-size:.84rem;margin-top:14px">
      清空重来：控制台执行 <kbd>localStorage.removeItem("${KEY}")</kbd></p>
  </div>`;
}
window.dl = () => {
  const blob = new Blob([JSON.stringify({round:2, annotator:S.who||"标注者1",
    prereg:S.prereg, i:S.i, n:S.ans.length, answers:S.ans}, null, 2)],
    {type:"application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "annotation_round2_results_" + (S.who||"标注者1") + ".json"; a.click();
};

// 随时导出当前进度：localStorage 只有一份，清站点数据、换浏览器、挪动文件都会丢。
(function(){
  const b = document.createElement("button");
  b.textContent = "导出当前进度";
  b.style.cssText = "position:fixed;right:14px;bottom:14px;z-index:99;padding:9px 13px;"
    + "font-size:13px;opacity:.7;box-shadow:0 1px 6px rgba(0,0,0,.18)";
  b.onmouseenter = () => b.style.opacity = 1;
  b.onmouseleave = () => b.style.opacity = .7;
  b.onclick = () => {
    if (!S.ans.length) { alert("还没有作答。"); return; }
    dl();
    setTimeout(() => alert("已导出 " + S.ans.length + " 对的作答。标完全部 "
      + PAIRS.length + " 对后记得再导一次。"), 300);
  };
  document.body.appendChild(b);
})();
render();
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rating-pairs", type=int, default=70, help="条件 R 对子数")
    ap.add_argument("--rubric-pairs", type=int, default=50, help="条件 S 对子数")
    ap.add_argument("--rating-gap", type=float, default=1.0, help="R：读者评分最小差")
    ap.add_argument("--rubric-match", type=float, default=0.5, help="R：评分表分差上限")
    ap.add_argument("--rubric-gap", type=float, default=1.5, help="S：评分表最小分差")
    ap.add_argument("--rating-match", type=float, default=0.3, help="S：读者评分差上限")
    ap.add_argument("--max-per-cell", type=int, default=MAX_PER_CELL,
                    help="单格最多取几对，调大能提高供给但会让某些品类占比变高")
    ap.add_argument("--checks", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--allow-reused", action="store_true",
                    help="允许复用第一轮出现过的正文（供给不足时）")
    ap.add_argument("--supply", action="store_true", help="只报告供给与功效，不生成文件")
    args = ap.parse_args()

    seen = set() if args.allow_reused else round1_texts()
    cells = collect(seen)
    if not cells:
        raise SystemExit("[ERROR] 没有可用文本。先确认 review_loop/*/validation_pilot.json 存在。")

    rng = random.Random(args.seed)
    random.seed(args.seed)                      # scramble() 用全局 random
    taken: set[tuple[str, str]] = set()
    pr = pick(cells, "rating", args.rating_gap, args.rubric_match,
              args.rating_pairs, taken, rng, args.max_per_cell)
    pv = pick(cells, "rubric", args.rubric_gap, args.rating_match,
              args.rubric_pairs, taken, rng, args.max_per_cell)

    print(f"\n{'=' * 64}\n  第二轮盲标：供给与功效\n{'=' * 64}")
    print(f"  第一轮已读过、本轮回避的正文：{len(seen)} 篇"
          f"{'（--allow-reused 已关闭回避）' if args.allow_reused else ''}")
    for name, got, want, g in (("R 阳性对照", pr, args.rating_pairs, args.rating_gap),
                               ("S 效度检验", pv, args.rubric_pairs, args.rubric_gap)):
        if not got:
            print(f"  {name}：0 对　⚠ 供给为零，放宽阈值再试")
            continue
        cs = sorted(p["contrast"] for p in got)
        share = Counter(p["cell"] for p in got).most_common(1)[0]
        print(f"  {name}：{len(got)} 对（要 {want}）　"
              f"覆盖 {len({p['cell'] for p in got})} 格　"
              f"最大单格占比 {share[1]}/{len(got)}　"
              f"对比度中位 {cs[len(cs) // 2]:.2f}　区间 {cs[0]:.2f}–{cs[-1]:.2f}")
        print(f"      若真实命中率为 60% / 65% / 70%，"
              f"测出显著的概率 = {power(len(got), .60):.0%} / "
              f"{power(len(got), .65):.0%} / {power(len(got), .70):.0%}")
        lo, hi = wilson(round(len(got) * .5), len(got))
        print(f"      若命中 50%，95% CI = [{lo:.1%}, {hi:.1%}]"
              f"　← 这是零结果能排除的范围，太宽就说明对子不够")

    if args.supply:
        return
    if not pr or not pv:
        raise SystemExit("[ERROR] 有条件供给为零，拒绝生成单条件的文件——"
                         "那等于退回第一轮的缺口。")

    items = pr + pv + build_checks(cells, args.checks, taken, rng)
    rng.shuffle(items)                          # 两条件与检验题交错
    for i, p in enumerate(items):
        p["idx"] = i

    OUT.parent.mkdir(exist_ok=True)
    # 题面与答案分离：HTML 里只有 idx 和两段正文，开控制台也看不到哪侧是高分侧。
    shown = [{"idx": p["idx"], "left": p["left"], "right": p["right"]} for p in items]
    key = [{k: v for k, v in p.items() if k not in ("left", "right")} for p in items]
    OUT.write_text(TEMPLATE.replace("__PAIRS__", json.dumps(shown, ensure_ascii=False)),
                   encoding="utf-8")
    KEYOUT.write_text(json.dumps({"seed": args.seed, "round": 2, "items": key},
                                 ensure_ascii=False, indent=1), encoding="utf-8")
    n_chk = len(items) - len(pr) - len(pv)
    print(f"\n  已生成 {OUT}　{OUT.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  答案另存 {KEYOUT}（HTML 内不含答案，无法在页面里判分或偷看）")
    print(f"  共 {len(items)} 对 = R {len(pr)} + S {len(pv)} + 检验题 {n_chk}，已打乱交错")
    print(f"  每段 {EXCERPT_CHARS} 字，与第一轮及评审看到的窗口一致")
    print(f"  localStorage key = annot_r2_v1（与第一轮 annot_v1 隔离，不影响已有进度）")
    print(f"\n  打开：open {OUT}")
    print(f"  预计 {len(items)} 对 × 约 3 分钟 ≈ {len(items) * 3 / 60:.1f} 小时，可分次完成")
    print(f"\n  ⚠ 本文件含已出版作品正文，仅供本地研究使用，勿上传或发布。")


if __name__ == "__main__":
    main()
