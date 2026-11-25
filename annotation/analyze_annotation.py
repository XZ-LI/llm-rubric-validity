#!/usr/bin/env python3
"""
analyze_annotation.py — 人工盲标结果的正式分析口径（第一轮 + 第二轮，可多标注者）

页面上的即时数字只是速览；凡要写进结论的数都从这里出。本脚本做六件事：

1. 判分
   第一轮的导出里自带 hit（页面当时就判了），第二轮的导出只有"选了哪边"，
   答案在 annotation/round2_key.json 里。两轮都重新对答案判一次，不信任导出里的
   既有判分——页面逻辑改过，重判才能保证两轮同口径。

2. 阳性对照（本脚本存在的主要理由）
   命中率落在 50% 有两种完全相反的解释：评分表没有效度，或者标注者从 2000 字
   根本读不出任何差别。要分开这两者，必须有一个**已知存在真实差异**的对子类型
   来证明这台"仪器"有分辨率。注意力检验题（打乱句序）不顶用，它只验是否在读。
     · 第二轮条件 R 就是为此设计的：读者评分差大、评分表分差配平。
     · 第一轮没有这个条件，但它可以**免费补算**：那 111 对里两本书的读者评分
       差中位 0.50，且与评分表方向无关（高分表侧同时评分更高的恰好 55/111）。
       按"你是否选中了读者评分更高的一侧"重判一遍，就得到一个弱版阳性对照。

3. 把稀释效应拆出来
   犹豫时的乱选把估计值往 50% 拉，而 50% 正好是零假设的方向——
   也就是说噪音会让人更容易"确认"评分表无效。按第二轮的 hard 标记分层，
   第一轮用作答耗时中位数分层（弱代理：耗时混了阅读速度和段落长度，
   且第一对因计时初始化恒为 0，已剔除）。

4. 剂量–反应
   按对比度分箱 + Cochran–Armitage 趋势检验。合并后的单一数字会掩盖这件事：
   若命中率随评分表分差单调上升，说明评分表携带了真信号，只是弱。

5. 多标注者
   同一文件可以给几个人标，把各自导出的 JSON 全部传进来。
   会给每人一行、合并一行，以及共同题上的两两一致率（Cohen's κ）。
   n=1 的结论只能说"预测不了某一位读者「，这是把它推向」预测不了读者"的唯一办法。

6. 功效与可排除区间
   每个零结果都附 95% CI 和"这个 n 能排除什么"。CI 跨越 50% 却写成
   "与 50% 无显著差异"是本项目 ERRATA 里已经犯过的那类错误。

用法：
  python3 analyze_annotation.py                       # 自动在常见位置找导出文件
  python3 analyze_annotation.py a.json b.json         # 显式给，可混两轮、可多人
  python3 analyze_annotation.py --json report.json    # 另存机读版
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path("review_loop")
ROUND1_HTML = Path("annotation/annotate.html")
ROUND2_KEY = Path("annotation/round2_key.json")
MIN_MS = 20_000          # 第二轮预注册：用时不足 20 秒的对子视为未读，剔除
SEARCH = [Path("."), Path("annotation"), Path.home() / "Downloads"]
PATTERNS = ["annotation_results*.json", "annotation_round2_results*.json"]

COND = {"real": "S　效度检验（第一轮，读者评分未配平）",
        "rubric": "S　效度检验（第二轮，读者评分已配平）",
        "rating": "R　阳性对照（读者评分差大，评分表配平）"}


# ───────────────────────── 统计 ─────────────────────────

def wilson(k: int, n: int) -> tuple[float, float]:
    if not n:
        return (0.0, 1.0)
    z, p = 1.959964, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def binom_p(k: int, n: int) -> float:
    """双尾精确二项检验（p=.5）。n 这个量级精确算得起，不用正态近似。"""
    if not n:
        return 1.0
    probs = [math.comb(n, i) for i in range(n + 1)]
    tot = float(sum(probs))
    obs = probs[k]
    return min(1.0, sum(p for p in probs if p <= obs * (1 + 1e-9)) / tot)


def two_prop(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """两个命中率之差的 z 检验（正态近似）。返回 (差值, p)。"""
    if not n1 or not n2:
        return (0.0, 1.0)
    p1, p2 = k1 / n1, k2 / n2
    pp = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    if se == 0:
        return (p1 - p2, 1.0)
    z = (p1 - p2) / se
    return (p1 - p2, 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))))


def cochran_armitage(bins: list[tuple[float, int, int]]) -> float:
    """有序分箱上的趋势检验。bins = [(箱得分, 命中数, 该箱 n), ...]，返回双尾 p。"""
    N = sum(n for _, _, n in bins)
    K = sum(k for _, k, _ in bins)
    if N == 0 or K in (0, N) or len(bins) < 2:
        return 1.0
    pbar = K / N
    num = sum(w * (k - n * pbar) for w, k, n in bins)
    sw = sum(w * n for w, _, n in bins)
    sw2 = sum(w * w * n for w, _, n in bins)
    var = pbar * (1 - pbar) * (sw2 - sw * sw / N)
    if var <= 0:
        return 1.0
    z = num / math.sqrt(var)
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def power(n: int, rate: float) -> float:
    if n <= 0:
        return 0.0
    crit = 1.959964 * math.sqrt(0.25 / n)
    se = math.sqrt(rate * (1 - rate) / n)
    return 0.5 * (1 + math.erf(((abs(rate - 0.5) - crit) / se) / math.sqrt(2)))


def kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    if not n:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


# ───────────────────────── 载入 ─────────────────────────

def ratings() -> dict[tuple[str, str], float]:
    out = {}
    for p in sorted(BASE.glob("*/validation_pilot.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("reviewer_model") != "qwen3-max":
            continue
        for r in d["results"]:
            out[(p.parent.name, r["file"])] = r["real_rating"]
    return out


def round1_key() -> dict[int, dict]:
    """第一轮没有单独的答案文件，答案嵌在 annotate.html 里，从那里反解。"""
    if not ROUND1_HTML.exists():
        return {}
    m = re.search(r"const PAIRS = (\[.*?\]);\nconst KEY",
                  ROUND1_HTML.read_text(encoding="utf-8"), re.S)
    return {p["idx"]: p for p in json.loads(m.group(1))} if m else {}


def round2_key() -> dict[int, dict]:
    if not ROUND2_KEY.exists():
        return {}
    return {p["idx"]: p for p in
            json.loads(ROUND2_KEY.read_text(encoding="utf-8"))["items"]}


def find_exports() -> list[Path]:
    out, seen = [], set()
    for d in SEARCH:
        for pat in PATTERNS:
            for f in sorted(d.glob(pat)):
                r = f.resolve()
                if r not in seen:
                    seen.add(r)
                    out.append(f)
    return out


def annotator_of(path: Path) -> str:
    """从文件名推断标注者：annotation_round2_results_b.json → b。

    同一个人的第一轮与第二轮导出必须归到同一个标注者，否则会被当成两个人，
    一致率和"多标注者"那一节就全是假的。
    """
    s = path.stem
    # 浏览器重复下载会加 " (1)"、" (2)"；备份习惯常加日期。这些都不是人名，
    # 否则同一个人的多次备份会被当成多位标注者重复计数。
    s = re.sub(r"\s*\(\d+\)$", "", s)
    s = re.sub(r"[-_ ]*\d{4}[-_]?\d{2}[-_]?\d{2}$", "", s)
    s = re.sub(r"[-_ ]*(backup|partial|备份|部分)$", "", s, flags=re.I)
    for pre in ("annotation_round2_results", "annotation_results"):
        if s.startswith(pre):
            rest = s[len(pre):].lstrip("_-").strip()
            return rest or "主标注者"
    return s


def load(path: Path, k1: dict, k2: dict, rate: dict) -> dict | None:
    """读一份导出，重新判分，返回标准化记录。"""
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  ⚠ 跳过 {path}：{e}")
        return None
    ans = d.get("answers") or []
    if not ans:
        return None
    rnd = 2 if (d.get("round") == 2 or "kind" not in ans[0]) else 1
    who = d.get("annotator") or annotator_of(path)
    key = k2 if rnd == 2 else k1
    if not key:
        print(f"  ⚠ 跳过 {path}：找不到第 {rnd} 轮的答案（"
              f"{ROUND2_KEY if rnd == 2 else ROUND1_HTML} 缺失）")
        return None

    rows = []
    n_rec = n_fast = 0
    for a in ans:
        k = key.get(a["idx"])
        if k is None:
            continue
        if a.get("recognized") or a.get("choice") is None:
            n_rec += 1                     # 认出了其中一本：按预注册剔除，不算选错
            continue
        if rnd == 2 and a.get("ms") is not None and a["ms"] < MIN_MS:
            n_fast += 1                    # 用时过短视为未读：按预注册剔除
            continue
        kind = k.get("kind", "real")
        r = {"round": rnd, "idx": a["idx"], "kind": kind, "cell": k.get("cell", "—"),
             "choice": a["choice"], "hard": a.get("hard"), "ms": a.get("ms"),
             "hit": None, "correct": None, "contrast": k.get("contrast"),
             "rating_hit": None}
        if kind == "check":
            r["correct"] = a["choice"] == k["good_side"]
        else:
            r["hit"] = a["choice"] == k["hi_side"]
            r["contrast"] = k.get("contrast") or abs(k.get("gap") or 0)
            # 读者评分侧：第一轮的免费阳性对照
            ra = rate.get((k["cell"], k["hi_file"]))
            rb = rate.get((k["cell"], k["lo_file"]))
            if ra is not None and rb is not None and ra != rb:
                hi_rating_side = k["hi_side"] if ra > rb else (
                    "left" if k["hi_side"] == "right" else "right")
                r["rating_hit"] = a["choice"] == hi_rating_side
                r["rating_gap"] = abs(ra - rb)
        rows.append(r)
    for r in rows:
        r["who"] = who
    if n_rec:
        print(f"  {path.name}：{n_rec} 对因认出而剔除")
    if n_fast:
        print(f"  {path.name}：{n_fast} 对用时不足 {MIN_MS // 1000} 秒而剔除")
    return {"path": path, "who": who, "round": rnd,
            "prereg": d.get("prereg"), "rows": rows}


# ───────────────────────── 报告 ─────────────────────────

W = 74


def dw(s: str) -> int:
    """终端显示宽度：中日韩全角字符占 2 列。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def pad(s: str, n: int) -> str:
    return s + " " * max(0, n - dw(s))


def head(t: str) -> None:
    print(f"\n{'─' * W}\n{t}\n{'─' * W}")


def line(label: str, rows: list[dict], field: str = "hit", indent: int = 0) -> dict:
    vals = [r[field] for r in rows if r.get(field) is not None]
    ind = " " * indent
    if not vals:
        print(f"  {pad(ind + label, 44)}  —")
        return {}
    n, k = len(vals), sum(vals)
    lo, hi = wilson(k, n)
    p = binom_p(k, n)
    star = " *" if p < 0.05 else "  "
    print(f"  {pad(ind + label, 44)}  {k:>3}/{n:<3}  {k / n:6.1%}  "
          f"[{lo:5.1%}, {hi:5.1%}]  p={p:.4f}{star}")
    return {"label": label, "k": k, "n": n, "rate": k / n,
            "ci": [lo, hi], "p": p}


def table_header() -> None:
    print(f"  {'':<44}  {'命中':>5}    {'比率':>4}   {'95% CI':^14}  p vs 50%")


def bins3(rows: list[dict], field: str = "contrast") -> list[tuple[float, int, int]]:
    vals = sorted(r[field] for r in rows if r.get(field) is not None)
    if len(vals) < 9:
        return []
    q1, q2 = vals[len(vals) // 3], vals[2 * len(vals) // 3]
    out = []
    for w, sel in ((1.0, lambda v: v <= q1),
                   (2.0, lambda v: q1 < v <= q2),
                   (3.0, lambda v: v > q2)):
        g = [r for r in rows if r.get(field) is not None and sel(r[field])
             and r.get("hit") is not None]
        if g:
            out.append((w, sum(r["hit"] for r in g), len(g), g))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", type=Path, help="导出的 JSON，可多份、可混两轮")
    ap.add_argument("--json", type=Path, help="另存机读版报告")
    args = ap.parse_args()

    k1, k2, rate = round1_key(), round2_key(), ratings()
    paths = args.files or find_exports()
    if not paths:
        raise SystemExit(
            "没找到导出文件。在标注页点「导出 JSON」，然后把文件放到以下任一位置：\n  "
            + "\n  ".join(str(d) for d in SEARCH)
            + "\n或直接把路径作为参数传进来。")

    subs = [s for s in (load(p, k1, k2, rate) for p in paths) if s]
    if not subs:
        raise SystemExit("导出文件都读不出有效作答。")

    # 同一人同一轮给了多份（中途备份 + 最终导出），只保留作答最多的那份。
    best: dict[tuple[str, int], dict] = {}
    dropped = []
    for s in subs:
        k = (s["who"], s["round"])
        if k not in best or len(s["rows"]) > len(best[k]["rows"]):
            if k in best:
                dropped.append(best[k])
            best[k] = s
        else:
            dropped.append(s)
    if dropped:
        print("\n  同一标注者同一轮的多份导出，只取最完整的一份，已忽略：")
        for s in dropped:
            print(f"    {s['path'].name}（{len(s['rows'])} 条）")
    subs = sorted(best.values(), key=lambda s: (s["who"], s["round"]))

    print(f"\n{'=' * W}\n  人工盲标分析　（{len(subs)} 份导出）\n{'=' * W}")
    for s in subs:
        c = Counter(r["kind"] for r in s["rows"])
        print(f"  第 {s['round']} 轮　标注者 {s['who']}　{s['path'].name}　"
              f"{len(s['rows'])} 条　"
              f"{'　'.join(f'{k}:{v}' for k, v in sorted(c.items()))}")

    rows = [r for s in subs for r in s["rows"]]
    who = sorted({s["who"] for s in subs})
    multi = len(who) > 1
    report: dict = {"n_submissions": len(subs), "sections": {}}

    # ── 注意力检验 ──
    chk = [r for r in rows if r["kind"] == "check"]
    if chk:
        ok = sum(r["correct"] for r in chk)
        head("注意力检验（打乱句序。只验是否在读，不验分辨力）")
        print(f"  通过 {ok} / {len(chk)}"
              + ("　⚠ 通过率 < 80%，该批次质量存疑，下面的数都要打折看"
                 if ok / len(chk) < 0.8 else "　正常"))
        report["sections"]["checks"] = {"passed": ok, "n": len(chk)}

    # ── 主结果 ──
    R = [r for r in rows if r["kind"] == "rating"]
    S1 = [r for r in rows if r["kind"] == "real"]
    S2 = [r for r in rows if r["kind"] == "rubric"]
    head("主结果：命中「高的那一侧」的比例")
    if multi:
        print(f"  ⚠ 合并了 {len(who)} 位标注者（{'、'.join(who)}）。同一批题被多人标过，")
        print("    试次之间不独立，下面的 p 值与 CI 偏乐观。"
              "正式引用请看「多标注者」一节的分人结果。")
    table_header()
    res = {}
    if R:
        res["R"] = line(COND["rating"], R)
    if S1:
        res["S1"] = line(COND["real"], S1)
    if S2:
        res["S2"] = line(COND["rubric"], S2)
    if S1 and S2:
        res["S"] = line("S　两轮合并", S1 + S2)
    report["sections"]["main"] = res

    # ── 第一轮的免费阳性对照 ──
    rc = [r for r in S1 if r.get("rating_hit") is not None]
    if rc:
        head("第一轮补算的弱版阳性对照"
             "（同一批作答，改问：你选中的是不是读者评分更高的一本）")
        table_header()
        res["R1_weak"] = line("R′　第一轮（评分差中位仅 0.5，信号弱）", rc, "rating_hit")
        print("\n  注意：这些对子当初不是按读者评分配的，评分差小，"
              "\n  所以它比第二轮的条件 R 弱得多，只能当旁证，不能替代。")

    # ── 把握度分层 ──
    hard_rows = [r for r in rows if r.get("hard") is not None and r["kind"] != "check"]
    if hard_rows:
        head("按把握度分层（第二轮的 hard 标记）")
        print("  犹豫时的乱选把估计值往 50% 拉，而 50% 正好是零假设的方向。")
        table_header()
        for name, kind in (("R 阳性对照", "rating"), ("S 效度检验", "rubric")):
            g = [r for r in hard_rows if r["kind"] == kind]
            if not g:
                continue
            line(f"{name}　有把握", [r for r in g if not r["hard"]], indent=0)
            line(f"{name}　基本是猜的", [r for r in g if r["hard"]], indent=0)
        n_hard = sum(r["hard"] for r in hard_rows)
        print(f"\n  标为「基本是猜的」：{n_hard} / {len(hard_rows)}"
              f"（{n_hard / len(hard_rows):.0%}）")

    ms_rows = [r for r in S1 if r.get("ms")]
    if len(ms_rows) >= 20:
        med = sorted(r["ms"] for r in ms_rows)[len(ms_rows) // 2]
        head(f"第一轮按作答耗时分层（中位 {med / 1000:.0f} 秒）")
        print("  弱代理：耗时混了阅读速度与段落长度，首对因计时初始化恒为 0 已剔除。")
        table_header()
        line("快（≤ 中位，多半是一眼就定）", [r for r in ms_rows if r["ms"] <= med])
        line("慢（> 中位，多半是纠结过）", [r for r in ms_rows if r["ms"] > med])

    # ── 剂量–反应 ──
    for name, g in (("S　评分表分差", S1 + S2), ("R　读者评分差", R)):
        b = bins3(g)
        if not b:
            continue
        head(f"剂量–反应：{name}越大，命中率是否越高")
        table_header()
        for w, k, n, grp in b:
            cs = sorted(r["contrast"] for r in grp)
            line(f"第 {int(w)} 箱　对比度 {cs[0]:.2f}–{cs[-1]:.2f}", grp)
        p = cochran_armitage([(w, k, n) for w, k, n, _ in b])
        print(f"\n  Cochran–Armitage 趋势检验　p = {p:.4f}"
              + ("　← 存在单调趋势，合并后的单一数字掩盖了它"
                 if p < 0.05 else "　无单调趋势"))

    # ── 两条件之差 ──
    if R and (S1 or S2):
        Sall = S2 or S1
        kR, nR = sum(r["hit"] for r in R), len(R)
        kS, nS = sum(r["hit"] for r in Sall), len(Sall)
        d, p = two_prop(kR, nR, kS, nS)
        head("两条件之差（R − S）")
        print(f"  {d * 100:+.1f} 个百分点　双比例 z 检验 p = {p:.4f}"
              + ("  *" if p < 0.05 else ""))
        if S2:
            print("  （两条件在同一场次交错出现，标注者无法分辨，故可直接相减）")
        else:
            print("  ⚠ S 取自第一轮，与 R 不同场次，差值含批次效应，仅供参考。")
        report["sections"]["difference"] = {"diff": d, "p": p}

    # ── 多标注者 ──
    if multi:
        head("多标注者")
        table_header()
        per = []
        for w in who:
            for kind, nm in (("rating", "R 阳性对照"), ("rubric", "S 效度检验"),
                             ("real", "S 第一轮")):
                g = [r for r in rows if r["who"] == w and r["kind"] == kind]
                if g:
                    per.append(line(f"{w}　{nm}", g))
        ids = defaultdict(dict)
        for r in rows:
            if r["kind"] in ("real", "rubric", "rating"):
                ids[r["who"]][(r["round"], r["idx"])] = \
                    1 if r["choice"] == "left" else 0
        names = list(ids)
        print()
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                common = set(ids[names[i]]) & set(ids[names[j]])
                if len(common) >= 10:
                    a = [ids[names[i]][c] for c in common]
                    b = [ids[names[j]][c] for c in common]
                    print(f"  {names[i]} × {names[j]}：共同题 {len(common)}　"
                          f"一致率 {sum(x == y for x, y in zip(a, b)) / len(common):.1%}　"
                          f"κ = {kappa(a, b):.3f}")
        report["sections"]["per_annotator"] = per
    else:
        head("标注者人数")
        print("  只有 1 位标注者。本分析支持的结论上限是"
              "「评分表预测不了某一位熟悉该品类的读者的阅读选择」，\n"
              "  不是「预测不了读者」。要推到后者，需要再找 2–3 人标同一个文件，"
              "把导出一起传进来。")

    # ── 判读 ──
    head("判读")
    r_res = res.get("R")
    s_res = res.get("S") or res.get("S2") or res.get("S1")
    s_src = ("两轮合并" if res.get("S") else
             "第二轮（读者评分已配平）" if res.get("S2") else "第一轮")
    if s_res:
        print(f"  S 条件口径：{s_src}，n = {s_res['n']}")
    if not r_res:
        print("  还没有条件 R 的数据（第二轮未做或未导出）。")
        print("  在此之前，S 条件落在 50% 这件事**无法解读**：")
        print("  它既可能是评分表没有效度，也可能是这个窗口下人就是读不出差别。")
    else:
        r_sig = r_res["ci"][0] > 0.5
        print(f"  仪器分辨率：条件 R 的 95% CI = "
              f"[{r_res['ci'][0]:.1%}, {r_res['ci'][1]:.1%}]")
        if not r_sig:
            print("  → CI 覆盖 50%，**没有证据表明标注者能从 2000 字读出效标信号**。")
            print("     此时 S 条件的零结果不可解读，人工标注这一项应整体报为「不确定」，")
            print("     不得用来支持「评分表无效度」。主效度主张只能靠 768 本 × 行为指标那条线。")
            print(f"     附：当前 n={r_res['n']}，若真实命中率为 60%，"
                  f"测出显著的概率只有 {power(r_res['n'], .60):.0%}；"
                  f"65% 时 {power(r_res['n'], .65):.0%}。")
            print("     也就是说，这个零结果更可能是功效不足，而不是分辨率真为零。")
        else:
            print("  → CI 不含 50%，标注者确实能读出效标信号，这台仪器有分辨率。")
            if not s_res:
                print("  → 但还没有 S 条件的数据，无法比对。")
            elif s_res["p"] >= 0.05:
                print(f"  → 而 S 条件为 {s_res['rate']:.1%}，"
                      f"CI = [{s_res['ci'][0]:.1%}, {s_res['ci'][1]:.1%}]，"
                      f"p = {s_res['p']:.4f}（{s_res['n']} 对）")
                print("     同一个人、同一个 2000 字窗口，读得出效标信号、读不出评分表信号。")
                print("     这是本项目能拿到的最强版本的人工证据。")
            elif s_res["rate"] > 0.5:
                print(f"  → 但 S 条件也显著（{s_res['rate']:.1%}，p = {s_res['p']:.4f}）：")
                print("     评分表确实携带了部分效标信号，主分析的零结果需要重新讨论，")
                print("     不能写成「评分表与读者行为无关」。")
            else:
                print(f"  → S 条件显著**低于** 50%（{s_res['rate']:.1%}，"
                      f"p = {s_res['p']:.4f}）：")
                print("     评分表偏好的一侧，人系统性地更不想读。这是反向预测，")
                print("     比零相关更强的结论，但先查配对与判分方向有没有写反。")
    print("\n  提醒：CI 跨越 50% 时不要写成「与 50% 无显著差异」就收尾——")
    print("  要同时报出这个 n 能排除什么。ERRATA 里已经有过这类教训。")

    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                             encoding="utf-8")
        print(f"\n  机读版已写入 {args.json}")
    print()


if __name__ == "__main__":
    main()
