#!/usr/bin/env python3
"""
validate_pilot.py — 评分表外部效度检验（试水版）

和 validate_rubric.py 同一个研究问题：机器归纳的评分表，能不能量出真实质量。
差别有三处，都是为了先花小钱看方向：

  1. 精简输出   评审只回 分数 + 失败/成功举证，不再生成 suggestion /
                strengths / top_issues —— 那三项是喂给 review_loop 改写系统
                提示用的，效度研究用不到。实测省 43% 输出 token。
                failures/success 保留：论文里要用它们展示"AI 依据什么打分"。
  2. 缺表自动补 没有 rubrics.json 的品类先归纳一张（1 次调用）。
  3. 记账      逐次记录真实 usage，含推理 token，跑完报实际花费。
                原脚本看不到这个数，成本只能靠猜。

不修改 review_loop.py / validate_rubric.py，只复用它们的函数。

用法：
  export OPENAI_API_KEY=sk-...
  python3 validate_pilot.py --dry-run
  python3 validate_pilot.py                      # 默认跑试水两个品类
  python3 validate_pilot.py --cell 民国言情:Nvpin --cell 快穿:Nvpin
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from review_loop import (  # noqa: E402
    BRAINSTORM_CATEGORIES,
    CALIBRATION_LOW_N,
    FEMALE_CATEGORIES,
    REVIEWER_MODEL,
    TOP_SOURCE_NOVELS,
    clean_json,
    derive_rubrics,
    format_calibration_block,
    load_calibration_anchors,
)
from validate_rubric import (  # noqa: E402
    CHAPTERS_TO_SCORE,
    load_category,
    pearson,
    spearman,
    split_chapters,
)

OUTPUT_BASE = Path("review_loop")

# 试水：离散度最高的两个品类，信号最强，一男一女
DEFAULT_CELLS = [("民国言情", "Nvpin"), ("动漫衍生", "Nanpin")]

PROVIDERS = {
    "openai":   ("https://api.openai.com/v1",                         "OPENAI_API_KEY",    "o3"),
    "deepseek": ("https://api.deepseek.com",                          "DEEPSEEK_API_KEY",  "deepseek-v4-pro"),
    "qwen":     ("https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY", "qwen3-max"),
}

# ── 单价表 ────────────────────────────────────────────────────────────────────
# 每百万 token。**只填你在各厂商后台亲自核对过的数字**，并写上核对日期。
# 之前这里写死成 o3 的 $2/$8 并套用到所有厂商，导致 DeepSeek / 通义的花费
# 报成了"按 o3 价格算会是多少"，是错的。现在改成：单价未核对就不报金额，
# 只报 token 用量——宁可不给数，也不给错的数。
#
# 填写格式： "模型名": {"in": 输入价, "cached_in": 缓存输入价, "out": 输出价,
#                       "cur": 币种, "verified": "YYYY-MM-DD 核对来源"}
# 阿里云通义为阶梯定价，按输入长度分档。本研究单次输入约 8K token，
# 稳定落在最便宜的第一档（0–32K）。若日后加长上下文（比如每本评更多章），
# 越过 32K 分界后单价会跳档，须重算。
PRICING: dict[str, dict] = {
    "qwen3-max": {"in": 2.5, "cached_in": 2.5, "out": 10.0, "cur": "CNY",
                  "verified": "2026-09-10 help.aliyun.com/zh/model-studio/model-pricing"
                              "（华北2；0–32K 档；缓存折扣未公示，故按原价计，偏保守）"},
    "qwen-plus": {"in": 0.8, "cached_in": 0.8, "out": 2.0, "cur": "CNY",
                  "verified": "2026-09-10 同上（0–128K 档）"},
    "qwen-turbo": {"in": 0.3, "cached_in": 0.3, "out": 0.6, "cur": "CNY",
                   "verified": "2026-09-10 同上（不分档；思考模式输出 ¥3/M，另配条目）"},
    # o3 有多个计费档。本研究用 Flex：非实时异步任务专用，价格是标准档的一半，
    # 而 o3 单次本就要跑近 10 分钟，延迟对我们无影响。
    # 代价是 Flex 更容易返回资源不可用，靠 call_json 的重试兜住。
    "o3": {"in": 1.00, "cached_in": 0.25, "out": 4.00, "cur": "USD",
           "verified": "2026-09-10 developers.openai.com/api/docs/pricing（Flex 档）"},
    # 标准档留档备查：in 2.00 / cached_in 0.50 / out 8.00
    #
    # ⚠ 待核实的对不上：按标准档算，试水那 40 次调用应为 $1.44
    #   （fresh 236,175×2 + cached 150,656×0.5 + out 111,551×8），
    #   但 platform 后台显示整个九月累计仅 $1.10，且已包含该次跑批。
    #   40 次单独不可能超过包含它的总额，故二者必有一错。
    #   两种可能：(a) 后台用量统计延迟未结清；(b) 本账户实际单价低于标价。
    #   后台是权威。核实办法：下次跑 o3 前记下后台数字，跑完再看差值，
    #   反推真实单价后回填此处（即测 DeepSeek 用的余额差法）。
    #   在核实前，用本表算出的 o3 金额应视为上限估计。
    "o4-mini": {"in": 0.55, "cached_in": 0.138, "out": 2.20, "cur": "USD",
                "verified": "2026-09-10 同上（Flex 档）"},
}


# ── 实测单次成本 ──────────────────────────────────────────────────────────────
# 有些厂商（DeepSeek）提供准实时余额接口，可以直接测："查余额 → 跑 N 次 → 再查余额"。
# 这比查价目表可靠：它自动包含了推理 token、缓存折扣和当期优惠。
# 阿里云账单是延迟结算的，这招对通义无效，只能去后台看单价。
#
# 注意 cache_hit 字段：实测时若输入大量命中缓存，得到的单价会偏低，
# 正式跑（每本书都不同）应上浮。
EMPIRICAL_PER_CALL: dict[str, dict] = {
    "deepseek-v4-pro": {
        "amount": 0.0900, "cur": "CNY",
        "measured": "2026-09-10 余额差法，2 次调用 ¥0.18",
        "cache_hit": 0.73,
        "note": "输入 73% 命中缓存；无缓存时应上浮，全量 840 本建议按 ¥85–90 预留",
    },
}


def price_for(model: str) -> dict | None:
    for key, val in PRICING.items():
        if key in model:
            return val
    return None


def empirical_for(model: str) -> dict | None:
    for key, val in EMPIRICAL_PER_CALL.items():
        if key in model:
            return val
    return None


class Ledger:
    """逐次记账，含推理 token —— 这是原脚本量不到的部分。"""

    def __init__(self, model: str = "") -> None:
        self.calls = 0
        self.tin = self.tcached = self.tout = self.treason = 0
        self.model = model                 # 用来查对应厂商的单价
        self._lock = threading.Lock()      # 并发下记账必须加锁，否则计数会丢

    def add(self, usage) -> None:
        with self._lock:
            self.calls += 1
            self.tin += getattr(usage, "prompt_tokens", 0) or 0
            self.tout += getattr(usage, "completion_tokens", 0) or 0
            det = getattr(usage, "completion_tokens_details", None)
            if det is not None:
                self.treason += getattr(det, "reasoning_tokens", 0) or 0
            pdet = getattr(usage, "prompt_tokens_details", None)
            if pdet is not None:
                self.tcached += getattr(pdet, "cached_tokens", 0) or 0

    @property
    def cost(self) -> tuple[float, str] | None:
        """返回 (金额, 币种)；该模型单价未核对时返回 None，不猜。"""
        p = price_for(self.model)
        if not p:
            return None
        fresh = max(0, self.tin - self.tcached)
        amount = (fresh / 1e6 * p["in"]
                  + self.tcached / 1e6 * p["cached_in"]
                  + self.tout / 1e6 * p["out"])
        return amount, p["cur"]

    def per_call(self) -> str:
        """按本次实测的每次调用用量，外推全量 840 本的 token 需求。"""
        if not self.calls:
            return ""
        return (f"  单次均值：输入 {self.tin // self.calls:,}"
                f"（缓存 {self.tcached // self.calls:,}）"
                f"，输出 {self.tout // self.calls:,}\n"
                f"  840 本外推：输入 {self.tin // self.calls * 840 / 1e6:.2f}M"
                f"，输出 {self.tout // self.calls * 840 / 1e6:.2f}M token")

    def report(self) -> str:
        if not self.calls:
            return "  （无调用）"
        vis = self.tout - self.treason
        ratio = f"{self.treason / vis:.2f}" if vis > 0 else "—"
        c = self.cost
        emp = empirical_for(self.model)
        if c:
            money = f"  本次花费 ≈ {c[1]} {c[0]:,.2f}（单价已核对：{price_for(self.model)['verified']}）"
        elif emp:
            est = emp["amount"] * self.calls
            money = (f"  本次花费 ≈ {emp['cur']} {est:,.2f}"
                     f"（按实测单次 {emp['cur']} {emp['amount']:.4f} × {self.calls} 次外推）\n"
                     f"    实测方式：{emp['measured']}\n"
                     f"    ⚠ {emp['note']}")
        else:
            money = (f"  金额未计算：{self.model or '该模型'} 既无核对单价，也无实测单次成本。\n"
                     f"  请拿下面的 token 数乘以后台单价——不要用别家的价格换算。")
        return (
            f"  模型 {self.model or '（未记录）'}\n"
            f"  调用 {self.calls} 次\n"
            f"  输入 {self.tin:,} token（其中命中缓存 {self.tcached:,}）\n"
            f"  输出 {self.tout:,} token = 可见 {vis:,} + 推理 {self.treason:,}\n"
            f"  推理 / 可见 = {ratio}\n"
            f"{self.per_call()}\n"
            f"{money}"
        )


def call_json(client, system: str, user: str, ledger: Ledger, label: str = "",
              model: str | None = None, extra_body: dict | None = None):
    """带记账与限速重试的调用。推理模型不接受 temperature。"""
    model = model or REVIEWER_MODEL
    kwargs = dict(model=model,
                  messages=[{"role": "system", "content": system},
                            {"role": "user", "content": user}])
    if not any(t in model for t in ("o1", "o3", "reasoner", "thinking")):
        kwargs["temperature"] = 0.2
    # PRICING 里 o3 / o4-mini 记的是 Flex 档价格，这里必须真的请求 Flex，
    # 否则会按标准档计费（贵一倍），预算表就成了低报。
    if any(t in model for t in ("o3", "o4-mini")):
        kwargs["service_tier"] = "flex"
    # 厂商私有参数（如通义的 enable_thinking）。RQ5 用它在同一模型内
    # 开关推理，做不混淆模型身份的对照。
    if extra_body:
        kwargs["extra_body"] = extra_body
    for attempt in range(6):
        try:
            resp = client.chat.completions.create(**kwargs)
            if resp.usage is not None:
                ledger.add(resp.usage)
            return resp.choices[0].message.content
        except Exception as e:
            msg = str(e).lower()
            if "rate_limit" in msg or "429" in msg:
                wait = 30 * (attempt + 1)
                print(f"\n    [限速] 等 {wait}s 重试（{attempt + 1}/6）...", flush=True)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"{label} 超过最大重试次数")


def genre_note_for(category: str) -> str:
    """沿用 review_loop 的品类特别说明，保证与主实验同一把尺子。"""
    note = ""
    if category in FEMALE_CATEGORIES:
        note += (
            "\n\n## 女频品类特别说明（评分前必读）\n"
            "【现代视角吐槽是优点】穿书/穿越女主用现代词汇、社畜梗、互联网用语吐槽古风世界，"
            "是玄幻言情/快穿等品类的核心卖点和标志性风格。"
            "请勿将'现代吐槽与古风基调混杂'列为缺陷——这正是读者付费阅读的核心体验。\n"
            "【扣分项】：女主恋爱脑、被动受害、文言文腔、信息堆砌无节奏。\n"
            "【加分项】：现代视角反差笑点、独立破局、爽感节奏、角色对话各有个性。"
        )
    if category in BRAINSTORM_CATEGORIES:
        note += (
            "\n\n## 脑洞品类特别说明（评分前必读）\n"
            "【脑洞鲜是硬门槛】如果核心设定只是穿越/重生/系统/读心/玄学算命等常见标签的换皮，"
            "即使文笔顺畅，脑洞相关维度不得高于7分，加权总分原则上不得高于8分。"
        )
    return note


def evaluate_lean(client, chapters: list[str], rubrics: list[dict],
                  category: str, calibration_block: str, ledger: Ledger,
                  model: str | None = None, extra_body: dict | None = None) -> dict:
    """与 review_loop.evaluate_chapters 同一套评分口径，只裁掉改写建议部分。"""
    rubric_str = "\n".join(
        f"  {i + 1}. 【{r['name']}】（权重{r.get('weight', 1)}）：{r.get('description', '')}"
        for i, r in enumerate(rubrics)
    )
    chapters_str = "".join(
        f"\n\n=== 第{i}章 ===\n{ch[:2000]}\n" for i, ch in enumerate(chapters, 1)
    )

    prompt = textwrap.dedent(f"""\
        你是严格的{category}网络小说评审专家，代表真实读者评分。

        {calibration_block}

        ## 评分维度
        {rubric_str}

        ## 待评内容
        {chapters_str}

        ## 评分流程（必须按顺序执行）

        第一步：对每个维度，先引用原文中2个具体的失败句子或段落，再引用1个做得好的地方。
        第二步：根据与上方高分范例的对比，给出0-10分（对齐真实读者评分标准，9分以上极难获得）。

        最后给出加权总分（严格按高分范例标准，不得因"整体不错"而虚高）。

        以JSON格式输出：
        {{
          "scores": {{"维度名": {{"score": 分数, "failures": ["失败例1", "失败例2"], "success": "好的地方"}}}},
          "weighted_total": 加权总分
        }}

        直接输出JSON，不要加代码块标记。不要输出改进建议、优点总结或改写示范。
    """)

    system = ("你是严格公正的文学评审，评分必须与真实读者评分对齐。先找问题，再给分，"
              f"不得礼貌性高分。直接输出JSON。{genre_note_for(category)}")

    # 解析失败要重试：通义在试水里因 JSON 截断丢了 2 本。
    # 截断多半是举证太长撑爆输出，所以重试时压缩举证要求而不是原样重问。
    for attempt in range(3):
        user = prompt if attempt == 0 else (
            prompt + "\n\n注意：上一次输出的JSON不完整。请严格控制长度——"
            "每条 failures 不超过30字，success 不超过30字，确保JSON完整闭合。"
        )
        raw = call_json(client, system=system, user=user, ledger=ledger,
                        label="评审打分", model=model, extra_body=extra_body)
        try:
            return json.loads(clean_json(raw))
        except json.JSONDecodeError:
            if attempt == 2:
                print(" [JSON解析连续3次失败]", end="")
                return {}
    return {}


def enumerate_all_cells(min_books: int = 15) -> list[tuple[str, str]]:
    """列出全部 品类×频道 格子。男频 Nanpin、女频 Nvpin 各 18 个目录，
    其中 3 个品类名两边都有（悬疑脑洞 / 游戏体育 / 科幻末世），算两个独立格子。"""
    cells = []
    for src in ("Nanpin", "Nvpin"):
        root = Path(src)
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir() or d.name == "raw":
                continue
            rated = 0
            for f in d.glob("*.txt"):
                head = f.read_text(encoding="utf-8", errors="replace")[:2000]
                m = re.search(r"^评分[：:]\s*(.+)$", head, re.MULTILINE)
                if m:
                    try:
                        if float(m.group(1)) > 0:
                            rated += 1
                    except ValueError:
                        pass
            if rated - TOP_SOURCE_NOVELS - CALIBRATION_LOW_N >= min_books:
                cells.append((d.name, src))
    return cells


def _dup_categories() -> set[str]:
    """男女频同名的品类——它们的输出目录必须分开，否则互相覆盖。"""
    def names(src):
        root = Path(src)
        return {d.name for d in root.iterdir() if d.is_dir() and d.name != "raw"} if root.exists() else set()
    return names("Nanpin") & names("Nvpin")


def cell_dir(category: str, source_dir: Path) -> Path:
    """输出目录。同名品类（男女频各一套）加频道后缀，避免评分表和结果互相覆盖；
    其余保持原路径，不动已有数据。"""
    if category in _dup_categories():
        return OUTPUT_BASE / f"{category}__{source_dir.name}"
    return OUTPUT_BASE / category


def run_meta(model: str, workers: int, repeat: int) -> dict:
    """每次运行都带上身份信息。西方奇幻的 summary 与 iter 目录对不上，
    就是因为没有这个——事后无法判断哪份记录属于哪次跑。"""
    now = datetime.now(timezone.utc)
    seed = f"{now.isoformat()}|{model}|{workers}|{repeat}"
    return {
        "run_id": now.strftime("%Y%m%dT%H%M%SZ") + "-" + hashlib.sha1(seed.encode()).hexdigest()[:6],
        "run_utc": now.isoformat(timespec="seconds"),
        "reviewer_model": model,
        "chapters_scored": CHAPTERS_TO_SCORE,
        "workers": workers,
        "repeat_policy": f"抽样 {repeat} 次" if repeat > 1 else "每部 1 次",
        "script": Path(__file__).name,
    }


def pick_reliability_sample(held_out: list[dict], k: int) -> set[str]:
    """信度只需要一个子样本。原来的 --repeat 把 N 倍成本压在全样本上，
    但估计评审噪声不需要全样本——按评分排序均匀抽 k 部即可，覆盖整个分数跨度。"""
    if k <= 0 or k >= len(held_out):
        return {n["_path"].name for n in held_out} if k >= len(held_out) else set()
    ordered = sorted(held_out, key=lambda n: n["_rating"])
    step = len(ordered) / k
    return {ordered[int(i * step)]["_path"].name for i in range(k)}


def repeat_for(n: dict, repeat: int, reliability_files: set[str]) -> int:
    return repeat if n["_path"].name in reliability_files else 1


def run_cell(client, category: str, source_dir: Path, ledger: Ledger,
             dry_run: bool, limit: int, workers: int = 1, repeat: int = 1,
             reliability_k: int = 0, model: str | None = None) -> list[dict]:
    print(f"\n{'=' * 66}\n  {category}（{source_dir}）\n{'=' * 66}")

    out_dir = cell_dir(category, source_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    novels = load_category(category, source_dir)
    novels.sort(key=lambda n: n["_rating"], reverse=True)
    contaminated = {n["_path"].name for n in novels[:TOP_SOURCE_NOVELS]}
    contaminated |= {n["_path"].name for n in novels[-CALIBRATION_LOW_N:]}
    held_out = [n for n in novels if n["_path"].name not in contaminated]
    if limit:
        held_out = held_out[:limit]

    ratings = [n["_rating"] for n in held_out]
    print(f"  有评分 {len(novels)} 部，排除污染 {len(contaminated)} 部，留出 {len(held_out)} 部")
    print(f"  留出集真实评分 {min(ratings):.1f}–{max(ratings):.1f}，离散度 {statistics.pstdev(ratings):.3f}")

    reliability_files = pick_reliability_sample(held_out, reliability_k) if repeat > 1 else set()
    if reliability_files:
        print(f"  信度子样本 {len(reliability_files)} 部 × 重复 {repeat} 次"
              f"（其余每部 1 次，而非全样本 ×{repeat}）")

    rubrics_path = out_dir / "rubrics.json"
    print(f"  评分表：{'已有' if rubrics_path.exists() else '需归纳（+1 次调用）'}")

    n_calls = sum(repeat_for(n, repeat, reliability_files) for n in held_out)
    if dry_run:
        total = n_calls + (0 if rubrics_path.exists() else 1)
        print(f"  [dry-run] 预计调用 {total} 次（并发 {workers}），未调 API。")
        return []

    rubrics = derive_rubrics(client, category, out_dir, source_dir=source_dir)
    anchors = load_calibration_anchors(category, source_dir=source_dir)
    calibration_block = format_calibration_block(anchors)

    jobs = []
    for n in held_out:
        body = n["_path"].read_text(encoding="utf-8", errors="replace").split("=" * 20, 1)[-1]
        chapters = split_chapters(body, CHAPTERS_TO_SCORE)
        if not chapters:
            print(f"  跳过（切不出章节）：{n.get('title', '')[:26]}")
            continue
        for rep in range(repeat_for(n, repeat, reliability_files)):
            jobs.append((n, chapters, rep))

    print(f"  开跑：{len(jobs)} 次调用，并发 {workers}")

    def work(job):
        n, chapters, rep = job
        ev = evaluate_lean(client, chapters, rubrics, category,
                           calibration_block, ledger, model=model)
        return n, rep, ev

    raw_scores: dict[str, list[float]] = {}
    evidence: dict[str, dict] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(work, j): j for j in jobs}
        for fut in as_completed(futs):
            n = futs[fut][0]
            done += 1
            try:
                n, rep, ev = fut.result()
                score = float(ev.get("weighted_total") or 0)
            except Exception as e:
                print(f"  [{done}/{len(jobs)}] {n.get('title','')[:22]} → 失败 {type(e).__name__} {str(e)[:50]}")
                continue
            if not score:
                print(f"  [{done}/{len(jobs)}] {n.get('title','')[:22]} → 解析失败")
                continue
            key = n["_path"].name
            raw_scores.setdefault(key, []).append(score)
            evidence.setdefault(key, ev.get("scores", {}))
            print(f"  [{done}/{len(jobs)}] ★{n['_rating']} {n.get('title','')[:22]} → {score}")

    by_file = {n["_path"].name: n for n in held_out}
    results = []
    for key, scores in raw_scores.items():
        n = by_file[key]
        results.append({
            "title": n.get("title", ""), "file": key,
            "real_rating": n["_rating"],
            "rubric_score": statistics.mean(scores),
            "rubric_scores": scores,                       # 重复评分全部留档，用于算信度
            "dimensions": {k: v.get("score") for k, v in evidence[key].items()
                           if isinstance(v, dict)},
            "evidence": evidence[key],
        })

    if results:
        payload = {
            **run_meta(model or REVIEWER_MODEL, workers, repeat),
            "category": category, "source_dir": str(source_dir),
            "output_spec": "lean",
            "n": len(results), "excluded_contaminated": sorted(contaminated),
            "results": results,
        }
        out_path = out_dir / "validation_pilot.json"
        if out_path.exists():
            # 不静默覆盖——西方奇幻就是这样丢掉一次 5 轮跑的记录的
            stamp = payload["run_id"]
            out_path.rename(out_dir / f"validation_pilot_prev_{stamp}.json")
            print(f"  旧结果已改名留档：validation_pilot_prev_{stamp}.json")
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  已保存：{out_path}")
    return results


def summarize(label: str, results: list[dict]) -> None:
    if len(results) < 3:
        print(f"\n  {label}：有效样本 {len(results)} 个，不足以计算相关。")
        return
    real = [r["real_rating"] for r in results]
    got = [r["rubric_score"] for r in results]
    r_p, r_s = pearson(got, real), spearman(got, real)

    print(f"\n{'─' * 66}\n  {label}   n = {len(results)}")
    print(f"  真实评分   均值 {statistics.mean(real):.2f}  离散度 {statistics.pstdev(real):.3f}"
          f"  跨度 {min(real):.1f}–{max(real):.1f}")
    print(f"  评分表分数 均值 {statistics.mean(got):.2f}  离散度 {statistics.pstdev(got):.3f}"
          f"  跨度 {min(got):.1f}–{max(got):.1f}")
    print(f"  Pearson  r = {r_p:+.3f}")
    print(f"  Spearman ρ = {r_s:+.3f}")

    ranked = sorted(results, key=lambda r: r["real_rating"])
    k = max(1, len(ranked) // 3)
    lo = statistics.mean(r["rubric_score"] for r in ranked[:k])
    hi = statistics.mean(r["rubric_score"] for r in ranked[-k:])
    print(f"  真实低分组(n={k}) 评分表均分 {lo:.2f}　真实高分组(n={k}) {hi:.2f}　区分度 Δ = {hi - lo:+.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", action="append", default=[],
                    help="品类:语料目录，可重复。默认 民国言情:Nvpin 与 动漫衍生:Nanpin")
    ap.add_argument("--skip-done", action="store_true",
                    help="跳过已有 validation_pilot.json 的格子（中断后续跑用，避免重复付费）")
    ap.add_argument("--all", action="store_true",
                    help="跑全部品类×频道格子（男频 Nanpin + 女频 Nvpin），共 37 个")
    ap.add_argument("--min-books", type=int, default=15,
                    help="留出集少于这么多本的格子跳过（样本太小无法算相关）")
    ap.add_argument("--limit", type=int, default=0, help="每品类只评前 N 部")
    ap.add_argument("--provider", default="openai", choices=list(PROVIDERS),
                    help="评审模型来源。中文模型快且便宜，适合全量跑")
    ap.add_argument("--model", default="", help="覆盖该 provider 的默认模型")
    ap.add_argument("--workers", type=int, default=1,
                    help="并发数。o3 单次近 10 分钟，840 本串行要 6 天，务必调高")
    ap.add_argument("--repeat", type=int, default=1,
                    help="信度子样本的重复评分次数（不作用于全样本）")
    ap.add_argument("--reliability-k", type=int, default=10,
                    help="每品类抽多少部进信度子样本，配合 --repeat 使用")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.all:
        cells = enumerate_all_cells(args.min_books)
    elif args.cell:
        cells = [tuple(c.split(":", 1)) for c in args.cell]
    else:
        cells = DEFAULT_CELLS
    base, env_key, default_model = PROVIDERS[args.provider]
    model = args.model or default_model

    client = None
    if not args.dry_run:
        import os
        key = os.environ.get(env_key)
        if not key:
            sys.exit(f"[ERROR] 未设置 {env_key}")
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=base)
        print(f"  评审模型：{args.provider} / {model}   并发 {args.workers}")
        # review_loop.derive_rubrics 内部读模块级的 REVIEWER_MODEL（"o3"），
        # 不接受模型参数。不改它就会拿通义的 client 去请求 o3 → 404。
        # 这里显式改写该常量，让归纳评分表和打分用同一个模型——
        # 否则两者口径也不一致（一个 o3 归纳、一个通义打分）。
        import review_loop as _rl
        _rl.REVIEWER_MODEL = model

    ledger = Ledger(model)
    pooled, per_cell = [], {}
    t0 = time.time()
    failed_cells = []
    for category, src in cells:
        # 跑完的格子结果已存盘，重跑等于白花钱。
        # 但必须同时核对评审模型：只看文件在不在，会把别的模型打的旧结果
        # 留在数据集里（民国言情就是试水时 o3 打的），混评审的样本不能合并分析。
        done_path = cell_dir(category, Path(src)) / "validation_pilot.json"
        if args.skip_done and done_path.exists():
            try:
                prev_model = json.loads(done_path.read_text(encoding="utf-8")).get("reviewer_model")
            except json.JSONDecodeError:
                prev_model = None
            if prev_model == model:
                print(f"\n  跳过（已由 {model} 打过）：{category}（{src}）")
                continue
            print(f"\n  重跑：{category}（{src}）—— 旧结果出自 {prev_model or '未知模型'}，"
                  f"与本次 {model} 不一致，不能混入同一数据集")
        # 一个格子失败不能带走整批——37 个格子串行，早期崩溃会浪费后面全部。
        try:
            res = run_cell(client, category, Path(src), ledger, args.dry_run, args.limit,
                           workers=args.workers, repeat=args.repeat,
                           reliability_k=args.reliability_k, model=model)
        except Exception as e:
            print(f"\n  [格子失败] {category}（{src}）：{type(e).__name__} {str(e)[:160]}")
            print("  继续下一个格子。")
            failed_cells.append((category, src, f"{type(e).__name__}: {str(e)[:120]}"))
            continue
        if res:
            per_cell[category] = res
            pooled.extend(res)

    if args.dry_run:
        return

    print(f"\n{'=' * 66}\n  结果\n{'=' * 66}")
    for category, res in per_cell.items():
        summarize(category, res)

    if len(per_cell) > 1 and len(pooled) >= 3:
        # 品类内标准化后合并：消掉品类间的均值差，只留下要研究的关系
        z_real, z_got = [], []
        for res in per_cell.values():
            real = [r["real_rating"] for r in res]
            got = [r["rubric_score"] for r in res]
            mr, sr = statistics.mean(real), statistics.pstdev(real)
            mg, sg = statistics.mean(got), statistics.pstdev(got)
            if sr and sg:
                z_real += [(v - mr) / sr for v in real]
                z_got += [(v - mg) / sg for v in got]
        if len(z_real) >= 3:
            print(f"\n{'─' * 66}\n  合并（品类内标准化）  n = {len(z_real)}")
            print(f"  Pearson  r = {pearson(z_got, z_real):+.3f}")
            print(f"  Spearman ρ = {spearman(z_got, z_real):+.3f}")

    if failed_cells:
        print(f"\n{'=' * 66}\n  失败的格子（{len(failed_cells)} 个，未计入结果）\n{'=' * 66}")
        for cat, src, err in failed_cells:
            print(f"  {cat}（{src}）：{err}")
        print("\n  修好后用 --skip-done 续跑，已完成的格子不会重复付费。")

    print(f"\n{'=' * 66}\n  实际用量\n{'=' * 66}")
    print(ledger.report())
    print(f"  耗时 {(time.time() - t0) / 60:.1f} 分钟")
    c = ledger.cost
    if c and ledger.calls:
        print(f"\n  单本 {c[1]} {c[0] / ledger.calls:.4f} → 840 本外推 ≈ {c[1]} {c[0] / ledger.calls * 840:,.2f}")


if __name__ == "__main__":
    main()
