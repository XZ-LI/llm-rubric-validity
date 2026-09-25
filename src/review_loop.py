#!/usr/bin/env python3
"""
review_loop.py — Self-improving novel generation loop for 西方奇幻.

Pipeline (×5 iterations):
  1. Derive rubrics from top-rated source novels (cached after first run)
  2. Generate: outline + 3 chapters using current system prompt
  3. Evaluate: reviewer GPT scores each chapter against rubrics
  4. Improve: editor GPT rewrites the system prompt based on low scores
  5. Repeat with improved prompt

Output saved to:
  review_loop/西方奇幻/rubrics.json
  review_loop/西方奇幻/iter_1/system_prompt.txt
  review_loop/西方奇幻/iter_1/novel.txt
  review_loop/西方奇幻/iter_1/evaluation.json
  ...
  review_loop/西方奇幻/iter_5/...
  review_loop/西方奇幻/summary.json   ← scores across all iterations

Usage:
  export OPENAI_API_KEY=sk-...
  python3 review_loop.py
  python3 review_loop.py --category 东方仙侠 --iterations 3
"""

import argparse
import json
import os
import re
import sys
import textwrap
import time
from pathlib import Path
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] openai not installed. Run: pip install openai")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_CATEGORY   = "西方奇幻"
DEFAULT_ITERATIONS = 5
MAX_ITERATIONS     = 10   # safety cap to prevent infinite loop
TARGET_SCORE       = 9.0  # keep regenerating until this score is reached
CHAPTERS_PER_ITER  = 3
WRITER_MODEL       = "o3"
REVIEWER_MODEL     = "o3"
SOURCE_DIR         = Path("Nanpin")  # overridden by --source-dir
OUTPUT_BASE        = Path("review_loop")
TOP_SOURCE_NOVELS      = 5     # how many top-rated source novels to derive rubrics from
CALIBRATION_HIGH_N     = 3     # top-rated novels shown to reviewer as 9+ anchors
CALIBRATION_LOW_N      = 2     # lowest-rated novels shown as reference floor
CHAPTER_CHARS          = 2500  # target chars per chapter
CREATIVITY_SOURCE_N    = 12    # high-rated novels used to learn premise mechanics

BRAINSTORM_CATEGORIES = {
    "古言脑洞", "现言脑洞", "悬疑脑洞", "历史脑洞", "玄幻脑洞", "都市脑洞",
}

# ── Rubric dimensions (seeded; GPT enriches from source novels) ───────────────

SEED_RUBRICS = [
    "钩子强度",      # Hook: does the first line grab attention?
    "核心矛盾",      # Core conflict: is there a sharp, escalating conflict that feels real?
    "节奏变化",      # Pacing variation: mix of sentence lengths?
    "人物鲜活度",    # Character vividness: distinct voices, reactions?
    "画面感",        # Visual imagery: sensory details, can you see it?
    "对话自然度",    # Dialogue authenticity: feels real, not stiff?
    "悬念设计",      # Cliffhanger: chapter ending compels next chapter?
    "情感感染力",    # Emotional resonance: does the scene make readers feel something specific?
    "世界观融合",    # World-building: woven in naturally, not info-dumped?
    "戏剧性反转",    # Dramatic reversal: at least one moment that surprises or shocks?
]

# ── Post-processing ───────────────────────────────────────────────────────────

def strip_technique_labels(text: str) -> str:
    _KW = (
        r"超短句|短句|长句渲染|长句描写|长句|对话爆发|对话推进|短景|节奏变化|节奏切换|氛围渲染|氛围烘托"
        r"|悬念收尾|悬念定格|悬念钩子|单独成行|内心独白|伏笔埋设|伏笔|反转|心理描写|动作描写"
        r"|情绪渲染|情感张力|画面感|感官描写|视觉冲击|开篇钩子|钩子|爽点|爆发点"
    )
    lines_out = []
    for line in text.splitlines():
        stripped = line.strip().lstrip("—─\u2014\u2500 \t")
        if re.fullmatch(rf"({_KW})[。．：:，,\s]*", stripped):
            continue
        m = re.match(rf"^([—─\u2014\u2500\s]*({_KW})[：:——\-\u2014]+)(.*)", line)
        if m and m.group(3).strip():
            lines_out.append(m.group(3).strip())
            continue
        clean = re.sub(rf"[（(【\[][^）)】\]]*({_KW})[^）)】\]]*[）)】\]]", "", line)
        lines_out.append(clean)
    return "\n".join(lines_out)


# ── Shared utilities ──────────────────────────────────────────────────────────

def call(client: OpenAI, model: str, system: str, user: str,
         temp: float = 0.7, label: str = "") -> str:
    if label:
        print(f"    [{label}]", end=" ", flush=True)
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    if not any(m in model for m in ("o1", "o3")):
        kwargs["temperature"] = temp
    for attempt in range(6):
        try:
            resp = client.chat.completions.create(**kwargs).choices[0].message.content
            if label:
                print("完成")
            return resp
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = 60 * (attempt + 1)
                print(f"\n  [限速] 等待{wait}秒后重试（第{attempt+1}次）...", flush=True)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("API 调用失败：超过最大重试次数")


def clean_json(raw: str) -> str:
    return re.sub(r"^```json\s*|^```\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)


def fallback_evaluation(raw: str) -> dict:
    """Recover the most important reviewer fields when the JSON is malformed."""
    result = {"scores": {}, "weighted_total": 0, "strengths": [], "top_issues": [], "raw": raw}

    m = re.search(r'"weighted_total"\s*:\s*(\d+(?:\.\d+)?)', raw)
    if m:
        result["weighted_total"] = float(m.group(1))

    strengths_m = re.search(r'"strengths"\s*:\s*\[(.*?)\]\s*,\s*"top_issues"', raw, re.S)
    if strengths_m:
        result["strengths"] = re.findall(r'"([^"\n][^"]{8,}?)"', strengths_m.group(1))[:2]

    issues_block = re.search(r'"top_issues"\s*:\s*\[(.*)\]\s*\}?$', raw, re.S)
    if issues_block:
        blocks = re.findall(r'\{(.*?)\}', issues_block.group(1), re.S)
        for block in blocks[:3]:
            issue = {}
            for key in ("problem", "prompt_fix", "rewrite_example"):
                km = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', block, re.S)
                if km:
                    issue[key] = km.group(1).replace('\\"', '"').replace("\\n", "\n")
            if issue:
                result["top_issues"].append(issue)

    return result


def parse_novel_txt(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    sep = "=" * 20
    parts = text.split(sep, 1)
    header_raw = parts[0]
    body = parts[1] if len(parts) > 1 else ""

    def field(label):
        m = re.search(rf"^{label}[：:]\s*(.+)$", header_raw, re.MULTILINE)
        return m.group(1).strip() if m else ""

    desc_m = re.search(r"简介[：:]\s*\n([\s\S]*?)(?:\n={10,}|$)", header_raw)
    description = desc_m.group(1).strip() if desc_m else ""

    ch1_m = re.search(r"第\s*1\s*章\s*(.+?)\n([\s\S]*?)(?=\n第\s*2\s*章|\Z)", body)
    ch1_text = ch1_m.group(2).strip()[:2000] if ch1_m else body[:2000]

    seq = []
    for n in range(2, 5):
        m = re.search(rf"第\s*{n}\s*章\s*(.+?)\n([\s\S]{{100,}}?)(?=\n第\s*{n+1}\s*章|\Z)", body)
        if m:
            seq.append({"chapter": n, "title": m.group(1).strip(), "opening": m.group(2).strip()[:500]})

    return {
        "title": field("书名"), "rating": field("评分"),
        "tags": field("标签"), "description": description,
        "ch1_text": ch1_text, "ch_sequence": seq,
    }


def load_source_novels(category: str, top_n: int = TOP_SOURCE_NOVELS,
                       source_dir: Path = SOURCE_DIR) -> list[dict]:
    cat_dir = source_dir / category
    if not cat_dir.exists():
        print(f"[WARN] 未找到品类目录：{cat_dir}")
        return []
    novels = []
    for f in cat_dir.glob("*.txt"):
        try:
            novels.append(parse_novel_txt(f))
        except Exception:
            pass
    novels.sort(key=lambda x: float(x.get("rating") or 0), reverse=True)
    return novels[:top_n]


def load_all_source_novels_deep(category: str, source_dir: Path = SOURCE_DIR,
                                char_budget: int = 37_000) -> list[dict]:
    """Load ALL novels in the category with longer excerpts, respecting a char budget.

    Used for --learn-only deep mode to fill o3's full context window.
    Each novel gets: full ch1 (up to 3000 chars) + ch2-5 openings (up to 800 chars each).
    Novels are sorted by rating descending so highest-rated are fully included first.
    """
    cat_dir = source_dir / category
    if not cat_dir.exists():
        return []

    novels = []
    for f in cat_dir.glob("*.txt"):
        try:
            d = parse_novel_txt(f)
            # Also grab more chapter content from the raw file
            raw = f.read_text(encoding="utf-8", errors="replace")
            sep = "=" * 20
            body = raw.split(sep, 1)[1] if sep in raw else raw
            extra_seqs = []
            for n in range(2, 6):
                m = re.search(
                    rf"第\s*{n}\s*章\s*(.+?)\n([\s\S]{{100,}}?)(?=\n第\s*{n+1}\s*章|\Z)", body)
                if m:
                    extra_seqs.append({
                        "chapter": n,
                        "title": m.group(1).strip(),
                        "opening": m.group(2).strip()[:800],
                    })
            d["ch_sequence"] = extra_seqs
            d["ch1_text"] = d.get("ch1_text", "")[:3000]
            novels.append(d)
        except Exception:
            pass
    novels.sort(key=lambda x: float(x.get("rating") or 0), reverse=True)

    # Trim to fit within char_budget
    used = 0
    result = []
    for n in novels:
        size = (len(n.get("description", "")) + len(n.get("ch1_text", "")) +
                sum(len(ch["opening"]) for ch in n.get("ch_sequence", [])) + 200)
        if used + size > char_budget:
            break
        result.append(n)
        used += size
    print(f"  深度学习：加载 {len(result)}/{len(novels)} 部小说（约{used//1000}K字符）")
    return result


def load_calibration_anchors(category: str, source_dir: Path = SOURCE_DIR) -> dict:
    """Load high-rated and low-rated source novels to anchor the reviewer's score scale."""
    cat_dir = source_dir / category
    if not cat_dir.exists():
        return {"high": [], "low": []}
    novels = []
    for f in cat_dir.glob("*.txt"):
        try:
            novels.append(parse_novel_txt(f))
        except Exception:
            pass
    novels.sort(key=lambda x: float(x.get("rating") or 0), reverse=True)
    return {
        "high": novels[:CALIBRATION_HIGH_N],    # e.g. 9.3-9.6
        "low":  novels[-CALIBRATION_LOW_N:],    # e.g. 6.4-6.9
    }


def load_creativity_sources(category: str, source_dir: Path = SOURCE_DIR,
                            top_n: int = CREATIVITY_SOURCE_N) -> list[dict]:
    """Load high-rated source novels with enough material to infer premise mechanics."""
    cat_dir = source_dir / category
    if not cat_dir.exists():
        return []

    novels = []
    for f in cat_dir.glob("*.txt"):
        try:
            novels.append(parse_novel_txt(f))
        except Exception:
            pass

    novels.sort(key=lambda x: float(x.get("rating") or 0), reverse=True)
    return novels[:top_n]


def format_calibration_block(anchors: dict) -> str:
    """Format calibration anchors for injection into the reviewer prompt."""
    lines = ["## 评分校准参考（真实读者评分，用于对齐你的评分标准）\n"]
    lines.append("### 高分范例（9分以上的真实评分作品）")
    for n in anchors.get("high", []):
        rating = n.get("rating", "?")
        lines.append(f"\n《{n['title']}》 — 真实评分 ★{rating}")
        lines.append(f"简介：{n.get('description','')[:200]}")
        lines.append(f"第一章开篇：\n{n.get('ch1_text','')[:800]}")
        lines.append("─" * 40)

    lines.append("\n### 低分范例（7分以下的真实评分作品，作为对比下限）")
    for n in anchors.get("low", []):
        rating = n.get("rating", "?")
        lines.append(f"\n《{n['title']}》 — 真实评分 ★{rating}")
        lines.append(f"第一章开篇：\n{n.get('ch1_text','')[:400]}")
        lines.append("─" * 40)

    lines.append("""
评分原则：
- 你的评分必须与上方真实读者评分对齐
- 9分以上 = 达到高分范例的品质水准（罕见，只有真正出色才给）
- 7-8分 = 达到平均发布水准，有明显优缺点
- 6分以下 = 低于低分范例，有严重问题
- 绝对禁止"礼貌性"评分：如果文字没有达到标准，不得给高分
""")
    return "\n".join(lines)


# ── Step 1: Derive rubrics ────────────────────────────────────────────────────

def derive_rubrics(client: OpenAI, category: str, out_dir: Path,
                   source_dir: Path = SOURCE_DIR, deep: bool = False) -> list[dict]:
    rubrics_path = out_dir / "rubrics.json"
    if rubrics_path.exists() and not deep:
        print("  已有评分标准，跳过推导。")
        return json.loads(rubrics_path.read_text(encoding="utf-8"))

    if deep:
        print(f"  深度学习模式：读取全部 {category} 小说，填满 o3 上下文窗口...")
        novels = load_all_source_novels_deep(category, source_dir=source_dir)
    else:
        print("  正在从高评分源小说推导评分标准...")
        novels = load_source_novels(category, source_dir=source_dir)

    if not novels:
        rubrics = [{"name": r, "description": "", "weight": 1} for r in SEED_RUBRICS]
        rubrics_path.write_text(json.dumps(rubrics, ensure_ascii=False, indent=2), encoding="utf-8")
        return rubrics

    examples = ""
    for n in novels:
        examples += f"\n{'─'*40}\n"
        examples += f"《{n['title']}》 ★{n['rating']} [{n['tags']}]\n"
        examples += f"简介：{n['description'][:300]}\n"
        examples += f"第一章开篇：\n{n['ch1_text']}\n"
        for ch in n.get("ch_sequence", []):
            examples += f"\n第{ch['chapter']}章《{ch['title']}》开篇：\n{ch['opening']}\n"
        examples += "\n"

    prompt = textwrap.dedent(f"""\
        以下是{category}品类中{'全部' if deep else '评分最高的'}{len(novels)}部网络小说的章节节选，
        按评分从高到低排列。请仔细阅读每部作品的开篇风格、节奏特征、角色塑造和世界构建方式。

        {examples}

        ## 任务
        基于以上{'所有' if deep else '高质量'}范文，归纳出评价一部{category}网络小说章节质量的10个评分维度。

        要求：
        1. 维度名称4字以内，直接反映{category}品类的核心质量要素
        2. 评分说明具体可操作：说清楚"什么样的文字得9-10分"和"什么样的得5分以下"
        3. 权重1-3（越影响读者体验越高）
        4. high_score_example 和 low_score_example 须直接引用上方原文中的具体句子或段落作为例证
        5. 针对{category}品类的特有规律（如境界升级节奏、机缘代价、宗门政治等），不要泛泛而谈

        从以下种子维度出发，结合品类特色调整或替换：
        {', '.join(SEED_RUBRICS)}

        以JSON数组输出，每项格式：
        {{"name": "维度名", "description": "评分说明（含具体标准）", "weight": 权重数字,
          "high_score_example": "原文引用+说明为何高分",
          "low_score_example": "原文引用或常见失分写法+说明为何低分"}}

        直接输出JSON数组，不要加代码块标记。
    """)

    raw = call(client, REVIEWER_MODEL,
        system="你是专业的文学评论家和网络小说研究者，擅长分析中文网文的质量维度。深度阅读所有范文后再归纳。",
        user=prompt, temp=0.3, label="推导评分标准")

    try:
        rubrics = json.loads(clean_json(raw))
    except json.JSONDecodeError:
        rubrics = [{"name": r, "description": "", "weight": 1} for r in SEED_RUBRICS]

    rubrics_path.write_text(json.dumps(rubrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  评分标准已保存：{rubrics_path}（{len(rubrics)}个维度）")
    return rubrics


def derive_creativity_guide(client: OpenAI, category: str, out_dir: Path,
                            source_dir: Path = SOURCE_DIR,
                            force: bool = False) -> str:
    """Learn high-rated premise mechanics and turn them into anti-clone rules."""
    if category not in BRAINSTORM_CATEGORIES:
        return ""

    guide_path = out_dir / "creativity_guide.md"
    if guide_path.exists() and not force:
        return guide_path.read_text(encoding="utf-8")

    novels = load_creativity_sources(category, source_dir=source_dir)
    if not novels:
        return ""

    examples = []
    for n in novels:
        seq = "\n".join(
            f"第{ch['chapter']}章开篇：{ch['opening'][:260]}"
            for ch in n.get("ch_sequence", [])[:3]
        )
        examples.append(textwrap.dedent(f"""\
            ────────────────────────────────
            《{n['title']}》 ★{n.get('rating','?')} [{n.get('tags','')}]
            简介：{n.get('description','')[:500]}
            第一章开篇：
            {n.get('ch1_text','')[:1200]}
            {seq}
        """))

    prompt = textwrap.dedent(f"""\
        以下是{category}品类评分最高的一批真实作品。请不要复述剧情，请抽象它们为什么有脑洞感。

        {''.join(examples)}

        ## 任务
        输出一份“{category}创意机制学习指南”，用于指导下一轮原创生成。

        必须包含：
        1. 高分样本的“脑洞发动机”类型归纳：例如心声公开、身份错置、制度漏洞、天幕直播、玄学规则、历史人物误读等。
        2. 每类发动机如何持续制造章节冲突：写成可操作规则，不要泛泛夸奖。
        3. 与现有高分样本相似度过高的禁区：列出必须避开的换皮模式。
        4. 新创意生成矩阵：至少给出8条“古代制度/权力场 + 异常规则 + 私人代价 + 关系误差”的组合方式。
        5. 新奇度评分表：0-10分，说明什么样的设定才算9分以上。
        6. 大纲生成前的强制流程：先生成6个候选脑洞，逐一说明与样本库的差异，再选择最不撞题的一个。

        输出须是简体中文 Markdown。重点是“可执行的创意规则”，不是文学评论。
    """)

    guide = call(
        client, REVIEWER_MODEL,
        system="你是中文网文创意策划和类型研究者，擅长从高分作品中抽象叙事机制，并设计避免撞题的原创脑洞。",
        user=prompt, temp=0.3, label="学习脑洞机制"
    )
    guide_path.write_text(guide, encoding="utf-8")
    print(f"  创意机制指南已保存：{guide_path}")
    return guide


def ensure_brainstorm_rubrics(rubrics: list[dict], category: str) -> list[dict]:
    """Add explicit novelty dimensions for brainstorm categories, even with cached rubrics."""
    if category not in BRAINSTORM_CATEGORIES:
        return rubrics

    existing = {r.get("name", "") for r in rubrics}
    additions = []
    if "脑洞鲜" not in existing and "创意鲜" not in existing:
        additions.append({
            "name": "脑洞鲜",
            "description": (
                "9-10分：核心设定是可持续规则机制，明显不同于样本库常见组合，"
                "能持续制造误会、代价、权力连锁和章节冲突。"
                "5分以下：只是穿越/重生/系统/读心/玄学等标签换皮，情节仍是常规宅斗宫斗。"
            ),
            "weight": 3,
        })
    if "撞题度" not in existing:
        additions.append({
            "name": "撞题度",
            "description": (
                "9-10分：能说明自己与高分样本的差异，规则、代价、权力场、人物关系均有新组合。"
                "5分以下：明显近似全家偷听心声、幼崽团宠、天幕剧透、女扮男科举、玄学升官等热门模板。"
            ),
            "weight": 3,
        })
    if "长线性" not in existing:
        additions.append({
            "name": "长线性",
            "description": (
                "9-10分：核心规则能自然推演前20章，每次使用都带来升级后的限制、代价或隐患。"
                "5分以下：脑洞只服务开篇噱头，第一章后就退化成普通打脸。"
            ),
            "weight": 2,
        })

    return additions + rubrics


# ── Step 2: Generate outline + 3 chapters ────────────────────────────────────

CATEGORY_META = {
    "西方奇幻": {
        "en": "Western Fantasy",
        "tropes": [
            "Knight or warrior protagonist in a medieval magical world",
            "Magic systems, noble houses, guilds, and ancient prophecies",
            "Dragons, demons, elves, dwarves as key factions or allies",
            "Protagonist gains hidden power or a mysterious system",
            "Political intrigue between kingdoms, churches, and magic academies",
        ],
        "hook": "the protagonist's first encounter with magic or a life-or-death trial",
    },
    "玄幻言情": {
        "en": "Xuanhuan Romance",
        "tropes": [
            "独立清醒女主，拒绝恋爱脑，以实力和智谋主导命运",
            "修仙/异世界体系，灵力/丹药/法器构成世界运行规则",
            "穿书/穿越/重生，女主利用先知优势反套路颠覆剧情",
            "系统加持或特殊体质（废材逆袭、归虚道骨等隐藏设定）",
            "反派/男配围绕女主形成张力，情感线服务于成长弧",
        ],
        "hook": "女主初醒发现身份危机，用反常识手段破局第一个死局",
    },
    "古风世情": {
        "en": "Ancient Style Slice-of-Life",
        "tropes": [
            "古代市井/宅斗背景，女主以智谋在权力夹缝中生存",
            "家族争产、婚姻政治、主仆博弈构成核心矛盾",
            "女主表面低调内里清醒，用柔软手段化解强硬困局",
            "情感线以信任与背叛为轴，男主须通过考验赢得女主",
        ],
        "hook": "女主在婚嫁/分家/危机时刻展现超出他人预期的清醒",
    },
    "快穿": {
        "en": "Quick Transmigration",
        "tropes": [
            "女主穿越多个小世界执行任务（攻略、复仇、打脸）",
            "每个世界有独立男主，情感线快速建立但各具特色",
            "打脸爽文节奏，女主一入场即碾压原剧情中的炮灰命运",
            "系统/空间/金手指辅助，世界观规则清晰",
        ],
        "hook": "女主降临新世界第一秒，凭直觉识破最大危机并逆转",
    },
    "宫斗宅斗": {
        "en": "Palace/Household Intrigue",
        "tropes": [
            "后宫或大宅权谋，女主在多方势力夹击中步步为营",
            "人物关系错综复杂，盟友随时可能变为对手",
            "情报战、心理战、资源战三线并行",
            "女主表面顺从，暗中布局，最终以少胜多",
        ],
        "hook": "女主在看似必败的局面中，用一步棋扭转所有人的预判",
    },
    "现言脑洞": {
        "en": "Contemporary Romance with Twists",
        "tropes": [
            "现代都市背景，奇特脑洞设定颠覆日常认知",
            "女主有隐藏技能或特殊视角，反差萌或高冷实力派",
            "男女主关系因误会/契约/竞争而充满张力",
            "幽默轻松或甜虐并存，节奏明快",
        ],
        "hook": "开篇脑洞设定直接上线，女主用非常规手段应对非常规困境",
    },
    "职场婚恋": {
        "en": "Workplace Romance",
        "tropes": [
            "职场权谋与情感纠葛并行，女主独立上进有专业能力",
            "男主是上司/竞争对手/合作伙伴，情感线在工作张力中升温",
            "反霸凌/反PUA，女主用能力而非柔弱赢得尊重",
            "现实细节扎实，职场规则准确",
        ],
        "hook": "女主在职场危机中展示出让所有人刮目相看的专业能力",
    },
    "古言脑洞": {
        "en": "Ancient Chinese Comedy/Twist",
        "tropes": [
            "核心卖点是新奇规则机制，而不是单纯穿越、重生、系统、读心标签",
            "古代制度必须被脑洞规则改写：礼法、宗族、科举、和亲、朝堂、祭祀、史书、婚契等都可成为发动机",
            "女主靠理解规则和利用信息差破局，幽默、爽感、情绪都从规则连锁反应中自然生长",
            "每次使用金手指都要留下私人代价或权力隐患，推动后续章节升级",
            "关系张力来自误听、误判、同盟债、身份悖论或立场冲突，感情线不得盖过脑洞主线",
        ],
        "hook": "开篇直接展示一个违反古代常识的规则，并让女主用它破解第一个死局，同时制造更大的隐患",
    },
    "科幻末世": {
        "en": "Sci-Fi Post-Apocalypse",
        "tropes": [
            "末世/星际背景，超能力/机甲/变异体构成战力体系",
            "女主有被低估的稀有能力，在危机中逐渐觉醒",
            "生存压力与人性考验并行，团队信任是核心主题",
            "情感线在极端环境中加速，生死相依催生羁绊",
        ],
        "hook": "末世第一天，女主用旁人都没想到的方式活过了最危险的一关",
    },
    "传统玄幻": {
        "en": "Traditional Xuanhuan Cultivation",
        "tropes": [
            "完整修炼境界体系（练气→筑基→金丹→元婴→化神…），升级过程有明确代价与门槛",
            "主角初始设定被低估（废灵根/杂灵根/凡人），靠奇遇/悟性/意志逆袭",
            "宗门/家族/帝国三层权力结构，主角在夹缝中步步为营",
            "炼丹/炼器/阵法/功法作为硬核世界观支柱，技术细节增强沉浸感",
            "机缘与代价对等：每次大机缘伴随对等风险或付出，不可无代价开挂",
            "配角群像鲜明：师门、对手、盟友各有独立动机，非纯工具人",
            "战斗描写重视力量对比与绝境反转，战前铺垫决定战后爽感",
        ],
        "hook": "主角在最绝望的境地（废灵根/被逐出师门/命悬一线）触碰到改变命运的第一条线索",
    },
}

CHAPTER_TYPE_CYCLE = [
    ("战斗章", "快节奏动作：短句切换，动作细腻，力量对比强烈"),
    ("人物章", "慢节奏内省：内心独白，角色关系深化，情绪细腻"),
    ("感情章", "浪漫张力：暧昧对话，欲言又止，感官细节"),
]


def build_style_examples_block(anchors: dict) -> str:
    """Format top-rated chapter excerpts as direct style examples for the writer."""
    if not anchors.get("high"):
        return ""
    lines = ["## 风格标杆（高评分真实原著节选——模仿其笔法、节奏和开篇方式）\n"]
    lines.append("以下是真实读者高度认可的作品。仔细分析它们的：句式节奏、开篇切入方式、")
    lines.append("对话与动作的穿插、画面感的构建方式——你的文字须达到同等水准。\n")
    for n in anchors["high"]:
        lines.append(f"{'─'*50}")
        lines.append(f"《{n['title']}》  真实评分 ★{n['rating']}")
        lines.append(f"简介：{n.get('description','')[:150]}")
        lines.append(f"\n第一章节选：\n{n.get('ch1_text','')[:1000]}")
        for ch in n.get("ch_sequence", [])[:1]:
            lines.append(f"\n第{ch['chapter']}章开篇（注意章节间的风格切换）：\n{ch['opening'][:400]}")
    lines.append(f"{'─'*50}\n")
    return "\n".join(lines)


FEMALE_CATEGORIES = {
    "玄幻言情", "古风世情", "快穿", "宫斗宅斗", "现言脑洞",
    "职场婚恋", "科幻末世", "豪门总裁", "年代", "种田",
    "星光璀璨", "青春甜宠", "民国言情", "女频悬疑", "女频衍生",
    "悬疑脑洞", "游戏体育", "古言脑洞",
}


def build_generation_system_prompt(extra_rules: str = "", style_block: str = "",
                                   category: str = "西方奇幻",
                                   creativity_guide: str = "") -> str:
    is_female = category in FEMALE_CATEGORIES
    if is_female:
        base = textwrap.dedent(f"""\
            你是一位专业的中文网络小说作家，擅长{category}品类（女频）。
            你的文字节奏明快、情感细腻、完全契合女频读者审美。

            ## 核心矛盾（最高优先级）
            【矛盾必须尖锐】每章必须有一个让读者揪心的核心冲突：
                ① 外部冲突：强权压迫、生死威胁、利益对撞——对手须有真实威胁感，不可轻易被化解
                ② 内部冲突：女主的欲望与代价之间的撕裂感（想活命 vs 不能暴露身份；想离开 vs 放不下某人）
                ③ 关系冲突：信任与背叛、债务与情感、立场对立的两人不得不合作
            【矛盾升级】每章结尾的困境必须比开篇更难，不许"轻松化解"——要么付出代价，要么留下隐患
            【戏剧性时刻】每章至少一个让读者屏息的反转或揭露，用具体细节而非空洞描述触发情绪

            ## 情绪感染力（核心要求）
            【让读者感同身受】用感官细节代替情绪标签：
                ✗ 错误："她很害怕。"
                ✓ 正确："手心渗汗，指节按在桌沿，硬撑着没让身体抖出来。"
            【情绪节点】每章须有一个让读者产生强烈情绪共鸣的时刻（愤怒/心疼/爽快/揪心），
                       该时刻须由具体场景触发，不得靠旁白说明
            【留白张力】重要情感用"没说出口的话"和"细微动作"传递，比直白表达更有穿透力

            ## 女主塑造（核心原则）
            【独立清醒】女主不恋爱脑，不靠男人解围——靠智谋、技能、先知优势主动破局。
            【反套路】至少在前三章颠覆一个常见套路，让读者感到惊喜。
            【成长弧】每章女主须有可见的成长或决策，不做被动受害者。

            ## 脑洞新奇度（脑洞品类最高优先级）
            【先有发动机】脑洞不能只是身份标签（穿越、重生、系统、读心）。必须是一个能持续制造误会、选择、代价和权力连锁反应的规则机制。
            【避免换皮】不得直接复刻高分样本的核心组合：全家偷听心声保忠烈、幼崽团宠救全家、天幕剧透始皇、女扮男科举名臣、单纯玄学算命升官等。可以学习结构，但必须替换规则、代价、权力场和关系误差。
            【四件套】每个新设定必须同时包含：古代制度/权力场 + 异常规则 + 私人代价 + 关系误差。缺一项则视为脑洞不足。
            【长线推演】核心规则必须能自然推演出至少20章事件，且每次使用都带来新隐患，不能一章用完。

            ## 穿书/穿越品类核心风格（玄幻言情必须遵守）
            【现代视角吐槽】穿越女主用现代人思维审视古风/修仙世界是本品类最大卖点。
                           女主的内心吐槽、社畜梗、现代词汇与古风场景形成的反差感是读者最爱的笑点和爽点。
                           ✓ 正确："这不就是职场PUA吗？我跳槽还来不及，谁要死磕。"
                           ✓ 正确："系统这个甲方改需求改得比我前老板还狠。"
                           ✗ 错误：全程保持古风沉浸感，不允许现代视角介入。
            【风格杂糅是特色】现代吐槽＋古风场景＋修仙术语并存，是这个品类的标志性风格，不是缺陷。
                           评审若将此视为"跳戏"属于误判——这正是读者买单的核心体验。

            ## 写作规范（必须遵守）
            【白话口语】全程白话文，禁用"甚是""不知为何""却见""只见""正欲"等文言词。
            　　✗ 错误："他甚是疑惑，不知为何她竟如此镇定。"
            　　✓ 正确："他看不懂她——这女人到底在想什么？"
            【虚词自然】多用"着""了""和""也""呢""吧"让句子读起来更流畅自然。
            【段落】每段2-4句，绝不写大段连续文字。重要时刻单独成行。
            【节奏变化】每章节奏须与上一章明显不同：
                       战斗/爽文章 → 短句密集，动作连贯，节奏急促
                       人物/感情章 → 长句渲染氛围，内心独白细腻，节奏舒缓
                       揭秘/转折章 → 短长交替，信息密度高，悬念层叠
            【对话】每句对话单独成段，附带动作或表情描写，口吻各异体现性格。
            【章节结尾】必须以悬念、反转或情绪高峰收尾，让读者无法停下来。
            【外形描写】角色首次登场必须有外形描写：面容、身材、气质（生动比喻）。
            【禁止】不写章节总结、元评论、"本章完"字样。
            【禁止】绝对不得在正文中标注写作技巧名称，如"超短句""长句渲染""悬念定格"等——直接执行，不做标注。

            所有输出一律使用简体中文。
        """)
    else:
        base = textwrap.dedent(f"""\
            你是一位专业的中文网络小说作家，擅长{category}品类。
            你的文字节奏快、画面感强、深度契合读者口味。

            ## 写作规范（必须遵守）
            【段落】每段2-4句，绝不写大段连续文字。重要时刻单独成行。
            【节奏变化】每章必须混合三种句式：
                       ① 超短句（≤5字）冲击感：他愣住了。血。沉默。
                       ② 中等对话（10-20字/句）推动场景
                       ③ 长句（30字以上）渲染氛围
                       同一节奏连续超过5句须切换。
            【内心独白】用直接引语：他心想：「这不可能。」
            【对话】每句对话单独成段，附带动作或表情描写。
            【章节结尾】必须以悬念、反转或紧迫感收尾，让读者无法停下来。
            【外形描写】角色首次登场必须有外形描写：面容、身材、气质、穿着，生动比喻。
                       女性角色须突出美貌特色，不同女性美丽风格各异。
            【感情线】同一时间最多两条感情线并行。
            【禁止】不写章节总结、元评论、"本章完"字样。
            【禁止】绝对不得在正文中标注写作技巧名称，如"超短句""长句渲染""短景＋心理＋动作""单独成行"等——直接执行该技巧，不做任何说明或标注。

            所有输出一律使用简体中文。
        """)
    if style_block:
        base += f"\n{style_block}\n"
    if creativity_guide:
        base += f"\n## 高分样本创意机制学习指南（必须用于避免撞题）\n{creativity_guide}\n"
    if extra_rules:
        base += f"\n## 本轮改进重点（根据上轮评审反馈）\n{extra_rules}\n"
    return base


def generate_outline(client: OpenAI, system_prompt: str, category: str) -> str:
    meta = CATEGORY_META.get(category, {"en": category, "tropes": [], "hook": "主角的命运转折"})
    tropes = "\n".join(f"  - {t}" for t in meta["tropes"])
    is_female = category in FEMALE_CATEGORIES
    is_brainstorm = category in BRAINSTORM_CATEGORIES

    if is_female:
        romance_section = textwrap.dedent("""\
            3. **感情线设计（女频标准）**
               - 男性角色A：姓名、外形气质、性格、登场章节、与女主的关系张力（不得一见钟情）
               - 男性角色B：姓名、外形气质、性格、登场章节（晚于A）、关系基调
               - 女主情感原则：先欣赏实力→再建立信任→最后心动（不可颠倒顺序）
        """)
        reader_label = "女频"
    else:
        romance_section = textwrap.dedent("""\
            3. **感情线设计（必须包含）**
               同一时间最多两位恋人并行，三条线错峰登场：
               - 恋人A：姓名、【外形美貌风格】、性格、登场章节、感情基调
               - 恋人B：姓名、【外形美貌风格】、性格、登场章节（晚于A）、感情基调
               - 恋人C：姓名、【外形美貌风格】、性格、登场章节（晚于B）、感情基调
        """)
        reader_label = "男频"

    creativity_section = ""
    if is_brainstorm:
        creativity_section = textwrap.dedent("""\
            0. **脑洞候选池（必须先做，不能跳过）**
               先生成6个彼此差异极大的核心脑洞候选，每个候选必须包含：
               - 候选名：一句话卖点
               - 古代制度/权力场：礼法、宗族、科举、和亲、朝堂、祭祀、史官、户籍、婚契、军功等
               - 异常规则：这个世界哪里不正常，规则如何触发，谁能感知
               - 私人代价：女主每次利用规则会失去什么、暴露什么或欠下什么
               - 关系误差：谁误会了女主，谁听到/看到/继承了错误信息，如何制造喜剧或危机
               - 长线冲突：该规则如何撑起前20章，而不是一章用完
               - 撞题风险：与高分样本中哪些常见模式相似，如何主动避开
               - 新奇度评分：0-10分，低于8.5的候选不能入选

               然后选择新奇度最高、撞题风险最低的1个候选，作为最终策划案。不得选择“全家偷听女主心声”“幼崽团宠救全家”“始皇/天幕剧透”“女扮男科举名臣”“玄学算命一路升官”的换皮版本。

        """)

    prompt = textwrap.dedent(f"""\
        品类：{category}（{meta['en']}）

        品类惯例：
        {tropes}

        为该品类创作一部原创网络小说的完整策划案。

        ===== 平台投稿信息 =====
        【书本名称】（10字以内，吸引眼球）
        【目标读者】{reader_label}
        【作品标签】3-5个，用"|"分隔
        【主角名1】（主角全名）
        【主角名2】（第一位重要男性角色全名）
        【作品简介】50-300字，第一句直接抓住读者
        =======================

        {creativity_section}
        1. **核心设定** — 一句话概括世界观与核心矛盾。脑洞品类须明确“规则机制、触发方式、限制、代价、长期隐患”。

        2. **主角档案**
           - 基本信息：姓名、出身、初始处境、隐藏实力
           - 外形描写：面容/身材/气质/穿着（生动比喻）
           - 性格成长弧：起点缺陷 → 触发事件 → 终点蜕变

        {romance_section}
        4. **大环境迁移（至少3个）**
           每个环境有不同考验，逼迫主角以新方式成长：
           - 环境一：地点、考验类型、主角收获
           - 环境二：地点、考验类型、主角收获
           - 环境三：地点、考验类型、主角收获

        5. **反派/核心矛盾** — 谁阻挡主角，为什么难以对抗。反派必须能利用同一套规则反制女主，不能只是无脑坏。

        6. **章节大纲** — 第1-3章，每章：标题 + 两句内容 + 一句结尾钩子
           （标注每章类型：战斗章/人物章/感情章/揭秘章/喘息章/转折章）

        7. **脑洞保鲜表**
           列出前20章每5章一次的规则升级、代价升级、关系误差升级，保证创意不是开篇噱头。

        开篇围绕：{meta['hook']}。至少颠覆一个常见套路。用简体中文输出。
    """)

    return call(client, WRITER_MODEL, system=system_prompt, user=prompt,
                temp=0.5, label="生成大纲")


def generate_chapter(client: OpenAI, system_prompt: str, outline: str,
                     chapter_n: int, prev_chapters: list[str]) -> str:
    ch_type_name, ch_type_instr = CHAPTER_TYPE_CYCLE[(chapter_n - 1) % len(CHAPTER_TYPE_CYCLE)]

    recent = ""
    if prev_chapters:
        recent = f"\n\n── 上一章节（仅供风格参考）──\n{prev_chapters[-1][:1500]}\n"

    avoid = ""
    if prev_chapters:
        last = prev_chapters[-1][:400]
        used = []
        if any(k in last for k in ["挥拳", "出剑", "攻击", "对决"]):
            used.append("战斗开篇")
        if any(k in last for k in ["心想", "脑海", "回忆"]):
            used.append("内心独白开篇")
        first_char = last.strip()[:1]
        if first_char in ("「", "\u201c", "\u201d"):
            used.append("对话开篇")
        if used:
            avoid = f"【须避开上章已用手法】：{'、'.join(used)}\n"

    prompt = textwrap.dedent(f"""\
        以下是已确定的小说策划案：

        {outline}
        {recent}

        ## 任务：写第{chapter_n}章

        【本章类型】{ch_type_name} — {ch_type_instr}
        {avoid}
        要求：
        - 字数：{CHAPTER_CHARS}字以上
        - 第一句直接进入场景，无铺垫废话
        - 本章结构、情绪基调须与上一章明显不同
        - 混合三种句式节奏（超短句 / 对话 / 长句渲染）
        - 章节结尾制造强烈悬念或反转

        请直接输出第{chapter_n}章正文（含章节标题"第{chapter_n}章 XXXX"）。
    """)

    return call(client, WRITER_MODEL, system=system_prompt, user=prompt,
                temp=0.9, label=f"写第{chapter_n}章")


# ── Step 3: Evaluate ──────────────────────────────────────────────────────────

def evaluate_chapters(client: OpenAI, chapters: list[str],
                      rubrics: list[dict], category: str,
                      calibration_block: str = "") -> dict:
    rubric_str = "\n".join(
        f"  {i+1}. 【{r['name']}】（权重{r.get('weight',1)}）：{r.get('description','')}"
        for i, r in enumerate(rubrics)
    )
    chapters_str = ""
    for i, ch in enumerate(chapters, 1):
        chapters_str += f"\n\n=== 第{i}章 ===\n{ch[:2000]}\n"

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
        第三步：给出一条针对写作指令的具体改进建议（精确到句式/开篇方式/具体手法）。

        最后：
        - 加权总分（严格按高分范例标准，不得因"整体不错"而虚高）
        - 最突出优点1-2条（必须有原文依据）
        - 最需要改进的问题2-3条，每条附：具体系统提示规则 + 一段示范改写（展示改好之后应该是什么样子）

        以JSON格式输出：
        {{
          "scores": {{"维度名": {{"score": 分数, "failures": ["失败例1", "失败例2"], "success": "好的地方", "suggestion": "系统提示改进建议"}}}},
          "weighted_total": 加权总分,
          "strengths": ["优点1（含原文）", "优点2（含原文）"],
          "top_issues": [
            {{"problem": "问题描述", "prompt_fix": "建议在系统提示中加入的具体规则", "rewrite_example": "改写后应该是什么样子（100字示范）"}}
          ]
        }}

        直接输出JSON，不要加代码块标记。
    """)

    genre_note = ""
    if category in FEMALE_CATEGORIES:
        genre_note = (
            "\n\n## 女频品类特别说明（评分前必读）\n"
            "【现代视角吐槽是优点】穿书/穿越女主用现代词汇、社畜梗、互联网用语吐槽古风世界，"
            "是玄幻言情/快穿等品类的核心卖点和标志性风格。"
            "请勿将'现代吐槽与古风基调混杂'列为缺陷——这正是读者付费阅读的核心体验。\n"
            "【扣分项】：女主恋爱脑、被动受害、文言文腔、信息堆砌无节奏。\n"
            "【加分项】：现代视角反差笑点、独立破局、爽感节奏、角色对话各有个性。"
        )
    if category in BRAINSTORM_CATEGORIES:
        genre_note += (
            "\n\n## 脑洞品类特别说明（评分前必读）\n"
            "【脑洞鲜是硬门槛】如果核心设定只是穿越/重生/系统/读心/玄学算命等常见标签的换皮，"
            "即使文笔顺畅，脑洞相关维度不得高于7分，加权总分原则上不得高于8分。\n"
            "【高分标准】9分以上必须具备可持续的规则机制：古代制度或权力场被异常规则改写，"
            "女主每次使用规则都会产生私人代价、关系误差或权力连锁反应。\n"
            "【撞题扣分】明显近似高分样本的组合，如全家偷听心声救忠烈、幼崽团宠改命、"
            "始皇/天幕剧透、女扮男科举名臣、玄学断案升官，须明确扣分并给出避开方式。"
        )
    raw = call(client, REVIEWER_MODEL,
        system=f"你是严格公正的文学评审，评分必须与真实读者评分对齐。先找问题，再给分，不得礼貌性高分。直接输出JSON。{genre_note}",
        user=prompt, temp=0.2, label="评审打分")

    try:
        result = json.loads(clean_json(raw))
    except json.JSONDecodeError:
        result = fallback_evaluation(raw)
    return result


# ── Step 4: Improve system prompt ─────────────────────────────────────────────

def improve_system_prompt(client: OpenAI, current_prompt: str,
                          evaluation: dict, iteration: int) -> str:
    issues = evaluation.get("top_issues", [])
    scores = evaluation.get("scores", {})

    low_scores = sorted(
        [(k, v["score"]) for k, v in scores.items() if isinstance(v, dict)],
        key=lambda x: x[1]
    )[:3]

    issues_str = "\n".join(
        f"  - 问题：{iss.get('problem','')}\n"
        f"    规则建议：{iss.get('prompt_fix','')}\n"
        f"    改写示范：{iss.get('rewrite_example','（无）')}"
        for iss in issues
    )
    low_str = "\n".join(
        f"  - {name}（得分{score}/10）\n"
        f"    失败例：{scores.get(name,{}).get('failures',[''])[0]}\n"
        f"    改进：{scores.get(name,{}).get('suggestion','')}"
        for name, score in low_scores
    )

    prompt = textwrap.dedent(f"""\
        这是第{iteration}轮生成循环的评审结果。

        ## 当前系统提示
        {current_prompt}

        ## 评审发现的主要问题（含改写示范）
        {issues_str}

        ## 得分最低的维度（含失败原文）
        {low_str}

        ## 任务
        根据以上评审反馈，输出一组新增写作规则（不要重复现有规则，只写新增部分）。
        要求：
        1. 针对每个低分维度，写1-2条具体可操作的规则（用【】标注）
        2. 每条规则附上✗/✓示范对比，格式：
           【规则名】规则说明
           ✗ 低分写法：...（直接引用失败例原文）
           ✓ 高分写法：...（引用改写示范）
        3. 规则须精确到句式/开篇方式/具体手法，不写泛泛建议
        4. 标注"（第{iteration}轮新增规则）"

        只输出新增规则文本，不要输出完整系统提示，不要加任何解释。
    """)

    improved = call(client, REVIEWER_MODEL,
        system="你是专业的提示工程师，擅长将文学评审反馈转化为精确的写作指令，包含✗/✓示范对比。",
        user=prompt, temp=0.3, label="改进系统提示")
    return improved


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_loop(category: str, target_score: float = TARGET_SCORE,
             max_iter: int = MAX_ITERATIONS,
             source_dir: Path = SOURCE_DIR,
             resume: bool = False) -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] 未设置 OPENAI_API_KEY")
        sys.exit(1)
    client = OpenAI(api_key=api_key)

    out_dir = OUTPUT_BASE / category
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  评审循环：{category}  目标分数≥{target_score}  上限{max_iter}轮")
    print(f"  素材来源：{source_dir}/  输出目录：{out_dir}/")
    print(f"{'='*60}\n")

    # Step 1: Derive rubrics (cached) + load calibration anchors
    rubrics = derive_rubrics(client, category, out_dir, source_dir=source_dir)
    rubrics = ensure_brainstorm_rubrics(rubrics, category)
    creativity_guide = derive_creativity_guide(client, category, out_dir, source_dir=source_dir)
    anchors = load_calibration_anchors(category, source_dir=source_dir)
    calibration_block = format_calibration_block(anchors)
    style_block = build_style_examples_block(anchors)
    high_ratings = [n["rating"] for n in anchors["high"] if n.get("rating")]
    low_ratings  = [n["rating"] for n in anchors["low"]  if n.get("rating")]
    print(f"  校准锚点：高分 {len(anchors['high'])} 部（{', '.join(high_ratings)}）"
          f"  低分 {len(anchors['low'])} 部（{', '.join(low_ratings)}）")

    # Resume mode: load state from existing summary + latest iter system_prompt
    all_scores = []
    it = 0
    foundation_prompt = None  # when set, subsequent iterations build on this instead of the base template
    if resume:
        summary_path = out_dir / "summary.json"
        if summary_path.exists():
            prev = json.loads(summary_path.read_text(encoding="utf-8"))
            all_scores = prev.get("scores", [])
            it = prev.get("total_rounds", 0)
            print(f"  [续跑] 从第{it}轮继续，已有得分：{[s['score'] for s in all_scores]}")
        # Find the latest completed iteration's system prompt.
        latest_iter_dir = out_dir / f"iter_{it}"
        if latest_iter_dir.exists() and (latest_iter_dir / "system_prompt.txt").exists():
            system_prompt = (latest_iter_dir / "system_prompt.txt").read_text(encoding="utf-8")
            if creativity_guide and "高分样本创意机制学习指南" not in system_prompt:
                system_prompt += f"\n## 高分样本创意机制学习指南（必须用于避免撞题）\n{creativity_guide}\n"
            foundation_prompt = system_prompt  # preserve as base for all subsequent improvements
            print(f"  [续跑] 使用系统提示：iter_{it}/system_prompt.txt")
        else:
            system_prompt = build_generation_system_prompt(
                style_block=style_block, category=category,
                creativity_guide=creativity_guide)
    else:
        system_prompt = build_generation_system_prompt(
            style_block=style_block, category=category,
            creativity_guide=creativity_guide)

    while True:
        it += 1
        print(f"\n{'─'*50}")
        print(f"  第 {it} 轮  （目标 {target_score}/10，上限 {max_iter} 轮）")
        print(f"{'─'*50}")

        iter_dir = out_dir / f"iter_{it}"
        iter_dir.mkdir(exist_ok=True)
        (iter_dir / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")

        # Step 2: Generate
        print("  [生成阶段]")
        outline = generate_outline(client, system_prompt, category)
        time.sleep(2)

        chapters = []
        for ch_n in range(1, CHAPTERS_PER_ITER + 1):
            ch_text = generate_chapter(client, system_prompt, outline, ch_n, chapters)
            ch_text = strip_technique_labels(ch_text)
            chapters.append(ch_text)
            time.sleep(2)

        novel_text = outline + "\n\n" + "=" * 60 + "\n\n"
        for ch in chapters:
            novel_text += ch + "\n\n"
        (iter_dir / "novel.txt").write_text(novel_text, encoding="utf-8")
        print(f"  已保存：{iter_dir}/novel.txt")

        # Step 3: Evaluate (with real-rating calibration anchors)
        print("  [评审阶段]")
        evaluation = evaluate_chapters(client, chapters, rubrics, category, calibration_block)
        (iter_dir / "evaluation.json").write_text(
            json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")

        total = evaluation.get("weighted_total", 0)
        all_scores.append({"iteration": it, "score": total})
        print(f"  本轮得分：{total:.1f}/10  （目标：{target_score}）")

        strengths = evaluation.get("strengths", [])
        issues    = evaluation.get("top_issues", [])
        if strengths:
            print(f"  优点：{strengths[0]}")
        if issues:
            print(f"  主要问题：{issues[0].get('problem','')}")

        # Check stopping conditions
        if total >= target_score:
            print(f"\n  达到目标分数 {total:.1f} >= {target_score}，循环结束！")
            break
        if it >= max_iter:
            print(f"\n  已达上限 {max_iter} 轮，停止循环（最高分 {max(s['score'] for s in all_scores):.1f}）。")
            break

        # Step 4: Improve prompt and regenerate
        print(f"  [改进阶段]  得分 {total:.1f} < {target_score}，继续改进...")
        improved_rules = improve_system_prompt(client, system_prompt, evaluation, it)
        if foundation_prompt:
            # Resume mode: preserve the foundation prompt, append new rules on top
            system_prompt = foundation_prompt + f"\n## 第{it}轮新增改进规则（来自评审反馈）\n{improved_rules}\n"
        else:
            # Rebuild with style block preserved (improvement step returns rules only)
            system_prompt = build_generation_system_prompt(
                extra_rules=improved_rules, style_block=style_block,
                category=category, creativity_guide=creativity_guide)
        time.sleep(2)

    # Save summary
    best = max(all_scores, key=lambda x: x["score"])
    summary = {
        "category":       category,
        "target_score":   target_score,
        "total_rounds":   it,
        "scores":         all_scores,
        "best_iteration": best["iteration"],
        "best_score":     best["score"],
        "score_trend":    [s["score"] for s in all_scores],
        "goal_reached":   best["score"] >= target_score,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    trend = " → ".join(f"{s:.1f}" for s in summary["score_trend"])
    print(f"\n{'='*60}")
    print(f"  循环完成！共 {it} 轮")
    print(f"  得分趋势：{trend}")
    print(f"  最佳轮次：第{best['iteration']}轮  得分：{best['score']:.1f}/10")
    print(f"  最佳小说：{out_dir}/iter_{best['iteration']}/novel.txt")
    print(f"  汇总报告：{out_dir}/summary.json")
    print(f"{'='*60}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Self-improving novel generation review loop")
    parser.add_argument("--category", "-c", default=DEFAULT_CATEGORY,
        help=f"Novel category (default: {DEFAULT_CATEGORY})")
    parser.add_argument("--target", "-t", type=float, default=TARGET_SCORE,
        help=f"Keep regenerating until this score is reached (default: {TARGET_SCORE})")
    parser.add_argument("--max-iter", "-n", type=int, default=MAX_ITERATIONS,
        help=f"Safety cap on iterations (default: {MAX_ITERATIONS})")
    parser.add_argument("--source-dir", "-s", default=None,
        help="Source novel library directory (default: Nanpin for male, Nvpin for female categories)")
    parser.add_argument("--resume", "-r", action="store_true",
        help="Resume from last completed iteration, using iter_{n+1}/system_prompt.txt as starting prompt")
    parser.add_argument("--learn-only", action="store_true",
        help="Only derive rubrics and print style examples from source novels, then exit.")
    args = parser.parse_args()

    # Auto-select source dir based on category if not specified
    female_cats = {
        "玄幻言情", "古风世情", "快穿", "宫斗宅斗", "现言脑洞",
        "职场婚恋", "科幻末世", "豪门总裁", "年代", "种田",
        "星光璀璨", "青春甜宠", "民国言情", "女频悬疑", "女频衍生",
        "悬疑脑洞", "游戏体育", "古言脑洞",
    }
    if args.source_dir:
        source_dir = Path(args.source_dir)
    elif args.category in female_cats:
        source_dir = Path("Nvpin")
        print(f"[自动选择] 女频品类，使用素材库：Nvpin/")
    else:
        source_dir = Path("Nanpin")

    if args.learn_only:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("[ERROR] 未设置 OPENAI_API_KEY")
            sys.exit(1)
        client = OpenAI(api_key=api_key)
        out_dir = Path("review_loop") / args.category
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  学习阶段：从 {source_dir}/{args.category}/ 推导评分标准")
        print(f"{'='*60}\n")

        rubrics = derive_rubrics(client, args.category, out_dir, source_dir=source_dir, deep=True)
        rubrics = ensure_brainstorm_rubrics(rubrics, args.category)
        creativity_guide = derive_creativity_guide(
            client, args.category, out_dir, source_dir=source_dir, force=True)
        anchors = load_calibration_anchors(args.category, source_dir=source_dir)

        print(f"\n── 推导出的评分维度（共{len(rubrics)}个）──")
        for r in rubrics:
            w = r.get("weight", 1)
            print(f"  【{r['name']}】(权重{w})  {r.get('description','')[:80]}")
            if r.get("high_score_example"):
                print(f"    ✓ 高分特征：{r['high_score_example'][:60]}")
            if r.get("low_score_example"):
                print(f"    ✗ 低分特征：{r['low_score_example'][:60]}")

        print(f"\n── 校准锚点 ──")
        print(f"  高分参考：{[n['title'] + ' ★' + n['rating'] for n in anchors['high']]}")
        print(f"  低分参考：{[n['title'] + ' ★' + n['rating'] for n in anchors['low']]}")
        if creativity_guide:
            print(f"\n── 脑洞机制指南预览 ──")
            print(creativity_guide[:600])
        print(f"\n  评分标准已保存 → {out_dir}/rubrics.json")
        print(f"  准备好后运行完整循环：")
        print(f"    python3 review_loop.py --category {args.category} --max-iter 10\n")
        return

    run_loop(args.category, target_score=args.target, max_iter=args.max_iter,
             source_dir=source_dir, resume=args.resume)


if __name__ == "__main__":
    main()
