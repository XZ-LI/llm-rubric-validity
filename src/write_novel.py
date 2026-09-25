#!/usr/bin/env python3
"""
write_novel.py — Long-form novel writer with hierarchical memory.

Maintains consistency across hundreds of chapters using four memory layers:

  Layer 1  Story Bible   — world rules, all character profiles, overall arc
                           Always in context. Updated every 10 chapters.
  Layer 2  Arc Plan      — detailed plan for the current 30-chapter arc
                           Auto-refreshed when an arc completes.
  Layer 3  Summary Log   — 2-3 sentence compression of every chapter written
                           Grows permanently; 100 chapters ≈ 5k tokens.
  Layer 4  Recent Prose  — last 2 full chapters for style + scene continuity

Project files (novels/<title>/):
  bible.json        — story bible (world, characters, overall arc)
  arc_plan.txt      — current arc plan
  summaries.jsonl   — one JSON line per chapter (永久记忆)
  state.json        — progress tracker
  chapters/         — full chapter text (001.txt, 002.txt, …)

Commands:
  init     Initialize project from a generate_novel.py output file
  write    Write the next chapter (use --count N to write N in sequence)
  status   Show current progress and memory sizes
  new-arc  Manually trigger a new arc plan
  update-bible  Manually trigger a bible update from recent summaries

Usage:
  python3 write_novel.py init --from generated/西方奇幻/novel.txt
  python3 write_novel.py write
  python3 write_novel.py write --count 10
  python3 write_novel.py status
  python3 write_novel.py new-arc

Requirements:
  pip install openai
  export OPENAI_API_KEY=sk-...
"""

import argparse
import json
import os
import sys
import time
import textwrap
from pathlib import Path
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] openai not installed. Run: pip install openai")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_MODEL        = "o3"
NOVELS_DIR           = Path("novels")
CHAPTER_TARGET_CHARS = 3000   # target length per chapter in Chinese characters
ARC_LENGTH           = 30     # chapters per arc before auto-refresh
BIBLE_UPDATE_EVERY   = 10     # chapters between automatic bible updates
RECENT_CHAPTERS_N    = 2      # how many full chapters to include as recent prose
SUMMARY_MAX_CHARS    = 150    # max chars per chapter summary
POV_ROTATION_EVERY   = 5      # every Nth chapter shifts to secondary character POV

# Solution 1 — Chapter type rotation cycle
CHAPTER_TYPE_CYCLE = [
    ("战斗章",  "快节奏动作场面：短句切换镜头，动作描写细腻，力量对比强烈，紧张感贯穿全章"),
    ("人物章",  "慢节奏内省：大量内心独白，角色关系深化，情绪细腻，揭示性格层次"),
    ("感情章",  "浪漫张力：暧昧对话、欲言又止、感官细节（气味/触感/眼神），感情推进一步"),
    ("揭秘章",  "信息揭露：读者提前于主角获得关键信息，悬念层层剥开，伏笔兑现"),
    ("喘息章",  "幽默或生活化场景：笑点或温情时刻，节奏放松，为下一波高潮积蓄情绪能量"),
    ("转折章",  "颠覆预期：重大反转或抉择，改变故事走向，结尾让读者目瞪口呆"),
]


# ── File helpers ──────────────────────────────────────────────────────────────

def project_dir(title: str) -> Path:
    safe = title.replace("/", "_").replace(" ", "_").replace("《", "").replace("》", "")
    return NOVELS_DIR / safe


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_summaries(proj: Path) -> list[dict]:
    f = proj / "summaries.jsonl"
    if not f.exists():
        return []
    return [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_summary(proj: Path, entry: dict) -> None:
    f = proj / "summaries.jsonl"
    with f.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_chapter(proj: Path, n: int) -> str:
    f = proj / "chapters" / f"{n:03d}.txt"
    return f.read_text(encoding="utf-8") if f.exists() else ""


def save_chapter(proj: Path, n: int, text: str) -> Path:
    d = proj / "chapters"
    d.mkdir(exist_ok=True)
    f = d / f"{n:03d}.txt"
    f.write_text(text, encoding="utf-8")
    return f


def load_state(proj: Path) -> dict:
    f = proj / "state.json"
    if not f.exists():
        return {"current_chapter": 0, "arc_number": 1, "arc_start": 1}
    return load_json(f)


def save_state(proj: Path, state: dict) -> None:
    save_json(proj / "state.json", state)


def load_arc_plan(proj: Path) -> str:
    f = proj / "arc_plan.txt"
    return f.read_text(encoding="utf-8") if f.exists() else ""


def save_arc_plan(proj: Path, text: str) -> None:
    (proj / "arc_plan.txt").write_text(text, encoding="utf-8")


# ── Bible formatter ───────────────────────────────────────────────────────────

def format_bible(bible: dict) -> str:
    """Render the story bible as readable text for prompt injection."""
    lines = []
    lines.append(f"【书名】《{bible.get('title', '未知')}》  品类：{bible.get('category', '')}")
    lines.append(f"【主题】{bible.get('theme', '')}")
    lines.append("")

    world = bible.get("world", {})
    lines.append("── 世界设定 ──")
    lines.append(f"背景：{world.get('setting', '')}")
    lines.append(f"力量体系：{world.get('power_system', '')}")
    rules = world.get("key_rules", [])
    if rules:
        lines.append("核心规则：" + "；".join(rules))
    factions = world.get("factions", [])
    if factions:
        def _fmt_faction(f):
            if isinstance(f, dict):
                return f"{f['name']}（{f.get('description', '')}）"
            return str(f)
        lines.append("主要势力：" + "、".join(_fmt_faction(f) for f in factions))
    lines.append("")

    lines.append("── 角色档案 ──")
    raw_chars = bible.get("characters", [])
    if isinstance(raw_chars, dict):
        char_list = []
        for role_key, val in raw_chars.items():
            if isinstance(val, dict):
                if "name" in val:
                    # flat form: {"protagonist": {"name": "...", ...}}
                    val.setdefault("role", role_key)
                    char_list.append(val)
                else:
                    # nested form: {"supporting": {"莉安娜·黑荆": {...}}}
                    for char_name, char_data in val.items():
                        if isinstance(char_data, dict):
                            entry = {"name": char_name, "role": role_key, **char_data}
                        else:
                            entry = {"name": char_name, "role": role_key}
                        char_list.append(entry)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        item.setdefault("role", role_key)
                        char_list.append(item)
        raw_chars = char_list
    for ch in raw_chars:
        lines.append(f"【{ch.get('role','?')}】{ch['name']}")
        if ch.get("appearance"):
            lines.append(f"  外形：{ch['appearance']}")
        lines.append(f"  性格：{ch.get('personality','')}")
        if ch.get("growth_arc"):
            ga = ch["growth_arc"]
            lines.append(f"  成长弧：起点({ga.get('start','')}) → 终点({ga.get('end','')})  当前阶段：{ga.get('current_stage','')}")
        lines.append(f"  当前状态：{ch.get('current_state','')}")
        lines.append(f"  能力：{ch.get('abilities','')}")
        rels = ch.get("relationships", {})
        if rels:
            lines.append("  关系：" + "；".join(f"{k}→{v}" for k, v in rels.items()))
    lines.append("")

    romance = bible.get("romance_lines", [])
    if isinstance(romance, dict):
        romance = [{"name": k, **v} for k, v in romance.items() if isinstance(v, dict)]
    if romance:
        lines.append("── 感情线（错峰出现，同时最多两条） ──")
        for r in romance:
            desc = r.get('beauty_style') or r.get('descriptor', '')
            tone = r.get('tone') or r.get('chapters', '')
            lines.append(f"  {r.get('name','')}：{desc} | {tone} | 状态：{r.get('status','')}")
        lines.append("")

    envs = bible.get("environments", [])
    if isinstance(envs, dict):
        envs = [{"name": k, **v} for k, v in envs.items() if isinstance(v, dict)]
    if envs:
        lines.append("── 大环境迁移 ──")
        for e in envs:
            growth = e.get('growth') or e.get('gain', '')
            challenge = e.get('challenge_type') or e.get('theme', '')
            lines.append(f"  {e.get('name','')}：{challenge} | 主角所获成长：{growth} | 状态：{e.get('status','')}")
        lines.append("")

    arc = bible.get("overall_arc", {})
    lines.append("── 整体弧线 ──")
    lines.append(f"开篇：{arc.get('beginning','')}")
    lines.append(f"中段：{arc.get('midpoint','')}")
    lines.append(f"高潮：{arc.get('climax','')}")
    lines.append(f"结局：{arc.get('resolution','')}")

    return "\n".join(lines)


def format_summaries(summaries: list[dict], max_recent: int = 10) -> str:
    if not summaries:
        return "（暂无章节摘要）"
    lines = []
    cutoff = max(0, len(summaries) - max_recent)
    if cutoff > 0:
        lines.append(f"  （第1-{summaries[cutoff-1]['chapter']}章：已压缩入故事圣经）")
    for s in summaries[cutoff:]:
        lines.append(f"  第{s['chapter']:03d}章《{s.get('title','')}》：{s['summary']}")
    return "\n".join(lines)


def load_nanpin_style_examples(
    current_chapter: int,
    max_chars: int = 8000,
    category: str = "西方奇幻",
    window: int = 2,
    style_dir: str = "Nanpin",
) -> str:
    """Load as many chapter excerpts as fit within max_chars from top-rated novels.

    Samples chapters [current_chapter-window .. current_chapter+window] per novel,
    cycling through all top novels until the character budget is exhausted.
    style_dir: "Nanpin" for male novels, "Nvpin" for female novels.
    """
    import re as _re
    category_dir = Path(style_dir) / category
    if not category_dir.exists():
        return ""
    novels = []
    for txt_path in sorted(category_dir.glob("*.txt")):
        try:
            content = txt_path.read_text(encoding="utf-8")
            m = _re.search(r"评分：(\d+\.?\d*)", content[:500])
            score = float(m.group(1)) if m else 0.0
            novels.append((score, txt_path.stem, content))
        except Exception:
            continue
    novels.sort(reverse=True)
    if not novels:
        return ""

    ch_low  = max(1, current_chapter - window)
    ch_high = current_chapter + window
    header = (
        f"## 高分《{category}》第{ch_low}-{ch_high}章风格参考"
        f"（对应当前写作位置，仅学习节奏与笔法，勿抄内容）\n"
    )
    segments: list[str] = []
    used_chars = len(header)

    for score, title, content in novels:
        # Build chapter index: ch_n -> position (keep last occurrence = full text body)
        seen: dict[int, int] = {}
        for m in _re.finditer(r"^第(\d+)章[ \t]", content, _re.MULTILINE):
            ch_n = int(m.group(1))
            seen[ch_n] = m.start()

        candidates = [(n, p) for n, p in sorted(seen.items()) if ch_low <= n <= ch_high]
        if not candidates:
            all_sorted = sorted(seen.items())
            if not all_sorted:
                continue
            closest = min(all_sorted, key=lambda x: abs(x[0] - current_chapter))
            candidates = [closest]

        novel_lines: list[str] = [f"### 《{title}》（评分{score}）"]
        for ch_n, pos in candidates:
            # Use remaining budget split evenly across remaining candidates
            remaining_budget = max_chars - used_chars - len("\n".join(novel_lines))
            if remaining_budget <= 200:
                break
            excerpt_chars = min(remaining_budget, 1500)
            excerpt = content[pos: pos + excerpt_chars].strip()
            novel_lines.append(f"── 第{ch_n}章节选 ──")
            novel_lines.append(excerpt)
            novel_lines.append("")

        chunk = "\n".join(novel_lines) + "\n"
        if used_chars + len(chunk) > max_chars:
            break
        segments.append(chunk)
        used_chars += len(chunk)

    if not segments:
        return ""
    return header + "\n".join(segments)


# Novels index: pre-load content once, sample per-chapter at write time
def _load_nanpin_novels(category: str = "西方奇幻", top_n: int = 2) -> list:
    import re as _re
    category_dir = Path("Nanpin") / category
    if not category_dir.exists():
        return []
    novels = []
    for txt_path in sorted(category_dir.glob("*.txt")):
        try:
            content = txt_path.read_text(encoding="utf-8")
            m = _re.search(r"评分：(\d+\.?\d*)", content[:500])
            score = float(m.group(1)) if m else 0.0
            novels.append((score, txt_path.stem, content))
        except Exception:
            continue
    novels.sort(reverse=True)
    return novels[:top_n]

_NANPIN_NOVELS = _load_nanpin_novels()


# ── Prompt builders ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
    你是一位专业的中文网络小说作家，擅长维持跨数百章的剧情连贯性与风格统一性。

    ## 写作规范（必须遵守）
    【语言风格】用现代白话文写作，贴近口语节奏。
               ✗ 避免：文言化表达（如"吾""汝""尔等""此乃""当真""岂料""然则"）。
               ✗ 避免：过于书面的四字成语堆砌，让句子显得硬邦邦。
               ✓ 正确：人物说话像真实的人在说话，内心独白像读者自己会想的那种话。
               ✓ 正确：情绪和动作用口语化、接地气的方式呈现，读起来流畅自然。
               ✓ 正确：善用虚词让句子呼吸自然——"着"表示动作持续（他盯着她）、
                       "了"表示状态变化（她笑了，心里却凉了半截）、
                       "和"连接并列让节奏不生硬（血和汗混在一起）。
                       虚词用得自然，句子才有人味，不像机器在排列词语。
    【段落】每段2-4句，绝不写大段连续文字。重要时刻单独成行。
    【节奏变化】不同章节的句式主导风格必须不同，禁止每章都用相同公式：
               - 战斗章：以极短句为主（≤8字/句），偶尔插入一句长句渲染爆发瞬间。
               - 感情章：以中长句为主（15-30字），节奏舒缓，情绪在字里行间渗透。
               - 揭秘章：短句与长句交替，信息在节奏变化中释放张力。
               - 喘息章：口语化短句为主，轻松随意，间或一句带情绪的长句点睛。
               - 人物章：长句内省为主，让角色的思绪在句子里流动，不急不躁。
               - 转折章：前半舒缓，后半骤然加速，用节奏本身传递"剧变"感。
               章节开篇方式也须与上一章明显不同：上章动作开篇则本章用对话或环境；
               上章内心独白开篇则本章用外部事件切入。
    【内心独白】用直接引语，口语化：他心想：「这不对劲。」或「完了，跑不掉了。」
    【系统提示】格式：【叮咚：获得×××】
    【对话】每句对话单独成段，附带动作或表情描写。对话要像真人说话，不要像在背台词。
    【章节结尾】必须以悬念、反转或紧迫感收尾，让读者无法停下来。
    【外形描写】角色首次登场时必须有具体外形描写：面容、身材、气质、穿着，用细节和比喻让读者留下印象。
               女性角色须从主角视角感受其美貌，不同女性有各自鲜明的美丽风格，不可千篇一律。
    【性格成长】主角的性格须随剧情推进逐步蜕变，当前章节的行为和反应须与其所处成长阶段一致。
    【环境切换】当主角进入新的大环境时，须用笔墨渲染环境氛围，并让主角意识到旧经验在此处的局限。
    【感情线节奏】同一时间最多两条感情线并行，不得三位恋人同时缠绕主角。
    【禁止】不写章节总结、不写元评论、不写"本章完"字样。
    【章节过滤词】正文中禁止出现单独成行的"---悬"、"---章节悬"或"---章节结尾---"；破折号变体如"——悬"、"——章节悬"、"——章节结尾——"也禁止。
               需要悬念时直接写剧情画面，不要用这些标记行。
    【严禁出现写作技巧标注】正文中绝对不能出现任何写作手法的名称或说明，包括但不限于：
               "超短句""长句渲染""长句描写""悬念收尾""悬念""短景""心理描写""动作描写"
               "节奏变化""对话爆发""单独成行""伏笔""反转""氛围渲染"等词语作为标注出现。
               ——这些技巧只能被执行，不能被说出来。违反此规则视为严重错误。
    【一致性】严格遵守故事圣经中的人物性格、外形、世界规则、力量体系。

    所有输出一律使用简体中文。
""")


CHAPTER_LINE_FILTER_WORDS = {"---悬", "---章节悬", "---章节结尾---"}


def normalize_chapter_filter_line(line: str) -> str:
    """Normalize standalone filter lines so dash and punctuation variants are caught."""
    import re as _re

    text = line.strip()
    text = _re.sub(r"^[—–-]+", "---", text)
    text = _re.sub(r"[—–-]+$", "---", text)
    text = _re.sub(r"[。．.!！…\s]+$", "", text)
    return text


def filter_chapter_lines(text: str) -> str:
    """Remove whole-line chapter filter words after generation."""
    kept = []
    for line in text.splitlines():
        if normalize_chapter_filter_line(line) in CHAPTER_LINE_FILTER_WORDS:
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def strip_technique_labels(text: str) -> str:
    """Remove writing-technique annotations the model accidentally includes in prose."""
    import re as _re

    # Technique keyword pattern
    _KW = (
        r"超短句|短句起势|短句收击|短句|长句细描|长句渲染|长句描写|长句|对话爆发|对话推进|短景"
        r"|节奏三拍|节奏变化|节奏切换|氛围渲染|氛围烘托|情绪缓冲|节奏喘息|下一场景"
        r"|悬念收尾|悬念定格|悬念钩子|单独成行|内心独白|伏笔埋设|伏笔|反转|心理描写|动作描写"
        r"|情绪渲染|情感张力|画面感|感官描写|视觉冲击|开篇钩子|钩子|爽点|爆发点"
        r"|冲突动词|战句|律动"
    )

    # Whole-line preview pattern: "下一场景XXX：..." → delete entire line
    _PREVIEW_LINE = _re.compile(r"^[\s—─—─]*下一场景.{0,20}[：:].{0,200}$")
    # Separators that can follow a label keyword
    _SEP = r"[。！？，,：:—\-—─]+"

    lines_out = []
    for line in text.splitlines():
        stripped = line.strip().lstrip("—─—─ \t")

        # Case 0: "下一场景XXX：..." whole-line preview → delete
        if _PREVIEW_LINE.match(line.strip()):
            continue

        # Case 1+2 combined: label at line start (with optional leading dashes/spaces)
        # If real prose (≥4 chars) follows → keep prose only; else → delete the label line
        m = _re.match(rf"^([—─—─\s]*({_KW})[^\n]{{0,8}}{_SEP})(.*)", line)
        if m:
            rest = m.group(3)
            if len(rest.strip()) >= 4:
                lines_out.append(rest)
            # else: pure label/stub → drop
            continue

        # Case 3: bracket/paren annotations 【超短句】（长句渲染）
        line = _re.sub(rf"[【（][^】）]{{0,15}}({_KW})[^】）]{{0,15}}[】）]", "", line)

        # Case 4: label after sentence-end punct or comma: "啪！超短句。prose"
        # Replace label+sep with just the preceding punct
        line = _re.sub(
            rf"([。！？，,])({_KW})[^\n]{{0,8}}{_SEP}",
            r"\1",
            line,
        )

        # Case 5: trailing "下一场景..." appended to a prose line → strip
        line = _re.sub(r"下一场景[^\n]*$", "", line).rstrip()

        # Drop stub lines (< 5 chars) left by Case 4 stripping
        if len(line.strip()) < 5:
            continue

        lines_out.append(line)

    return filter_chapter_lines("\n".join(lines_out))


def get_chapter_type(chapter_n: int) -> tuple[str, str]:
    """Return (type_name, type_instruction) for this chapter number."""
    return CHAPTER_TYPE_CYCLE[(chapter_n - 1) % len(CHAPTER_TYPE_CYCLE)]


def extract_recent_techniques(recent_chapters: list[tuple[int, str]]) -> str:
    """Solution 2 — scan recent chapters for structural patterns to avoid repeating."""
    if not recent_chapters:
        return ""
    used = []
    for _, text in recent_chapters:
        t = text[:500]
        if any(k in t for k in ["挥拳", "出剑", "攻击", "战斗", "对决"]):
            used.append("战斗/对决开篇")
        if any(k in t for k in ["心想", "脑海", "回忆", "想起"]):
            used.append("内心独白开篇")
        if text.strip().startswith("「") or text.strip()[:1] in ("\u201c", "\u201d"):
            used.append("对话开篇")
        if any(k in t for k in ["突然", "没想到", "万万没想到"]):
            used.append("突然反转")
    if not used:
        return ""
    return "【近期已用手法，本章须避开】：" + "、".join(set(used))


def build_chapter_prompt(
    bible: dict,
    arc_plan: str,
    summaries: list[dict],
    recent_chapters: list[tuple[int, str]],
    next_chapter_n: int,
    next_chapter_hint: str,
    style_dir: str = "Nanpin",
) -> str:
    bible_text    = format_bible(bible)
    summary_text  = format_summaries(summaries)
    recent_text   = ""
    for n, text in recent_chapters:
        recent_text += f"\n\n── 第{n:03d}章（近期原文，仅供风格参考）──\n{text}\n"

    # Solution 1 — chapter type
    ch_type_name, ch_type_instr = get_chapter_type(next_chapter_n)

    # Solution 2 — anti-repetition
    avoid_note = extract_recent_techniques(recent_chapters)

    # Solution 4 — POV rotation every Nth chapter
    if next_chapter_n % POV_ROTATION_EVERY == 0:
        raw_chars = bible.get("characters", [])
        if isinstance(raw_chars, dict):
            char_list = []
            for role_key, val in raw_chars.items():
                if isinstance(val, dict):
                    if "name" in val:
                        char_list.append({"role": role_key, **val})
                    else:
                        for char_name, char_data in val.items():
                            entry = {"name": char_name, "role": role_key}
                            if isinstance(char_data, dict):
                                entry.update(char_data)
                            char_list.append(entry)
                elif isinstance(val, list):
                    for item in val:
                        if isinstance(item, dict):
                            char_list.append({"role": role_key, **item})
        else:
            char_list = raw_chars
        supporting = [c for c in char_list if isinstance(c, dict) and c.get("role") == "supporting"]
        pov_char = supporting[0]["name"] if supporting else None
        pov_note = (
            f"\n        【视角转换】本章从配角\u300c{pov_char}\u300d的视角叙述，"
            f"展示主角不知道的信息，制造戏剧性反差。主角可出现但不是叙事中心。"
        ) if pov_char else ""
    else:
        pov_note = ""

    # Hard cap style examples to avoid exceeding TPM limit.
    # Output reserve 4000 tokens; total input budget = 18,000 tokens.
    # Style examples get at most 3,000 chars (~1,800 tokens).
    style_budget_chars = 3000

    style_cat = bible.get("category", "西方奇幻")
    style_examples = load_nanpin_style_examples(
        current_chapter=next_chapter_n,
        max_chars=style_budget_chars,
        category=style_cat,
        style_dir=style_dir,
    )
    style_block = f"\n{style_examples}\n" if style_examples else ""

    return textwrap.dedent(f"""\
        ## 故事圣经
        {bible_text}

        ## 当前弧线计划
        {arc_plan}

        ## 历史章节摘要（完整记忆）
        {summary_text}
        {recent_text}
        {style_block}
        ## 任务：写第{next_chapter_n}章

        【本章类型】{ch_type_name} — {ch_type_instr}
        {avoid_note}{pov_note}

        本章要点：{next_chapter_hint}

        要求：
        - 字数：{CHAPTER_TARGET_CHARS}字以上
        - 第一句直接进入场景，无铺垫废话
        - 严格遵守故事圣经中的人物性格和世界规则
        - 本章结构、开篇方式、情绪基调须与上两章明显不同
        - 混合至少三种句式节奏（超短句 / 对话爆发 / 长句渲染）
        - 章节结尾制造强烈悬念或反转

        请直接输出章节正文（包含章节标题"第{next_chapter_n}章 XXXX"）。
    """)


def build_compress_prompt(chapter_n: int, chapter_text: str) -> str:
    return textwrap.dedent(f"""\
        以下是小说第{chapter_n}章的完整正文。

        请用{SUMMARY_MAX_CHARS}字以内概括本章内容，要求：
        1. 记录关键情节事件（发生了什么）
        2. 记录角色状态变化（谁的处境改变了）
        3. 记录任何新揭露的信息或伏笔
        4. 说明章节结尾的悬念点

        只输出摘要文字，不要加任何前缀。

        --- 正文 ---
        {chapter_text[:4000]}
    """)


def build_arc_plan_prompt(
    bible: dict,
    summaries: list[dict],
    arc_number: int,
    arc_start: int,
) -> str:
    arc_end = arc_start + ARC_LENGTH - 1
    bible_text   = format_bible(bible)
    summary_text = format_summaries(summaries)

    return textwrap.dedent(f"""\
        ## 故事圣经
        {bible_text}

        ## 已完成章节摘要
        {summary_text}

        ## 任务：制定第{arc_number}弧弧线计划（第{arc_start}-{arc_end}章）

        根据故事圣经中的整体走向和已完成的剧情，为接下来{ARC_LENGTH}章制定详细计划。

        输出格式：
        第{arc_number}弧：[弧线标题]（第{arc_start}-{arc_end}章）

        弧线目标：[这条弧线要完成什么，推进整体剧情到什么位置]

        关键节点：
        第{arc_start}章：[开篇事件]
        第{arc_start+5}章：[中期转折]
        第{arc_start+15}章：[高潮升级]
        第{arc_start+25}章：[弧线收尾，为下弧埋下伏笔]

        章节类型循环（自动按此顺序分配，可根据剧情微调）：
        战斗章→人物章→感情章→揭秘章→喘息章→转折章→（循环）
        每{POV_ROTATION_EVERY}章安排一次配角视角章（POV轮换）。

        逐章提示：
        （格式：第X章【类型】：一句话写作提示，共{ARC_LENGTH}行）
        第{arc_start}章【战斗章】：...
        第{arc_start+1}章【人物章】：...
        ...
        第{arc_end}章：...

        用简体中文输出。
    """)


def build_bible_update_prompt(
    bible: dict,
    recent_summaries: list[dict],
) -> str:
    bible_text   = format_bible(bible)
    summary_text = format_summaries(recent_summaries)

    return textwrap.dedent(f"""\
        ## 当前故事圣经
        {bible_text}

        ## 最近章节摘要
        {summary_text}

        ## 任务：更新故事圣经

        根据最近章节中发生的事件，更新故事圣经中需要修改的内容。

        以JSON格式输出完整的更新后故事圣经，结构与原始圣经相同。
        需要更新的字段包括（如有变化）：
        - 角色的 current_state、relationships、growth_arc.current_stage
        - romance_lines 中各恋人的 status（active/dormant/transitioned）
        - environments 中各环境的 status（current/completed/upcoming）
        - 新揭露的世界信息（key_rules、factions）
        不要改变 appearance、整体弧线设定、感情线基调，除非剧情已明确改变。

        直接输出JSON，不要加代码块标记。
    """)


def build_init_bible_prompt(novel_text: str) -> str:
    return textwrap.dedent(f"""\
        以下是一部中文网络小说的策划案和第一章内容。

        请从中提取信息，生成结构化的故事圣经（Story Bible），以JSON格式输出。

        JSON结构如下（严格遵守字段名）：
        {{
          "title": "书名",
          "category": "品类",
          "theme": "一句话主题",
          "world": {{
            "setting": "世界背景描述",
            "power_system": "力量/修炼/魔法体系说明",
            "key_rules": ["规则1", "规则2"],
            "factions": [
              {{"name": "势力名", "description": "简介"}}
            ],
            "geography": "地理概要"
          }},
          "characters": [
            {{
              "name": "角色名",
              "role": "protagonist/antagonist/supporting",
              "appearance": "外形描写：面容、身材、气质、标志性穿着（女性角色须突出美貌特色）",
              "personality": "性格特点",
              "growth_arc": {{
                "start": "开篇性格缺陷或局限",
                "end": "最终蜕变后的性格",
                "current_stage": "当前所处成长阶段名称",
                "milestones": ["阶段1触发事件", "阶段2触发事件"]
              }},
              "background": "背景经历",
              "current_state": "当前处境和状态",
              "abilities": "能力/技能",
              "relationships": {{"其他角色名": "关系描述"}}
            }}
          ],
          "romance_lines": [
            {{
              "name": "恋人姓名",
              "beauty_style": "美貌风格（清冷仙气/娇媚/英气飒爽/温婉等）",
              "appearance": "具体外形描写",
              "personality": "性格",
              "tone": "感情基调（青梅竹马/一见钟情/欢喜冤家等）",
              "entry_arc": "预计登场章节范围",
              "exit_or_transition": "退场或转化时机",
              "status": "active/dormant/transitioned"
            }}
          ],
          "environments": [
            {{
              "name": "环境名称",
              "description": "地点/世界层级简介",
              "challenge_type": "核心考验类型（生存/权谋/情感/信念等）",
              "growth": "主角在此获得的成长或领悟",
              "status": "current/completed/upcoming"
            }}
          ],
          "overall_arc": {{
            "beginning": "故事起点和初始冲突",
            "midpoint": "中段转折和升级",
            "climax": "高潮对决",
            "resolution": "结局走向"
          }},
          "total_planned_chapters": 200
        }}

        --- 策划案与第一章 ---
        {novel_text[:6000]}

        直接输出JSON，不要加任何前缀或代码块标记。
    """)


# ── API calls ─────────────────────────────────────────────────────────────────

def call(client: OpenAI, model: str, system: str, user: str, temp: float = 0.7) -> str:
    import time
    kwargs = dict(
        model    = model,
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    if not any(m in model for m in ("o1", "o3")):
        kwargs["temperature"] = temp
    for attempt in range(6):
        try:
            return client.chat.completions.create(**kwargs).choices[0].message.content
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = 60 * (attempt + 1)
                print(f"\n  [限速] 等待{wait}秒后重试（第{attempt+1}次）...", end=" ", flush=True)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("API 调用失败：超过最大重试次数")


# ── Core operations ───────────────────────────────────────────────────────────

def init_project(client: OpenAI, model: str, novel_path: Path, max_chapter: int = 9999) -> None:
    novel_text = novel_path.read_text(encoding="utf-8")

    # Extract chapter 1 text (after the === separator)
    sep = "=" * 40
    parts = novel_text.split(sep, 1)
    outline_text = parts[0]
    chapter1_text = parts[1].strip() if len(parts) > 1 else ""

    # Extract title from outline
    import re
    title_match = re.search(r"【书本名称】\s*(.+)", outline_text)
    title = title_match.group(1).strip() if title_match else novel_path.parent.name

    proj = project_dir(title)
    proj.mkdir(parents=True, exist_ok=True)
    (proj / "chapters").mkdir(exist_ok=True)

    print(f"初始化项目：{proj}")

    # Step 1: Generate story bible
    print("  正在生成故事圣经...", end=" ", flush=True)
    bible_raw = call(client, model,
        system="你是专业的小说策划，请从文本中提取信息生成结构化故事圣经。直接输出JSON。",
        user=build_init_bible_prompt(novel_text),
        temp=0.3,
    )
    # Clean JSON if wrapped in code blocks
    bible_raw = re.sub(r"^```json\s*|^```\s*|\s*```$", "", bible_raw.strip(), flags=re.MULTILINE)
    try:
        bible = json.loads(bible_raw)
    except json.JSONDecodeError:
        print("\n[WARN] 无法解析JSON，使用最小圣经结构。")
        bible = {"title": title, "category": "", "theme": "", "world": {}, "characters": [], "overall_arc": {}}
    save_json(proj / "bible.json", bible)
    print("完成")

    # Step 2: Generate arc plan for chapters 1-30
    print("  正在生成第1弧计划（第1-30章）...", end=" ", flush=True)
    arc_plan = call(client, model,
        system="你是专业的小说策划师，请制定详细的弧线计划。",
        user=build_arc_plan_prompt(bible, [], arc_number=1, arc_start=1),
        temp=0.5,
    )
    save_arc_plan(proj, arc_plan)
    print("完成")

    # Step 3: Save chapter 1
    # Step 3: Extract ALL existing chapters from the novel file
    import re as _re
    # Find all chapter blocks: "第N章 Title\n...text..."
    all_chapter_matches = list(_re.finditer(r"^第(\d+)章[ \t]+(.+)$", novel_text, _re.MULTILINE))
    # Filter to full chapter blocks (not brief outline entries): a full chapter has >500 chars of body
    existing_chapters: list[tuple[int, str, str]] = []  # (chapter_n, title, text)
    for i, m in enumerate(all_chapter_matches):
        ch_n = int(m.group(1))
        ch_title = m.group(2).strip()
        body_start = m.start()
        body_end = all_chapter_matches[i + 1].start() if i + 1 < len(all_chapter_matches) else len(novel_text)
        body = novel_text[body_start:body_end].strip()
        if len(body) > 500 and ch_n <= max_chapter and ch_n not in [c[0] for c in existing_chapters]:
            existing_chapters.append((ch_n, ch_title, body))

    if not existing_chapters and chapter1_text:
        existing_chapters = [(1, "", chapter1_text)]

    last_chapter_saved = 0
    for ch_n, ch_title, ch_text in existing_chapters:
        save_chapter(proj, ch_n, ch_text)
        print(f"  第{ch_n}章已保存（{len(ch_text)}字）。")
        print(f"  正在压缩第{ch_n}章摘要...", end=" ", flush=True)
        summary_raw = call(client, model,
            system="请简洁准确地概括章节内容。",
            user=build_compress_prompt(ch_n, ch_text),
            temp=0.3,
        )
        clean_title = ch_title.split("（")[0].strip()[:30]
        append_summary(proj, {"chapter": ch_n, "title": clean_title, "summary": summary_raw.strip()})
        print("完成")
        last_chapter_saved = max(last_chapter_saved, ch_n)

    if not existing_chapters:
        print("  [WARN] 未找到章节正文，请手动添加至 chapters/001.txt")

    # Step 5: Save initial state
    save_state(proj, {
        "current_chapter": last_chapter_saved,
        "arc_number": 1,
        "arc_start": 1,
    })

    print(f"\n项目初始化完成 → {proj}/")
    print(f"  下一步：python3 write_novel.py write --project \"{title}\"")


def get_next_chapter_hint(arc_plan: str, chapter_n: int, proj=None) -> str:
    """Extract the hint for chapter_n. Checks outline_91_200.txt first for ch 91-200."""
    import re
    if proj is not None and 91 <= chapter_n <= 200:
        outline_file = proj / "outline_91_200.txt"
        if outline_file.exists():
            outline = outline_file.read_text(encoding="utf-8")
            m = re.search(rf"第{chapter_n}章【[^】]*】[：:]?\s*(.+)", outline)
            if m:
                return m.group(1).strip()
    pattern = rf"第{chapter_n}章[：:【]\s*(.+)"
    m = re.search(pattern, arc_plan)
    if m:
        return m.group(1).strip()
    return f"根据弧线计划自然推进剧情，为第{chapter_n}章写出精彩内容。"


def write_chapter(client: OpenAI, model: str, proj: Path, style_dir: str = "Nanpin",
                  system_prompt: str = SYSTEM_PROMPT) -> int:
    """Write the next chapter. Returns the chapter number written."""
    state    = load_state(proj)
    bible    = load_json(proj / "bible.json")
    arc_plan = load_arc_plan(proj)
    summaries = load_summaries(proj)

    next_n = state["current_chapter"] + 1
    hint   = get_next_chapter_hint(arc_plan, next_n, proj=proj)

    # For ch 91-200, supplement arc_plan context with the detailed outline
    if 91 <= next_n <= 200:
        outline_file = proj / "outline_91_200.txt"
        if outline_file.exists():
            outline_text = outline_file.read_text(encoding="utf-8")
            arc_plan = outline_text[:6000]  # use outline as primary arc context

    # Load last RECENT_CHAPTERS_N full chapters
    recent = []
    for i in range(max(1, next_n - RECENT_CHAPTERS_N), next_n):
        text = load_chapter(proj, i)
        if text:
            recent.append((i, text))

    print(f"  写第{next_n}章  提示：{hint[:60]}...", end=" ", flush=True)

    chapter_text = call(client, model,
        system=system_prompt,
        user=build_chapter_prompt(bible, arc_plan, summaries, recent, next_n, hint, style_dir=style_dir),
        temp=0.9,
    )
    chapter_text = strip_technique_labels(chapter_text)
    save_chapter(proj, next_n, chapter_text)
    print("已写入", end=" ", flush=True)

    # Wait before summary call to avoid back-to-back TPM exhaustion
    time.sleep(65)

    # Compress to summary
    summary_text = call(client, model,
        system="请简洁准确地概括章节内容。",
        user=build_compress_prompt(next_n, chapter_text),
        temp=0.3,
    )
    import re
    title_m = re.search(r"第\s*\d+\s*章\s*(.+)", chapter_text[:200])
    ch_title = title_m.group(1).strip()[:30] if title_m else ""
    append_summary(proj, {"chapter": next_n, "title": ch_title, "summary": summary_text.strip()})
    print("→ 摘要已保存")

    # Update state
    state["current_chapter"] = next_n
    save_state(proj, state)

    # Check if bible update is needed
    if next_n % BIBLE_UPDATE_EVERY == 0:
        print(f"  [自动] 第{next_n}章：更新故事圣经...", end=" ", flush=True)
        update_bible(client, model, proj)
        print("完成")

    # Check if arc refresh is needed
    arc_start = state.get("arc_start", 1)
    if next_n >= arc_start + ARC_LENGTH - 1:
        new_arc_n     = state.get("arc_number", 1) + 1
        new_arc_start = arc_start + ARC_LENGTH
        outline_file  = proj / "outline_91_200.txt"
        if outline_file.exists() and new_arc_start <= 200:
            # outline_91_200.txt already covers this arc — skip AI re-generation
            print(f"  [大纲] 第{new_arc_n}弧（第{new_arc_start}章起）由 outline_91_200.txt 覆盖，跳过自动生成")
        else:
            print(f"  [自动] 弧线完成，生成第{new_arc_n}弧计划（第{new_arc_start}章起）...", end=" ", flush=True)
            refresh_arc(client, model, proj, new_arc_n, new_arc_start)
            print("完成")
        state["arc_number"] = new_arc_n
        state["arc_start"]  = new_arc_start
        save_state(proj, state)

    return next_n


def refresh_arc(client: OpenAI, model: str, proj: Path, arc_number: int, arc_start: int) -> None:
    bible     = load_json(proj / "bible.json")
    summaries = load_summaries(proj)
    arc_plan  = call(client, model,
        system="你是专业的小说策划师，请制定详细的弧线计划。",
        user=build_arc_plan_prompt(bible, summaries, arc_number, arc_start),
        temp=0.5,
    )
    save_arc_plan(proj, arc_plan)


def update_bible(client: OpenAI, model: str, proj: Path) -> None:
    bible    = load_json(proj / "bible.json")
    state    = load_state(proj)
    summaries = load_summaries(proj)
    # Pass only the most recent summaries for efficiency
    recent_summaries = summaries[-BIBLE_UPDATE_EVERY:]

    import re
    updated_raw = call(client, model,
        system="你是专业的小说策划师，请根据最新剧情更新故事圣经。直接输出JSON。",
        user=build_bible_update_prompt(bible, recent_summaries),
        temp=0.2,
    )
    updated_raw = re.sub(r"^```json\s*|^```\s*|\s*```$", "", updated_raw.strip(), flags=re.MULTILINE)
    try:
        updated = json.loads(updated_raw)
        save_json(proj / "bible.json", updated)
    except json.JSONDecodeError:
        print("\n  [WARN] 圣经更新JSON解析失败，保留原版。")


# ── Status display ────────────────────────────────────────────────────────────

def show_status(proj: Path) -> None:
    if not proj.exists():
        print(f"[ERROR] 项目目录不存在：{proj}")
        return

    state     = load_state(proj)
    summaries = load_summaries(proj)
    bible     = load_json(proj / "bible.json") if (proj / "bible.json").exists() else {}
    arc_plan  = load_arc_plan(proj)

    current   = state.get("current_chapter", 0)
    arc_n     = state.get("arc_number", 1)
    arc_start = state.get("arc_start", 1)

    print(f"\n{'='*50}")
    print(f"  项目：《{bible.get('title','?')}》  品类：{bible.get('category','?')}")
    print(f"{'='*50}")
    print(f"  已写章节：{current} 章")
    print(f"  当前弧线：第{arc_n}弧（从第{arc_start}章开始）")
    print(f"  弧线进度：{current - arc_start + 1}/{ARC_LENGTH} 章")
    print(f"  摘要数量：{len(summaries)} 条")

    chapters_dir = proj / "chapters"
    total_chars = sum(
        len(f.read_text(encoding="utf-8"))
        for f in chapters_dir.glob("*.txt")
    ) if chapters_dir.exists() else 0
    print(f"  总字数约：{total_chars:,} 字")

    # Token estimate
    summary_tokens = len(summaries) * 75
    print(f"\n  每次生成预估输入 token：")
    print(f"    故事圣经：    ~1,500")
    print(f"    弧线计划：    ~{len(arc_plan)//4:,}")
    print(f"    历史摘要：    ~{summary_tokens:,}  ({len(summaries)} 章 × 75)")
    print(f"    近期原文：    ~3,000  (最近{RECENT_CHAPTERS_N}章)")
    print(f"    合计：        ~{1500 + len(arc_plan)//4 + summary_tokens + 3000:,}")

    print(f"\n  下一章：第{current+1}章")
    hint = get_next_chapter_hint(arc_plan, current + 1)
    print(f"  写作提示：{hint[:80]}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def find_project(title_or_path: Optional[str]) -> Optional[Path]:
    """Resolve --project argument to a project directory."""
    if title_or_path is None:
        # Auto-detect: use the only project if there's exactly one
        if NOVELS_DIR.exists():
            projects = [d for d in NOVELS_DIR.iterdir() if d.is_dir()]
            if len(projects) == 1:
                return projects[0]
            if len(projects) > 1:
                print("[ERROR] 多个项目存在，请用 --project 指定：")
                for p in projects:
                    print(f"  {p.name}")
                sys.exit(1)
        print("[ERROR] 未找到项目，请先运行 init。")
        sys.exit(1)

    # Try as a path first, then as a title
    p = Path(title_or_path)
    if p.exists():
        return p
    p2 = project_dir(title_or_path)
    if p2.exists():
        return p2
    print(f"[ERROR] 找不到项目：{title_or_path}")
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Long-form novel writer with hierarchical memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("command", choices=["init", "write", "status", "new-arc", "update-bible"])
    parser.add_argument("--from", dest="from_file",
        help="[init] Path to generate_novel.py output file.")
    parser.add_argument("--max-chapter", type=int, default=9999,
        help="[init] Only import existing chapters up to this number (e.g. --max-chapter 2 keeps ch1-2, rewrites from ch3).")
    parser.add_argument("--project", "-p",
        help="Project title or path. Auto-detected if only one project exists.")
    parser.add_argument("--count", "-n", type=int, default=1,
        help="[write] Number of chapters to write in sequence (default: 1).")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL,
        help=f"OpenAI model (default: {DEFAULT_MODEL}).")
    parser.add_argument("--style-dir", default="Nanpin",
        choices=["Nanpin", "Nvpin"],
        help="[write] Novel library to sample style from: Nanpin (男频) or Nvpin (女频). Default: Nanpin.")
    parser.add_argument("--system-prompt", dest="system_prompt_file",
        help="[write] Path to a .txt file whose contents replace the default SYSTEM_PROMPT.")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] 未设置 OPENAI_API_KEY 环境变量。")
        print("        export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    if args.command == "init":
        if not args.from_file:
            print("[ERROR] init 命令需要 --from 参数。")
            print("  示例：python3 write_novel.py init --from generated/西方奇幻/novel.txt")
            sys.exit(1)
        novel_path = Path(args.from_file)
        if not novel_path.exists():
            print(f"[ERROR] 文件不存在：{novel_path}")
            sys.exit(1)
        init_project(client, args.model, novel_path, max_chapter=args.max_chapter)

    elif args.command == "write":
        proj = find_project(args.project)
        sys_prompt = SYSTEM_PROMPT
        if args.system_prompt_file:
            sp = Path(args.system_prompt_file)
            if not sp.exists():
                print(f"[ERROR] system-prompt 文件不存在：{sp}")
                sys.exit(1)
            sys_prompt = sp.read_text(encoding="utf-8")
            print(f"使用自定义 system prompt：{sp}")
        print(f"项目：{proj.name}")
        for i in range(args.count):
            print(f"[{i+1}/{args.count}]", end=" ")
            chapter_n = write_chapter(client, args.model, proj, style_dir=args.style_dir,
                                      system_prompt=sys_prompt)
            if i < args.count - 1:
                print("  [冷却] 等待120秒，避免限速...", flush=True)
                time.sleep(120)
        print(f"\n完成。共写至第{chapter_n}章。")
        print(f"文件目录：{proj}/chapters/")

    elif args.command == "status":
        proj = find_project(args.project)
        show_status(proj)

    elif args.command == "new-arc":
        proj  = find_project(args.project)
        state = load_state(proj)
        arc_n = state.get("arc_number", 1) + 1
        arc_start = state.get("arc_start", 1) + ARC_LENGTH
        print(f"生成第{arc_n}弧计划（第{arc_start}章起）...", end=" ", flush=True)
        refresh_arc(client, args.model, proj, arc_n, arc_start)
        state["arc_number"] = arc_n
        state["arc_start"]  = arc_start
        save_state(proj, state)
        print("完成")
        print(f"弧线计划已保存 → {proj}/arc_plan.txt")

    elif args.command == "update-bible":
        proj = find_project(args.project)
        print("更新故事圣经...", end=" ", flush=True)
        update_bible(client, args.model, proj)
        print("完成")
        print(f"故事圣经已更新 → {proj}/bible.json")


if __name__ == "__main__":
    main()
