#!/usr/bin/env python3
"""
generate_novel.py — Generate webnovels with GPT modeled after each Tomato Novel category.

Modes:
  [default]   Title-only — uses example titles from books.json as style hints.
  --learn     Content-learning — reads all downloaded novels: all 30 descriptions +
              top-10 Chapter 1 openings + chapter endings + cross-category craft examples.
  --analyze   Analysis first — reverse-engineers the genre formula, saves a report,
              then generates. Uses --analyze-model for the analysis step.

Improvements applied:
  1. EXCERPT_CHARS = 3500  (fuller chapter openings, more style signal)
  2. Chapter 1 endings extracted (last 500 chars) to teach cliffhanger patterns
  3. Two-pass generation: outline pass (temp 0.5) → chapter pass (temp 0.9)
  4. Cross-category craft examples in system prompt (universal webnovel technique)
  5. Strict Chinese webnovel formatting rules in system prompt
  6. Separate --analyze-model flag (stronger model for analysis, faster for generation)

Usage:
  python3 generate_novel.py --category 西方奇幻 --analyze
  python3 generate_novel.py --category 西方奇幻 --learn
  python3 generate_novel.py --learn --skip-existing
  python3 generate_novel.py --list

Requirements:
  pip install openai
  export OPENAI_API_KEY=sk-...
"""

import argparse
import json
import os
import random
import re
import sys
import textwrap
from collections import Counter
from pathlib import Path
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] openai package not installed. Run: pip install openai")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────

BOOKS_FILE    = Path("books.json")
NOVELS_DIR    = Path("Nanpin")
OUTPUT_DIR    = Path("generated")
DEFAULT_MODEL = "o3"
DEFAULT_ANALYZE_MODEL = "o3"          # stronger reasoning for genre analysis

TOP_CHAPTER_EXAMPLES  = 5     # highest-rated novels to include chapter excerpts for
EXCERPT_CHARS         = 2000  # ① fuller opening: covers setup + first twist
ENDING_CHARS          = 300   # ② chapter ending: teaches cliffhanger pattern
CROSS_CATEGORY_COUNT  = 2     # ④ novels sampled from other categories for craft examples
CHAPTER_WORD_TARGET   = 2000  # target length for the generated Chapter 1
MAX_EXAMPLE_TITLES    = 8     # fallback: titles shown in title-only mode


# ── Category metadata ─────────────────────────────────────────────────────────

CATEGORY_META: dict[str, dict] = {
    "东方仙侠": {
        "en": "Eastern Cultivation (Xianxia)",
        "tropes": [
            "Protagonist starts weak, discovers hidden talent or rare physique",
            "Cultivation realms: Qi Condensation → Foundation → Core → Nascent Soul → etc.",
            "Sects, treasure hunts, spirit stones, and pills as resources",
            "Rivals, betrayal arcs, and revenge-driven growth",
            "Ancient ruins, inheritances, and legendary weapons",
        ],
        "hook": "the protagonist's first breakthrough or a sect trial gone wrong",
    },
    "传统玄幻": {
        "en": "Traditional Xuanhuan / High Fantasy",
        "tropes": [
            "Vast fantasy continent with kingdoms, empires, and monster territories",
            "Unique magic or power systems (elements, bloodlines, arrays)",
            "Protagonist rises from insignificance to world-shaking power",
            "Political intrigue among noble clans and royal families",
            "Ancient gods, demons, and catastrophic historical events",
        ],
        "hook": "the protagonist receiving or losing a powerful inheritance",
    },
    "动漫衍生": {
        "en": "Anime-Derived Fan Fiction",
        "tropes": [
            "Transmigration into a known anime or manga world",
            "Protagonist retains meta-knowledge of plot events",
            "Butterfly effects: changing canon storylines",
            "Power scaling adapted or enhanced relative to source material",
            "Breaking the 'villain dies' trope by befriending or reforming antagonists",
        ],
        "hook": "the moment the protagonist realizes which world they've entered",
    },
    "历史古代": {
        "en": "Historical / Ancient China",
        "tropes": [
            "Imperial court intrigue, eunuchs, and concubine politics",
            "Military strategy and war campaigns",
            "Scholar-official system and imperial examinations",
            "Underdog rising through merit or cunning in a rigid hierarchy",
            "Rich period-accurate customs, food, clothing, and architecture",
        ],
        "hook": "the protagonist arriving at court or surviving an assassination attempt",
    },
    "历史脑洞": {
        "en": "Historical 'What If' / Brain Holes",
        "tropes": [
            "Modern knowledge transplanted into a historical setting",
            "Protagonist introduces anachronistic inventions or strategies",
            "Changing the outcome of famous historical battles or events",
            "Humor and irony from the clash of modern sensibility with ancient norms",
            "Rapid economic or military advantage through foreknowledge",
        ],
        "hook": "the protagonist's first exploit using modern knowledge in the past",
    },
    "悬疑灵异": {
        "en": "Mystery / Supernatural Thriller",
        "tropes": [
            "Paranormal cases investigated by an unusual protagonist",
            "Hidden rules governing spirits, curses, and the spirit world",
            "Unreliable narrators and slow-burn reveals",
            "Folklore and mythological creatures adapted to modern settings",
            "Survival horror elements with escalating stakes",
        ],
        "hook": "an impossible crime scene or first encounter with the supernatural",
    },
    "悬疑脑洞": {
        "en": "Mystery Brain Holes",
        "tropes": [
            "Extreme puzzle-solving premises (locked rooms, logic traps)",
            "Reality-bending twists that recontextualize earlier scenes",
            "Unreliable timelines or perspectives",
            "High-stakes games or competitions with hidden rules",
            "Social engineering and psychological manipulation as plot tools",
        ],
        "hook": "the protagonist presented with an unsolvable puzzle",
    },
    "战神赘婿": {
        "en": "War God / Underestimated Son-in-Law",
        "tropes": [
            "Secretly powerful protagonist hiding identity as a lowly son-in-law",
            "Constant humiliation followed by dramatic reveals",
            "Powerful underworld or military background unveiled gradually",
            "Loyal subordinates arriving one by one to shock in-laws",
            "Face-slapping moments where the protagonist's true status is exposed",
        ],
        "hook": "the first public humiliation and a hint of the protagonist's real power",
    },
    "抗战谍战": {
        "en": "Anti-Japanese War / Spy Thriller",
        "tropes": [
            "1930s–1940s China, occupied territories and resistance fighters",
            "Double agents, encrypted messages, and deep-cover missions",
            "Moral ambiguity: loyalty to country vs. survival",
            "Period-accurate factions: Nationalist, Communist, Japanese, puppet governments",
            "Tense infiltration missions and narrow escapes",
        ],
        "hook": "the protagonist's cover being nearly blown on their first mission",
    },
    "游戏体育": {
        "en": "Gaming / Sports",
        "tropes": [
            "Protagonist becomes a legendary player in an esport or traditional sport",
            "Team dynamics, rivalries, and tournament arcs",
            "Tactical breakdowns of in-game or on-field decisions",
            "Underdog team rising through training montages and key matches",
            "System or cheat-code assists in some subgenres",
        ],
        "hook": "a clutch play that changes how others see the protagonist",
    },
    "玄幻脑洞": {
        "en": "Fantasy Brain Holes",
        "tropes": [
            "Subversive or comedic takes on classic cultivation/fantasy tropes",
            "Protagonist uses unconventional logic to solve problems others can't",
            "World-building with absurd but internally consistent rules",
            "Satire of common webnovel clichés",
            "Rapid escalation and sudden power-ups played for comedy or shock",
        ],
        "hook": "the protagonist's absurd solution to an impossible situation",
    },
    "男频衍生": {
        "en": "Male-Oriented Fan Fiction",
        "tropes": [
            "Transmigration or rebirth into a novel, game, or film world",
            "Protagonist uses plot armor awareness to avoid canon deaths",
            "Recruiting canon characters with foreknowledge of their potential",
            "Changing the villain's fate or siding with the 'losing' faction",
            "Power fantasy with harem or bromance elements",
        ],
        "hook": "the protagonist identifying their transmigration target world",
    },
    "科幻末世": {
        "en": "Sci-Fi / Post-Apocalypse",
        "tropes": [
            "Zombie outbreak, alien invasion, or environmental collapse",
            "Survival bases, resource scarcity, and faction warfare",
            "Ability awakening system (protagonist gains rare or OP power)",
            "Moral descent or preservation of humanity under pressure",
            "Technological remnants and discovering pre-apocalypse secrets",
        ],
        "hook": "the first day of the apocalypse and the protagonist's awakening",
    },
    "西方奇幻": {
        "en": "Western Fantasy",
        "tropes": [
            "European medieval setting with knights, mages, elves, and dragons",
            "Class-based magic or RPG-style leveling systems",
            "Dungeon exploration and monster-hunting guilds",
            "Prophecy, chosen one, or reincarnation into a weak noble",
            "Political conflict between church, crown, and arcane factions",
        ],
        "hook": "the protagonist's first dungeon run or noble court appearance",
    },
    "都市修真": {
        "en": "Urban Cultivation",
        "tropes": [
            "Ancient cultivator reborn or transported into modern city",
            "Hidden immortal sects operating within modern society",
            "Protagonist balancing mundane life (school, work) with cultivation",
            "Pill-refining, artifact-crafting adapted to modern resources",
            "Clash between ancient cultivation world and modern institutions",
        ],
        "hook": "the protagonist discovering cultivation in a modern context",
    },
    "都市日常": {
        "en": "Urban Slice of Life",
        "tropes": [
            "Calm, low-conflict protagonist with a unique skill (cooking, medicine, etc.)",
            "Heartwarming interactions with neighbors, coworkers, or strangers",
            "Subtle 'healing' narrative where helping others heals the protagonist too",
            "Gradual reputation-building without overt power fantasy",
            "Rich sensory descriptions of food, city life, and seasons",
        ],
        "hook": "the protagonist's first day in a new city or job, helping someone unexpectedly",
    },
    "都市种田": {
        "en": "Urban Farming / Building",
        "tropes": [
            "Protagonist develops a space, farm, or business from scratch",
            "Magical or systematic farming: spirit soil, miracle crops, space inventory",
            "Slow-burn wealth accumulation and community building",
            "Return-to-hometown tropes with family reconciliation",
            "Contrast between simple rural values and corrupt urban elites",
        ],
        "hook": "the protagonist discovering a spirit field or gaining a farming system",
    },
    "都市脑洞": {
        "en": "Urban Brain Holes",
        "tropes": [
            "Modern city setting with a wildly unconventional twist",
            "Protagonist solves problems with lateral thinking or absurd powers",
            "Social satire of corporate culture, internet trends, or consumer society",
            "Comedy arising from mundane situations taken to extremes",
            "Meta-awareness or fourth-wall adjacent humor",
        ],
        "hook": "the absurd premise revealed in the opening scene",
    },
    "都市高武": {
        "en": "Urban High Martial Arts",
        "tropes": [
            "Modern world where martial arts have evolved to superhuman levels",
            "Rankings, tournaments, and national or global martial competitions",
            "Protagonist holds a secret or suppressed level far above peers",
            "Mixing street-level brawls with high-stakes political power struggles",
            "Ancient martial legacies hidden within modern families or schools",
        ],
        "hook": "the protagonist's deliberately suppressed power accidentally revealed",
    },
}


# ── Novel .txt parser ─────────────────────────────────────────────────────────

def parse_novel_txt(path: Path) -> dict:
    """
    Parse a downloaded novel .txt file.

    Extracts:
      - Header fields: title, author, rating, tags, description
      - chapter1_title, chapter1_text  (first EXCERPT_CHARS chars)   ← improvement ①
      - chapter1_ending                (last ENDING_CHARS chars)      ← improvement ②
    """
    text = path.read_text(encoding="utf-8", errors="replace")

    sep = "=" * 20
    parts = text.split(sep, 1)
    header_raw = parts[0]
    body = parts[1] if len(parts) > 1 else ""

    def field(label: str) -> str:
        m = re.search(rf"^{label}[：:]\s*(.+)$", header_raw, re.MULTILINE)
        return m.group(1).strip() if m else ""

    desc_match = re.search(r"简介[：:]\s*\n([\s\S]*?)(?:\n={10,}|$)", header_raw)
    description = desc_match.group(1).strip() if desc_match else ""

    # Chapter 1 full text
    ch1_match = re.search(r"第\s*1\s*章\s*(.+?)\n([\s\S]*?)(?=\n第\s*2\s*章|\Z)", body)
    if ch1_match:
        ch1_title    = ch1_match.group(1).strip()
        ch1_full     = ch1_match.group(2).strip()
    else:
        ch1_any = re.search(r"第\s*\d+\s*章\s*(.+?)\n([\s\S]{200,})", body)
        ch1_title = ch1_any.group(1).strip() if ch1_any else ""
        ch1_full  = ch1_any.group(2).strip() if ch1_any else body

    ch1_text   = ch1_full[:EXCERPT_CHARS]
    ch1_ending = ch1_full[-ENDING_CHARS:] if len(ch1_full) > ENDING_CHARS else ""

    # Solution 5 — extract chapters 2-5 openings to learn chapter-type variation
    ch_sequence = []
    for ch_num in range(2, 6):
        pattern = rf"第\s*{ch_num}\s*章\s*(.+?)\n([\s\S]{{100,}}?)(?=\n第\s*{ch_num+1}\s*章|\Z)"
        m = re.search(pattern, body)
        if m:
            ch_sequence.append({
                "chapter": ch_num,
                "title": m.group(1).strip(),
                "opening": m.group(2).strip()[:600],
            })

    return {
        "title":           field("书名"),
        "author":          field("作者"),
        "rating":          field("评分"),
        "word_count":      field("字数"),
        "tags":            field("标签"),
        "description":     description,
        "chapter1_title":  ch1_title,
        "chapter1_text":   ch1_text,
        "chapter1_ending": ch1_ending,
        "ch_sequence":     ch_sequence,
    }


def load_content_examples(category: str) -> dict:
    """
    Parse all novels in Nanpin/<category>/, sorted by rating descending.

    Returns:
      all_descriptions  — every novel (description only, no chapter text)
      top_chapters      — top TOP_CHAPTER_EXAMPLES novels with opening + ending
      tag_frequency     — Counter of tags across all novels
    """
    cat_dir = NOVELS_DIR / category
    if not cat_dir.exists():
        return {"all_descriptions": [], "top_chapters": [], "tag_frequency": Counter()}

    parsed = []
    for f in cat_dir.glob("*.txt"):
        try:
            parsed.append(parse_novel_txt(f))
        except Exception:
            pass

    def rating_key(ex):
        try:
            return float(ex.get("rating") or 0)
        except ValueError:
            return 0.0

    parsed.sort(key=rating_key, reverse=True)

    tag_counter: Counter = Counter()
    for ex in parsed:
        for tag in re.split(r"[|｜,，、]", ex.get("tags") or ""):
            tag = tag.strip()
            if tag:
                tag_counter[tag] += 1

    return {
        "all_descriptions": parsed,
        "top_chapters":     parsed[:TOP_CHAPTER_EXAMPLES],
        "tag_frequency":    tag_counter,
    }


def load_cross_category_examples(exclude: str, seed: Optional[int]) -> list[dict]:
    """
    ④ Sample CROSS_CATEGORY_COUNT high-rated novels from categories other than
    `exclude`. Used to teach universal webnovel craft in the system prompt.
    """
    rng = random.Random(seed)
    other_cats = [c for c in CATEGORY_META if c != exclude]
    rng.shuffle(other_cats)

    examples = []
    for cat in other_cats:
        cat_dir = NOVELS_DIR / cat
        if not cat_dir.exists():
            continue
        files = sorted(cat_dir.glob("*.txt"))
        if not files:
            continue
        # Pick the first file (files aren't sorted by rating here, just grab one)
        try:
            ex = parse_novel_txt(rng.choice(files))
            ex["_category"] = cat
            examples.append(ex)
        except Exception:
            continue
        if len(examples) >= CROSS_CATEGORY_COUNT:
            break

    return examples


# ── Data loading ──────────────────────────────────────────────────────────────

def load_books() -> dict[str, list[dict]]:
    if not BOOKS_FILE.exists():
        return {}
    raw: list[dict] = json.loads(BOOKS_FILE.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict]] = {}
    for book in raw:
        cat = book.get("category") or "Unknown"
        grouped.setdefault(cat, []).append(book)
    return grouped


# ── Prompt building ───────────────────────────────────────────────────────────

def build_system_prompt(cross_examples: Optional[list[dict]] = None) -> str:
    """
    ④ ⑤ System prompt with:
       - Universal webnovel craft examples from other categories
       - Strict Chinese webnovel formatting rules
    """
    # ⑤ Formatting rules
    formatting_rules = textwrap.dedent("""\
        ## 中文网络小说写作规范（必须遵守）

        【段落】每段2-4句话，绝不写大段连续文字。重要时刻单独成行。
        【节奏】大量使用短句制造紧张感。情绪转折前后留白（空行）。
        【内心独白】用直接引语或括号：他心想：「这不可能。」
        【系统提示】格式：【叮咚：获得×××】或「叮！检测到……」
        【对话】每句对话单独成段，附带动作或表情描写。
        【章节结尾】必须以悬念、反转或紧迫感收尾，让读者无法停下来。
        【外形描写】角色首次登场时必须有外形描写：面容、身材、气质、穿着，用具体细节和比喻，让读者一眼留下印象。
                    女性角色的外形描写须突出其美貌，从主角视角感受其吸引力，不同女性须有各自鲜明的美丽风格，不可千篇一律。
                    男主角的外形描写须体现其独特气质（英俊/冷峻/邪魅/朴素等），与内心反差形成张力。
        【禁止】不写章节总结、不写元评论、不写"本章完"类字样。
        【禁止】绝对不得在正文中标注写作技巧名称，如"超短句""长句渲染""短景＋心理＋动作""单独成行"等——直接执行该技巧，不做任何说明或标注。
    """)

    # ④ Cross-category craft block
    craft_block = ""
    if cross_examples:
        craft_block = "## 跨品类参考——通用网文技巧示例\n"
        craft_block += "以下来自不同品类的高评分作品，展示中文网文的通用写作技巧：\n"
        craft_block += "钩子写法、节奏控制、章节收尾——这些技巧跨品类通用。\n\n"
        for ex in cross_examples:
            craft_block += f"  【{ex.get('_category','?')}】《{ex['title']}》 ★{ex.get('rating','?')}\n"
            if ex.get("description"):
                craft_block += f"  简介：{ex['description'][:200]}\n"
            if ex.get("chapter1_text"):
                craft_block += f"  开篇节选：{ex['chapter1_text'][:400]}\n"
            if ex.get("chapter1_ending"):
                craft_block += f"  章末节选：{ex['chapter1_ending'][:300]}\n"
            craft_block += "\n"

    return textwrap.dedent(f"""\
        你是一位专业的中文网络小说作家，擅长各类热门网文品类（网络小说）。
        你的文字节奏快、画面感强、深度契合读者口味。
        你清楚每个品类的读者期待，能写出让人欲罢不能的开篇、层层递进的张力和魅力十足的主角。

        {formatting_rules}
        {craft_block}
        所有输出一律使用简体中文。
    """)


def _format_tag_frequency(tag_counter: Counter, top_n: int = 20) -> str:
    if not tag_counter:
        return "  （无标签数据）"
    lines = []
    for tag, count in tag_counter.most_common(top_n):
        lines.append(f"  {count:2}次  {tag}  {'█' * count}")
    return "\n".join(lines)


def _format_descriptions(novels: list[dict]) -> str:
    if not novels:
        return "  （无）"
    blocks = []
    for i, ex in enumerate(novels, 1):
        line = f"  {i:2}. 《{ex['title']}》"
        if ex.get("rating"):
            line += f"  ★{ex['rating']}"
        if ex.get("tags"):
            line += f"  [{ex['tags']}]"
        if ex.get("description"):
            line += f"\n      {ex['description']}"
        blocks.append(line)
    return "\n".join(blocks)


def _format_chapter_examples(novels: list[dict]) -> str:
    """② Include opening, ending, and chapter 2-5 sequence to teach variation patterns."""
    if not novels:
        return "  （无）"
    blocks = []
    for i, ex in enumerate(novels, 1):
        block = f"  --- #{i} 《{ex['title']}》  ★{ex.get('rating','?')}  [{ex.get('tags','')}] ---\n"
        if ex.get("description"):
            block += f"\n  【简介】\n  {ex['description']}\n"
        if ex.get("chapter1_text"):
            label = ex.get("chapter1_title") or "第一章"
            block += f"\n  【{label} · 开篇节选】\n  {ex['chapter1_text']}\n"
        if ex.get("chapter1_ending"):
            block += f"\n  【{label} · 章末节选（注意收尾方式）】\n  {ex['chapter1_ending']}\n"
        # Solution 5 — show how chapters 2-5 vary in tone and structure
        seq = ex.get("ch_sequence", [])
        if seq:
            block += f"\n  【章节变化规律（第2-{seq[-1]['chapter']}章开篇，观察如何切换节奏与视角）】\n"
            for ch in seq:
                block += f"  ▸ 第{ch['chapter']}章《{ch['title']}》：{ch['opening'][:300]}\n"
        blocks.append(block)
    return "\n".join(blocks)


def build_outline_prompt(
    category: str,
    meta: dict,
    example_titles: list[str],
    content_pools: Optional[dict],
    genre_report: Optional[str],
) -> str:
    """③ Pass 1 prompt — generate structured outline only."""
    tropes_block = "\n".join(f"  • {t}" for t in meta["tropes"])

    ref_section = _build_ref_section(content_pools, example_titles)
    analysis_section = f"## 品类公式（来自分析报告）\n{genre_report}\n" if genre_report else ""

    return textwrap.dedent(f"""\
        ## 品类
        {category}（{meta['en']}）

        ## 品类惯例
        {tropes_block}

        {ref_section}
        {analysis_section}
        ## 任务：生成大纲

        为该品类创作一部原创网络小说的完整策划案。

        ===== 平台投稿信息 =====
        【书本名称】（10字以内，吸引眼球）
        【目标读者】男频 或 女频
        【作品标签】用"|"分隔，选3-5个（参考上方标签频率数据）
        【主角名1】（主角全名）
        【主角名2】（第一位恋人全名）
        【作品简介】50-500字，第一句直接抓住读者，不得出现低俗、暴力、血腥内容
        =======================

        1. **核心设定** — 一句话概括世界观与核心矛盾

        2. **主角档案**
           - 基本信息：姓名、出身、初始处境、隐藏实力或目标
           - **外形描写**：身高体型、面部特征（眼、眉、唇等细节）、气质风格、标志性穿着或配饰；
             描写须生动具体，可用比喻，让读者一眼记住
           - **性格成长弧（必填）**：
             * 起点性格：开篇时主角的核心缺陷或局限（如：自卑、冷漠、鲁莽、怯懦等）
             * 成长触发：哪些关键事件/人物迫使主角改变自己
             * 终点性格：百章后主角蜕变成什么样的人，与起点形成鲜明对比
             * 成长节奏：说明性格转变分几个阶段发生，每阶段对应的剧情里程碑

        3. **感情线设计（必须包含，不可省略）**
           主角须与至少三位恋人产生感情纠葛，每位恋人需有独特性格与登场时机。
           **核心原则：同一时间段内最多只有两位恋人同时在场，三条感情线须分布在不同剧情阶段，依次登场、依次退场或转化。**
           **女性角色外形要求：所有女性恋人均须是容貌出众的美人，外形描写须细致动人，突出各自不同的美丽风格（如：清冷仙气、娇媚妩媚、英气飒爽、温婉如玉等）。**
           每位恋人须包含以下信息：
           - 恋人A：姓名、【外形】面容/身材/气质/标志性穿搭（用生动比喻描写，强调美貌特色）、性格特点、登场阶段（第几章起）、退场或转化时机、感情基调（青梅竹马/一见钟情/欢喜冤家等）
           - 恋人B：姓名、【外形】面容/身材/气质/标志性穿搭（美貌风格须与恋人A形成反差）、性格特点、登场阶段（须晚于恋人A或在恋人A关系发生重大转折后）、退场或转化时机、感情基调
           - 恋人C：姓名、【外形】面容/身材/气质/标志性穿搭（第三种截然不同的美丽类型）、性格特点、登场阶段（须晚于恋人B或在恋人B关系发生重大转折后）、退场或转化时机、感情基调
           感情线推进节奏：说明三条线如何**错峰出现**——任意时间点最多两条线并行，避免三角以上的同时缠绕，制造"旧情未了、新情又起"的张力

        4. **大环境迁移与考验（必须包含，不可省略）**
           主角须经历至少三个截然不同的大环境，每个环境对应一段独立的考验与成长：
           - 环境一：地点/世界层级、核心考验类型（生存/权谋/情感/信念等）、主角在此获得的成长或领悟、离开该环境的转折事件
           - 环境二：地点/世界层级、核心考验类型（须与环境一不同维度）、主角在此获得的成长或领悟、离开该环境的转折事件
           - 环境三：地点/世界层级、核心考验类型（须与前两个不同维度）、主角在此获得的成长或领悟
           **要求**：每次环境切换须有情节驱动（不可凭空传送），且新环境的规则/威胁须让主角之前的经验部分失效，逼迫其以新方式成长

        5. **反派/核心矛盾** — 谁或什么阻挡主角，为什么难以对抗

        6. **章节大纲** — 第1-10章，每章：标题 + 两句内容概括 + 一句结尾钩子
           （感情线须在前10章内至少出现两次埋下伏笔；同一章内最多同时出现两位恋人，不得三线齐聚；
            大纲中须标注主角何时离开第一个大环境、踏入下一环境）

        开篇围绕：{meta['hook']}。
        至少颠覆该品类的一个常见套路。
        用简体中文输出。
    """)


def build_chapter_prompt(outline: str, category: str, meta: dict) -> str:
    """③ Pass 2 prompt — write Chapter 1 based on the approved outline."""
    return textwrap.dedent(f"""\
        以下是已确定的小说策划案：

        {outline}

        ## 任务：写第一章正文

        根据上方策划案，写出第一章完整正文。要求：

        - 字数：{CHAPTER_WORD_TARGET}字以上
        - 严格遵守中文网文写作规范（短段落、短句、内心独白、章末悬念）
        - 开篇第一句必须直接抓住读者，不写任何铺垫废话
        - 章节结尾必须制造强烈悬念或反转，让读者迫切想看下一章
        - 品类：{category}（{meta['en']}）

        直接输出章节正文，不要重复策划案内容。
        用简体中文输出。
    """)


def _build_ref_section(content_pools: Optional[dict], example_titles: list[str]) -> str:
    if content_pools and content_pools["all_descriptions"]:
        all_desc    = content_pools["all_descriptions"]
        top_ch      = content_pools["top_chapters"]
        tag_counter = content_pools.get("tag_frequency", Counter())
        return textwrap.dedent(f"""\
            ## 品类全览 — 全部{len(all_desc)}部作品（标题·评分·简介）
            研究简介写法：每位作者如何用2-4句话钩住读者。

            {_format_descriptions(all_desc)}

            ## 标签频率（来自全部{len(all_desc)}部作品）
            填写【作品标签】时从此处选取，选高频且符合你作品的标签。

            {_format_tag_frequency(tag_counter)}

            ## 精品参考 — 评分最高{len(top_ch)}部（简介 + 第一章开篇 + 章末）
            仔细研究开篇和章末：句子节奏、主角登场方式、张力构建、收尾悬念。
            吸收风格，写出原创内容。

            {_format_chapter_examples(top_ch)}
        """)
    else:
        titles_block = "\n".join(f"  - {t}" for t in example_titles)
        return textwrap.dedent(f"""\
            ## 参考书目
            {titles_block}
        """)


def build_analysis_prompt(category: str, meta: dict, content_pools: dict) -> str:
    all_desc = content_pools["all_descriptions"]
    top_ch   = content_pools["top_chapters"]

    return textwrap.dedent(f"""\
        你是专攻中文网络小说的文学分析师。

        以下是【{category}】品类全部{len(all_desc)}部作品的简介，
        以及评分最高{len(top_ch)}部作品的第一章节选（含章末）。
        请逆向拆解这个品类的成功公式——那些让这些小说让读者上瘾的可复制规律。

        ## 全部作品 — 简介
        {_format_descriptions(all_desc)}

        ## 标签频率
        {_format_tag_frequency(content_pools.get("tag_frequency", Counter()))}

        ## 评分最高{len(top_ch)}部 — 第一章节选（开篇+章末）
        {_format_chapter_examples(top_ch)}

        ## 分析任务

        请针对以下每项具体作答（引用上方原文中的书名和具体句子）：

        1. **钩子公式** — 开篇第一句如何抓住读者？前100字建立了什么处境或情绪？
           归纳出多部作品共有的规律。

        2. **主角蓝图** — 主角的初始状态是什么？拥有什么隐藏实力/系统/先知？
           表面弱势与真实潜力之间的核心张力是什么？

        3. **句式与节奏** — 长句短句比例？内心独白风格？
           第一场景推进速度？如何注入幽默或紧张感？

        4. **简介公式** — 全部{len(all_desc)}部作品的简介如何构建推销文案？
           归纳出通用模板（如：设定→反转→承诺回报）。
           哪些关键词和句式反复出现？

        5. **章末技巧** — 分析章末节选：用什么方式制造悬念？
           有哪些收尾套路（揭秘、威胁、反转、伏笔）？

        6. **品类标志** — 哪些设定、人物类型或情节节拍跨多部作品反复出现？
           与相邻品类的核心区别是什么？

        7. **避坑指南** — 哪些过度使用的套路应该在新作中颠覆或谨慎处理？

        最后写一段**一段话写作公式**，让作者按此可以写出该品类真实、有竞争力的作品。

        用简体中文输出。
    """)


# ── Generation ────────────────────────────────────────────────────────────────

def run_analysis(client: OpenAI, category: str, content_pools: dict, model: str) -> str:
    meta   = CATEGORY_META[category]
    prompt = build_analysis_prompt(category, meta, content_pools)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是精准的文学分析师，给出具体分析，避免泛泛而谈。"},
            {"role": "user",   "content": prompt},
        ],
        **({"temperature": 0.3} if "o1" not in model and "o3" not in model else {}),
    )
    return response.choices[0].message.content


def generate_for_category(
    client: OpenAI,
    category: str,
    books_by_category: dict[str, list[dict]],
    model: str,
    analyze_model: str,
    outline_only: bool,
    use_content: bool = False,
    use_analysis: bool = False,
    seed: Optional[int] = None,
) -> str:
    meta = CATEGORY_META.get(category)
    if meta is None:
        raise ValueError(f"未知品类 '{category}'，运行 --list 查看支持的品类。")

    # ④ Cross-category craft examples
    cross_examples = load_cross_category_examples(category, seed) if use_content else None

    # Content pools
    content_pools: Optional[dict] = None
    if use_content or use_analysis:
        content_pools = load_content_examples(category)
        n_desc = len(content_pools["all_descriptions"])
        n_ch   = len(content_pools["top_chapters"])
        if n_desc == 0:
            print(f"\n  [warn] Nanpin/{category}/ 无小说文件，退回标题模式。")
            content_pools = None
        else:
            print(f"\n  已加载 {n_desc} 部简介 + {n_ch} 部章节节选", end=" ", flush=True)

    # Genre analysis report
    genre_report: Optional[str] = None
    report_path = OUTPUT_DIR / category / "genre_report.txt"

    if use_analysis:
        if content_pools:
            print(f"— 正在分析（{analyze_model}）...", end=" ", flush=True)
            genre_report = run_analysis(client, category, content_pools, analyze_model)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(genre_report, encoding="utf-8")
            print(f"报告已保存 → {report_path}")
        else:
            print("  [warn] 无内容可分析。")
    elif report_path.exists():
        genre_report = report_path.read_text(encoding="utf-8")

    # Fallback to title list
    example_titles: list[str] = []
    if not content_pools:
        available = [b["title"] for b in books_by_category.get(category, []) if b.get("title")]
        example_titles = random.Random(seed).sample(available, min(MAX_EXAMPLE_TITLES, len(available)))

    # ⑤ System prompt with formatting rules + ④ cross-category examples
    system = build_system_prompt(cross_examples)

    # ③ Two-pass generation
    outline_prompt = build_outline_prompt(
        category      = category,
        meta          = meta,
        example_titles= example_titles,
        content_pools = content_pools,
        genre_report  = genre_report,
    )

    outline_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": outline_prompt},
        ],
        temperature=0.5,  # ③ lower temp for coherent structure
    )
    outline = outline_response.choices[0].message.content

    if outline_only:
        return outline

    # Pass 2: write Chapter 1 from the approved outline
    chapter_prompt = build_chapter_prompt(outline, category, meta)
    chapter_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",    "content": system},
            {"role": "user",      "content": outline_prompt},
            {"role": "assistant", "content": outline},
            {"role": "user",      "content": chapter_prompt},
        ],
        temperature=0.9,  # ③ higher temp for vivid prose
    )
    chapter = chapter_response.choices[0].message.content

    return outline + "\n\n" + "=" * 40 + "\n\n" + chapter


def save_output(category: str, content: str) -> Path:
    out_dir  = OUTPUT_DIR / category
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "novel.txt"
    out_file.write_text(content, encoding="utf-8")
    return out_file


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate webnovels with GPT for each Tomato Novel category",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--category", "-c",
        help="单个品类（如 西方奇幻），不填则处理全部品类。")
    parser.add_argument("--list", "-l", action="store_true",
        help="列出所有支持的品类及已下载数量后退出。")
    parser.add_argument("--learn", action="store_true",
        help="读取已下载小说内容作为风格参考。")
    parser.add_argument("--analyze", action="store_true",
        help="先分析品类公式再生成（隐含 --learn）。")
    parser.add_argument("--outline-only", action="store_true",
        help="只输出大纲，不写第一章正文。")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL,
        help=f"生成用模型（默认：{DEFAULT_MODEL}）。")
    parser.add_argument("--analyze-model", default=DEFAULT_ANALYZE_MODEL,
        help=f"分析用模型（默认：{DEFAULT_ANALYZE_MODEL}）。")  # ⑥
    parser.add_argument("--seed", type=int, default=None,
        help="随机种子，用于复现相同的采样结果。")
    parser.add_argument("--skip-existing", action="store_true",
        help="跳过已存在 novel.txt 的品类。")
    args = parser.parse_args()

    if args.list:
        print(f"{'品类':<12}  {'英文名':<38}  已下载")
        print("-" * 65)
        for cat, meta in CATEGORY_META.items():
            count = len(list((NOVELS_DIR / cat).glob("*.txt"))) if (NOVELS_DIR / cat).exists() else 0
            print(f"{cat:<12}  {meta['en']:<38}  {count} 部")
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] 未设置 OPENAI_API_KEY 环境变量。")
        print("        export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    client            = OpenAI(api_key=api_key)
    books_by_category = load_books()
    use_content       = args.learn or args.analyze
    use_analysis      = args.analyze

    categories = [args.category] if args.category else list(CATEGORY_META.keys())

    if args.category and args.category not in CATEGORY_META:
        print(f"[ERROR] 未知品类 '{args.category}'，运行 --list 查看支持的品类。")
        sys.exit(1)

    total = len(categories)
    for i, category in enumerate(categories, 1):
        out_file = OUTPUT_DIR / category / "novel.txt"

        if args.skip_existing and out_file.exists():
            print(f"[{i:2}/{total}] {category} — 已跳过（novel.txt 存在）")
            continue

        meta = CATEGORY_META[category]
        print(f"[{i:2}/{total}] {category} ({meta['en']}) ...", end=" ", flush=True)

        try:
            content = generate_for_category(
                client            = client,
                category          = category,
                books_by_category = books_by_category,
                model             = args.model,
                analyze_model     = args.analyze_model,
                outline_only      = args.outline_only,
                use_content       = use_content,
                use_analysis      = use_analysis,
                seed              = args.seed,
            )
            saved = save_output(category, content)
            print(f"已保存 → {saved}")
        except Exception as e:
            print(f"\n  [ERROR] {e}")
            if total == 1:
                sys.exit(1)
            err_dir = OUTPUT_DIR / category
            err_dir.mkdir(parents=True, exist_ok=True)
            (err_dir / "error.txt").write_text(str(e), encoding="utf-8")

    print(f"\n完成。输出目录：{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
