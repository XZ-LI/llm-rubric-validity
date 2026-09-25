#!/usr/bin/env python3
"""
contamination_probe.py — 评审模型是否"见过"这些书？

为什么要先做这个：
  如果模型的训练语料里包含番茄小说本身（甚至包含评分），那它打分时可能不是在
  评文本质量，而是在回忆标签。那会**虚假地抬高**相关系数——比零相关更危险，
  因为它看起来像好结果。中文模型（DeepSeek / 通义）尤其可能吃过中文网文。

怎么问：
  只给书名 + 作者，不给正文。问三件事：认不认识、主角是谁、简介一句话、
  番茄评分多少。然后和文件里的真实简介 / 评分对照。

关键对照组——假书名：
  混入几个不存在的书名。如果模型对假书也"认得"并编出简介，说明它对真书的
  "记忆"同样可能是编的，整个探针的阳性结果都不能采信。
  这一步不做，污染检查本身就没有效度。

用法：
  python3 contamination_probe.py --provider deepseek
  python3 contamination_probe.py --provider qwen --model qwen3-max
  python3 contamination_probe.py --provider openai --model o3
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from review_loop import clean_json, parse_novel_txt  # noqa: E402

PROVIDERS = {
    "openai":   ("https://api.openai.com/v1",                             "OPENAI_API_KEY",    "o3"),
    "deepseek": ("https://api.deepseek.com",                              "DEEPSEEK_API_KEY",  "deepseek-v4-pro"),
    "qwen":     ("https://dashscope.aliyuncs.com/compatible-mode/v1",     "DASHSCOPE_API_KEY", "qwen3-max"),
}

# 不存在的书名：仿网文命名习惯拼出来的，用作"模型会不会瞎编"的对照
FAKE_BOOKS = [
    ("重生九零：我靠养珍珠蚌暴富", "青梧不栖"),
    ("龙椅上的第三只狸奴", "墨鲤听雪"),
    ("穿成反派他嫂子的读心兔", "半盏孤灯照"),
]

QUESTION = """下面给你一部中文网络小说的书名和作者，**不提供正文**。

书名：《{title}》
作者：{author}

请如实回答，不确定就明确说不知道——编造会严重影响这项研究，宁可答"不知道"：

1. 你是否在训练数据里见过这部作品？
2. 主角叫什么名字？
3. 用一句话说明它的核心设定或剧情。
4. 它在番茄小说上的读者评分大概是多少（0-10）？

以JSON输出，不要代码块标记：
{{"seen": true/false, "protagonist": "名字或不知道", "premise": "一句话或不知道",
  "rating_guess": 数字或null, "confidence": "high"/"medium"/"low"}}"""


def ask(client, model: str, title: str, author: str) -> dict:
    kwargs = dict(model=model, messages=[
        {"role": "system", "content": "你在参与一项关于训练数据污染的研究。准确性至关重要：不知道就说不知道，绝对不要编造。"},
        {"role": "user", "content": QUESTION.format(title=title, author=author)},
    ])
    if not any(t in model for t in ("o1", "o3", "gpt-5")):
        kwargs["temperature"] = 0
    raw = client.chat.completions.create(**kwargs).choices[0].message.content
    try:
        return json.loads(clean_json(raw))
    except json.JSONDecodeError:
        return {"seen": None, "_raw": (raw or "")[:200]}


def pick_real_books(n: int) -> list[dict]:
    """从两个试水品类里按评分跨度均匀取样。"""
    books = []
    for cat, src in (("民国言情", "Nvpin"), ("动漫衍生", "Nanpin")):
        got = []
        for f in sorted((Path(src) / cat).glob("*.txt")):
            try:
                d = parse_novel_txt(f)
                d["_rating"] = float(d.get("rating") or 0)
            except Exception:
                continue
            if d["_rating"] > 0 and d.get("title"):
                d["_cat"] = cat
                got.append(d)
        got.sort(key=lambda d: -d["_rating"])
        step = max(1, len(got) // (n // 2))
        books += got[::step][: n // 2]
    return books


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="deepseek", choices=list(PROVIDERS))
    ap.add_argument("--model", default="")
    ap.add_argument("--n-real", type=int, default=6)
    args = ap.parse_args()

    base, env_key, default_model = PROVIDERS[args.provider]
    model = args.model or default_model
    key = os.environ.get(env_key)
    if not key:
        sys.exit(f"[ERROR] 未设置 {env_key}")

    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=base)

    real = pick_real_books(args.n_real)
    random.seed(7)
    items = [(b["title"], b.get("author", "") or "未知", b) for b in real] + \
            [(t, a, None) for t, a in FAKE_BOOKS]
    random.shuffle(items)

    print(f"\n{'=' * 70}\n  污染探针：{args.provider} / {model}")
    print(f"  真书 {len(real)} 部 + 假书 {len(FAKE_BOOKS)} 部（对照）\n{'=' * 70}")

    rows = []
    for title, author, book in items:
        kind = "真书" if book else "假书"
        print(f"\n  [{kind}] 《{title[:26]}》")
        try:
            r = ask(client, model, title, author)
        except Exception as e:
            print(f"    调用失败：{type(e).__name__} {str(e)[:120]}")
            continue
        seen, conf = r.get("seen"), r.get("confidence")
        print(f"    自称见过：{seen}   置信度：{conf}")
        print(f"    主角：{str(r.get('protagonist'))[:40]}")
        print(f"    简介：{str(r.get('premise'))[:90]}")
        if book:
            gt_desc = (book.get("description") or "").replace("\n", " ")[:90]
            print(f"    ── 真实简介：{gt_desc}")
            guess, truth = r.get("rating_guess"), book["_rating"]
            if isinstance(guess, (int, float)):
                print(f"    评分猜测 {guess} vs 真实 ★{truth}   误差 {abs(guess - truth):.1f}")
        rows.append({"kind": kind, "title": title, "seen": seen, "confidence": conf,
                     "rating_guess": r.get("rating_guess"),
                     "real_rating": book["_rating"] if book else None,
                     "premise": r.get("premise"), "protagonist": r.get("protagonist"),
                     "real_desc": (book.get("description") if book else None)})

    def rate(kind):
        s = [x for x in rows if x["kind"] == kind]
        yes = [x for x in s if x["seen"] is True]
        return len(yes), len(s)

    ry, rn = rate("真书")
    fy, fn = rate("假书")
    print(f"\n{'=' * 70}\n  判读\n{'=' * 70}")
    print(f"  真书自称见过：{ry}/{rn}")
    print(f"  假书自称见过：{fy}/{fn}   ← 这个数必须接近 0，否则整个探针无效")
    if fn and fy / fn >= 0.34:
        print("\n  ⚠ 模型对不存在的书也称'见过'并编出简介。")
        print("    它的'记忆'不可信，本探针无法判定污染——需改用时间切分法：")
        print("    用训练截止之后新上架的书做对照子集。")
    elif rn and ry / rn >= 0.5:
        print("\n  ⚠ 假书对照干净，但真书识别率高 → 存在污染风险。")
        print("    请人工核对上面的'简介 vs 真实简介'是否真的对得上；")
        print("    若对得上，评分实验必须补一个'训练截止后新书'子集。")
    else:
        print("\n  ✓ 未见明显污染迹象。可以进入交叉评审实验。")

    out = Path("review_loop") / f"contamination_{args.provider}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"provider": args.provider, "model": model, "rows": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  已保存：{out}")


if __name__ == "__main__":
    main()
