#!/usr/bin/env python3
"""
fetch_engagement.py — 路径 C：抓曝光归一化的互动指标

为什么要它：番茄读者评分里约 28% 的方差由篇幅与曝光解释，拿它当质量效标不干净。
「在读人数」是纯曝光量。但有了**收藏数**就能构造

    收藏率 = 加书架人数 / 累计阅读人数

——"看到这本书的人里有多少愿意留下它"。这是**结构性**剥离曝光，
而不是像 residual_criterion.py 那样在统计上控制它。两条路互为印证。

数据来源：本地 TomatoNovelDownloader 的 /api/search，raw 字段含
all_bookshelf_count（收藏）、read_count（在读）、read_count_all（累计阅读）、
shelf_cnt_history（累计加书架）、keep_publish_days 等 70+ 项。

── 2026-09-18 重写。第一版跑出 190/838（22.7%），排查后发现根因有三：
   1. 服务中途挂掉，之后 600 多本全部记成「未命中」——把「查无此书」
      和「服务不可达」混为一谈，导致失败原因无法诊断，数据也无法信任。
   2. 只在全部跑完时写盘，服务一死全部结果丢失。
   3. 没有断点续跑，重跑要对平台重复请求 838 次。
   本版：健康检查 + 连续失败熔断 + 分类记录失败原因 + 增量存盘 + 续跑。

用法：
  ./TomatoNovelDownloader --server      # 必须先起服务
  python3 fetch_engagement.py --limit 20    # 小样本试
  python3 fetch_engagement.py               # 全量（自动跳过已抓到的）
  python3 fetch_engagement.py --retry-missing   # 只重试上次失败的
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = Path("review_loop")
HOST = "http://127.0.0.1:18423"
API = f"{HOST}/api/search"
OUT = BASE / "engagement.json"

# 熔断阈值。fetch() 内部已能等待服务从卡顿中恢复，所以走到这里的
# 「服务错误」都是等待 5 分钟后仍不可达的硬故障——阈值可以放宽，
# 只在真正没救时才中止。第一版设 12 太激进，把可恢复的卡顿当成了崩溃。
CIRCUIT_BREAK = 30
SAVE_EVERY = 40

DROP_PREFIX = ("thumb", "audio_thumb", "horiz_thumb", "expand_thumb", "detail_page_thumb")
DROP_EXACT = {"abstract", "book_abstract_v2", "copyright_info", "recommend_info",
              "read_count_show_strategy", "visibility_info", "extra",
              "original_author_infos", "category_v2_map", "sub_title_extra_list"}

_lock = threading.Lock()

# ── 全局请求节流 ──────────────────────────────────────────────────────────────
# 服务在持续请求下会卡住（三次跑批都遇到）。治本办法是放慢请求速率，
# 而不是等它卡完再自愈。
# 必须是**全局**间隔而非每线程 sleep：后者在并发 N 下实际间隔会变成 1/N。
_rate_lock = threading.Lock()
_last_req = [0.0]
REQUEST_INTERVAL = 0.0          # 秒，由 --delay 设置


def throttle() -> None:
    if REQUEST_INTERVAL <= 0:
        return
    with _rate_lock:
        wait = _last_req[0] + REQUEST_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_req[0] = time.monotonic()


def slim(raw: dict) -> dict:
    """只留标量，丢掉正文、URL、嵌套结构。"""
    out = {}
    for k, v in (raw or {}).items():
        if k in DROP_EXACT or k.startswith(DROP_PREFIX):
            continue
        if isinstance(v, bool) or isinstance(v, (int, float)):
            out[k] = v
        elif isinstance(v, str) and len(v) <= 120:
            out[k] = v
    return out


def server_alive() -> bool:
    try:
        with urllib.request.urlopen(f"{HOST}/api/status", timeout=8) as r:
            return r.status == 200
    except Exception:
        return False


def held_out_books() -> list[dict]:
    """从 37 个格子的结果文件反查留出书；书名与 book_id 取自 txt 头部
    （books.json 里的标题有残缺，搜不中）。"""
    out, seen = [], set()
    for p in sorted(BASE.glob("*/validation_pilot.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("reviewer_model") != "qwen3-max":
            continue
        for r in d["results"]:
            f = Path(d["source_dir"]) / d["category"] / r["file"]
            if not f.exists():
                continue
            head = f.read_text(encoding="utf-8", errors="replace")[:1500]
            bid = re.search(r"book_id=(\d+)", head)
            title = re.search(r"^书名[：:]\s*(.+)$", head, re.MULTILINE)
            if not (bid and title):
                continue
            key = f"{p.parent.name}|{r['file']}"
            if key in seen:
                continue
            seen.add(key)
            out.append({"key": key, "cell": p.parent.name, "file": r["file"],
                        "book_id": bid.group(1), "title": title.group(1).strip(),
                        "rating": r["real_rating"], "score": r["rubric_score"]})
    return out


def load_existing() -> tuple[dict, dict]:
    """(已抓到的, 上次失败的)。续跑靠它，避免对平台重复请求。"""
    if not OUT.exists():
        return {}, {}
    try:
        d = json.loads(OUT.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}, {}
    return d.get("books") or {}, d.get("failures") or {}


def wait_for_server(max_wait: int = 300) -> bool:
    """服务在持续请求下会短暂卡住（不是崩溃），随后自行恢复。
    第一版熔断把这种可恢复的卡顿当成永久故障，跑到 49% 就中止了。
    这里改为等它缓过来，最多等 max_wait 秒。"""
    waited = 0
    delay = 5
    while waited < max_wait:
        time.sleep(delay)
        waited += delay
        if server_alive():
            return True
        delay = min(delay * 2, 30)
    return False


def fetch(book: dict, retries: int = 3) -> tuple[str, dict | None]:
    """→ (状态, 数据)。状态为 ok / notfound / error。
    notfound = 服务答了但搜不到这本；error = 服务没答。两者必须分开，
    否则服务故障会被统计成「这些书不存在」。"""
    q = urllib.parse.quote(book["title"])
    last = ""
    for attempt in range(retries + 1):
        try:
            throttle()
            with urllib.request.urlopen(f"{API}?q={q}", timeout=30) as resp:
                items = json.load(resp).get("items", [])
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:60]}"
            if attempt < retries:
                time.sleep(2 ** attempt)      # 指数退避 1s → 2s → 4s
                continue
            # 重试用尽：先分清是服务卡住还是这次请求本身有问题
            if not server_alive():
                with _lock:
                    print("  [服务无响应] 暂停等待恢复…", flush=True)
                if wait_for_server():
                    with _lock:
                        print("  [服务已恢复] 继续", flush=True)
                    try:
                        throttle()
                        with urllib.request.urlopen(f"{API}?q={q}", timeout=30) as resp:
                            items = json.load(resp).get("items", [])
                    except Exception as e2:
                        return "error", {"reason": f"恢复后仍失败 {type(e2).__name__}"}
                else:
                    return "error", {"reason": f"服务持续不可达：{last}"}
            else:
                return "error", {"reason": last}

        m = next((i for i in items if str(i.get("book_id")) == book["book_id"]), None)
        if m:
            data = slim(m.get("raw") or {})
            if data.get("read_count_all"):
                return "ok", data
            return "notfound", {"reason": f"命中但缺 read_count_all（候选 {len(items)}）"}
        return "notfound", {"reason": f"搜到 {len(items)} 条但无 book_id 匹配"}
    return "error", {"reason": last or "unknown"}


def save(got: dict, failures: dict, total: int) -> None:
    OUT.write_text(json.dumps({
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n": len(got), "attempted": total,
        "n_failed": len(failures), "failures": failures, "books": got,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2,
                    help="并发数。服务在持续请求下会卡顿，2 比 3 稳")
    ap.add_argument("--retry-missing", action="store_true",
                    help="只重试上次失败的那些")
    ap.add_argument("--fresh", action="store_true", help="忽略已有结果，从头抓")
    ap.add_argument("--delay", type=float, default=0.8,
                    help="全局请求间隔（秒）。服务在持续请求下会卡顿，"
                         "放慢速率比事后自愈更治本")
    args = ap.parse_args()

    global REQUEST_INTERVAL
    REQUEST_INTERVAL = args.delay

    if not server_alive():
        raise SystemExit(
            "[ERROR] 下载器服务无响应。先启动：\n"
            "  cd ~/Downloads/Tomato-Novel-Downloader-main && ./TomatoNovelDownloader --server\n"
            "（第一版就是因为服务中途挂掉，把 648 本误记为『查无此书』）")

    books = held_out_books()
    got, prev_fail = ({}, {}) if args.fresh else load_existing()

    if args.retry_missing:
        todo = [b for b in books if b["key"] in prev_fail]
        print(f"重试模式：上次失败 {len(prev_fail)} 条，其中可定位 {len(todo)} 本")
    else:
        todo = [b for b in books if b["key"] not in got]
        if got:
            print(f"续跑：已有 {len(got)} 本，跳过；待抓 {len(todo)} 本")

    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        print("没有待抓的书。")
        return

    print(f"待抓 {len(todo)} 本，并发 {args.workers}，请求间隔 {args.delay}s\n")

    failures: dict = {} if args.fresh else dict(prev_fail)
    stats = {"ok": 0, "notfound": 0, "error": 0}
    consec_err = 0
    aborted = False
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(fetch, b): b for b in todo}
        done = 0
        for fut in as_completed(futs):
            b = futs[fut]
            done += 1
            try:
                status, data = fut.result()
            except Exception as e:
                status, data = "error", {"reason": f"{type(e).__name__}: {e}"}

            with _lock:
                stats[status] += 1
                if status == "ok":
                    got[b["key"]] = {**{k: b[k] for k in
                                        ("cell", "file", "book_id", "title", "rating", "score")},
                                     **data}
                    failures.pop(b["key"], None)
                    consec_err = 0
                else:
                    failures[b["key"]] = {"title": b["title"], "status": status,
                                          **(data or {})}
                    consec_err = consec_err + 1 if status == "error" else 0

                if done % SAVE_EVERY == 0:
                    save(got, failures, len(books))
                    print(f"  {done}/{len(todo)}  ok={stats['ok']} "
                          f"查无={stats['notfound']} 错误={stats['error']}"
                          f"  {(time.time() - t0) / 60:.1f} 分", flush=True)

                if consec_err >= CIRCUIT_BREAK:
                    aborted = True

            if aborted:
                break

    save(got, failures, len(books))

    print(f"\n{'=' * 62}")
    if aborted:
        print(f"⚠ 连续 {CIRCUIT_BREAK} 次服务错误，已中止——服务很可能又挂了。")
        print("  已抓到的都已存盘。重启服务后直接再跑一次即可续跑。")
    print(f"  本次：成功 {stats['ok']}　查无此书 {stats['notfound']}　服务错误 {stats['error']}")
    print(f"  累计已抓 {len(got)}/{len(books)}（{len(got) / max(1, len(books)) * 100:.1f}%）")
    if failures:
        by = {}
        for v in failures.values():
            by[v["status"]] = by.get(v["status"], 0) + 1
        print(f"  待解决 {len(failures)} 条：{by}")
        print("  重试：python3 fetch_engagement.py --retry-missing")
    print(f"  耗时 {(time.time() - t0) / 60:.1f} 分   已保存：{OUT}")


if __name__ == "__main__":
    main()
