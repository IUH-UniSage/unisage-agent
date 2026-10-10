"""Tóm tắt log unisage-agent của một lượt chat thành dạng dễ trace.

Dùng:
    .venv/bin/python -I .claude/skills/rag-response-trace/scripts/parse_log.py <log.txt>
        [--message-id <id>] [--expect <file_id|tên file>]...

Log là stderr của uvicorn với format
"%(asctime)s %(levelname)s %(name)s: %(message)s". Một lượt chat được nhận diện
bằng message_id (id tin nhắn ASSISTANT) trong các dòng `node=...`. Nếu log chứa
nhiều lượt và không truyền --message-id thì lấy lượt cuối cùng.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

LINE_START = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}[,.]?\d*)\s+(?P<level>[A-Z]+)\s+(?P<logger>[\w.]+):\s?(?P<msg>.*)$"
)
NODE = re.compile(
    r"node=(?P<node>\S+) model=(?P<model>\S+) conversation_id=(?P<conv>\S+) "
    r"message_id=(?P<mid>\S+) user_id=(?P<uid>\S+) ip=(?P<ip>\S+)"
)
PROMPT = re.compile(r"prompt node=(?P<node>\S+) conversation_id=(?P<conv>\S+) message_id=(?P<mid>[^:\s]+):")
RERANK_SQ = re.compile(r"LLM rerank SQ(?P<n>\d+) (?P<q>.*?): kept \[(?P<kept>.*)\]; dropped \[(?P<dropped>.*)\]$", re.S)
CONTEXT_CHUNK = re.compile(r"^\s{2}\[(?P<idx>\d+)\] \((?P<src>[^)]*)\) ?(?P<body>.*)$")
INTENT_HINTS = {
    "04_IntentRouting_SocialChat": "social_chat (template tĩnh, không RAG)",
    "05_OffTopicRejectNode": "off_topic (template tĩnh, không RAG)",
    "07_CalculationNode": "academic_calculation",
    "06_QueryTransformationNode": "academic_advisory (đi RAG)",
}


@dataclass
class Entry:
    logger: str
    msg: str


@dataclass
class Turn:
    message_id: str
    conversation_id: str = ""
    user_id: str = ""
    nodes: list[tuple[str, str]] = field(default_factory=list)
    prompts: dict[str, list[str]] = field(default_factory=dict)
    others: list[Entry] = field(default_factory=list)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", help="file log (hoặc - để đọc stdin)")
    parser.add_argument("--message-id")
    parser.add_argument("--expect", action="append", default=[],
                        help="file_id hoặc tên file tài liệu kỳ vọng (lặp lại được)")
    parser.add_argument("--chars", type=int, default=300, help="độ dài preview mỗi chunk")
    args = parser.parse_args()

    text = sys.stdin.read() if args.log == "-" else Path(args.log).read_text(encoding="utf-8", errors="replace")
    entries = _split_entries(text)
    turns = _group_turns(entries)
    if not turns:
        print("Không thấy dòng `node=... message_id=...` nào. Log có đúng của unisage-agent không?")
        return 1

    if args.message_id:
        turn = turns.get(args.message_id)
        if turn is None:
            print(f"Không có message_id {args.message_id}. Có: {list(turns)}")
            return 1
    else:
        turn = list(turns.values())[-1]
        if len(turns) > 1:
            print(f"(Log có {len(turns)} lượt: {list(turns)} — đang xem lượt cuối, dùng --message-id để chọn)\n")

    _report(turn, [_nfc(e).lower() for e in args.expect], args.chars)
    return 0


def _split_entries(text: str) -> list[Entry]:
    entries: list[Entry] = []
    for raw in text.splitlines():
        match = LINE_START.match(raw)
        if match:
            entries.append(Entry(match["logger"], match["msg"]))
        elif entries:
            entries[-1].msg += "\n" + raw
    return entries


def _group_turns(entries: list[Entry]) -> dict[str, Turn]:
    turns: dict[str, Turn] = {}
    current: Turn | None = None
    for entry in entries:
        node = NODE.search(entry.msg)
        prompt = PROMPT.search(entry.msg)
        if node:
            current = turns.setdefault(node["mid"], Turn(node["mid"]))
            current.conversation_id, current.user_id = node["conv"], node["uid"]
            current.nodes.append((node["node"], node["model"]))
        elif prompt:
            target = turns.setdefault(prompt["mid"], Turn(prompt["mid"]))
            body = entry.msg.split(":\n", 1)[1] if ":\n" in entry.msg else ""
            target.prompts.setdefault(prompt["node"], []).append(body)
        elif current is not None and not entry.logger.startswith("unisage.http"):
            current.others.append(entry)
    return turns


def _report(turn: Turn, expects: list[str], chars: int) -> None:
    print(f"# Lượt chat message_id={turn.message_id}")
    print(f"- conversation_id: {turn.conversation_id}")
    print(f"- user_id: {turn.user_id}" + ("  ⚠ guest — chỉ thấy tài liệu public!" if turn.user_id == "guest" else ""))

    print("\n## Chuỗi node")
    for name, model in turn.nodes:
        print(f"- {name}" + (f"  (model={model})" if model not in ("-", "None") else ""))
    names = [n for n, _ in turn.nodes]
    intents = [hint for node, hint in INTENT_HINTS.items() if node in names]
    print(f"\n→ Intent suy ra: {', '.join(intents) or 'không xác định (có thể greeting hoặc lỗi)'}")
    for flag, note in (("09a_LLMRerankNode", "LLM rerank đã chạy"),
                       ("09b_WebSearchNode", "Web search đã chạy (có sub-query không còn chunk nào)"),
                       ("11_TicketFallbackNode", "⚠ TicketFallback: KHÔNG còn chunk nào và không có web → trả lời mẫu từ chối")):
        if flag in names:
            print(f"- {note}")

    hyde = turn.prompts.get("06_QueryTransformationNode", [])
    print("\n## Văn bản dùng để embed (HyDE / sub-query)")
    if not hyde:
        print("- (không có — APP_DEBUG tắt trên server hoặc lượt không đi RAG)")
    for i, body in enumerate(hyde, 1):
        print(f"\n**SQ{i}**\n```text\n{body.strip()}\n```")

    print("\n## LLM rerank")
    rerank_lines = [e.msg for e in turn.others if e.logger.endswith("llm_rerank")]
    if not rerank_lines:
        print("- (không có dòng rerank)")
    for msg in rerank_lines:
        match = RERANK_SQ.match(msg)
        if not match:
            print(f"- {msg}")
            continue
        print(f"\n**SQ{match['n']}** {match['q']}")
        for label, items in (("kept", match["kept"]), ("dropped", match["dropped"])):
            parts = [p.strip() for p in items.split("; ") if p.strip()]
            print(f"- {label} ({len(parts)}):")
            for part in parts:
                print(f"    - {_mark(part, expects)}{part}")

    print("\n## Context đưa vào LLM sinh câu trả lời (10_GenerationSynthesisNode)")
    contexts = turn.prompts.get("10_GenerationSynthesisNode", [])
    if not contexts:
        print("- (không có — APP_DEBUG tắt hoặc không tới bước generation)")
    for body in contexts[-1:]:
        if "(không có tài liệu liên quan)" in body:
            print("- ⚠ academic_context RỖNG: '(không có tài liệu liên quan)'")
        chunks = _context_chunks(body)
        for idx, src, content in chunks:
            print(f"\n[{idx}] {_mark(src, expects)}{src}\n```text\n{content[:chars]}{'…' if len(content) > chars else ''}\n```")
        meta = body.split("## Văn Bản Quy Chế", 1)[0].strip()
        if meta:
            print(f"\n<details><summary>academic_metadata</summary>\n\n```text\n{meta}\n```\n</details>")
        if expects:
            hit = any(_mark(src, expects) for _, src, _ in chunks)
            print("\n→ Tài liệu kỳ vọng " + ("CÓ" if hit else "⚠ KHÔNG") + " nằm trong context gửi LLM.")

    web = turn.prompts.get("09b_WebSearchNode", [])
    if web:
        print("\n## Kết quả web search đưa vào prompt")
        for body in web:
            print(f"- {body.splitlines()[0] if body else ''}")

    warnings = [e for e in turn.others if not e.logger.endswith("llm_rerank")]
    if warnings:
        print("\n## Dòng log khác trong lượt")
        for e in warnings:
            print(f"- [{e.logger}] {e.msg.splitlines()[0][:300]}")


def _context_chunks(body: str) -> list[tuple[str, str, str]]:
    chunks: list[list[str]] = []
    inside = False
    for line in body.splitlines():
        if "<academic_context>" in line:
            inside = True
            continue
        if "</academic_context>" in line:
            break
        if not inside:
            continue
        match = CONTEXT_CHUNK.match(line)
        if match:
            chunks.append([match["idx"], match["src"], match["body"]])
        elif chunks:
            chunks[-1][2] += "\n" + line
    return [(i, s, c.strip()) for i, s, c in chunks]


def _mark(text: str, expects: list[str]) -> str:
    low = _nfc(text).lower()
    return "✅ " if any(e and (e in low or _stem(e) in low) for e in expects) else ""


def _stem(name: str) -> str:
    return re.sub(r"\.pdf$", "", name)


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


if __name__ == "__main__":
    raise SystemExit(main())
