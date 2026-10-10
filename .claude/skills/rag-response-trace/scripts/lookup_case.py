"""Tra một dòng câu hỏi demo theo ID và đối chiếu với tài liệu nguồn trong dataset.

Dùng:
    .venv/bin/python -I .claude/skills/rag-response-trace/scripts/lookup_case.py d0001
    ... --xlsx <file.xlsx> --dataset <thư mục dataset> --context 400

In ra: thông tin dòng (câu hỏi, đáp án kỳ vọng, evidence, doc ids...), thông tin
manifest của từng tài liệu kỳ vọng, và vị trí evidence trong PDF (trang + đoạn
xung quanh) để biết chunk nào lẽ ra phải được lấy ra.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import glob
import re
import sys
import unicodedata
from pathlib import Path

DEFAULT_XLSX_GLOB = "/home/huy/Main/questions/*.xlsx"
DEFAULT_DATASET = "/home/huy/Main/unisage-gateway/dataset"
SHEET_NAME = "Câu hỏi không trùng"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_id", nargs="?")
    parser.add_argument("--xlsx", default=None)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--sheet", default=SHEET_NAME)
    parser.add_argument("--context", type=int, default=500, help="số ký tự quanh evidence")
    parser.add_argument("--find", action="append", default=[],
                        help="cụm từ cần dò thêm trong PDF kỳ vọng (lặp lại được)")
    parser.add_argument("--doc", help="tra manifest theo một phần tên file/object key, rồi thoát")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    if args.doc:
        return _search_manifest(dataset, args.doc)
    if not args.case_id:
        parser.error("cần case_id hoặc --doc")

    xlsx = Path(args.xlsx) if args.xlsx else _find_xlsx()
    row = _find_row(xlsx, args.sheet, args.case_id.strip())
    if row is None:
        print(f"Không tìm thấy ID {args.case_id} trong sheet '{args.sheet}' của {xlsx}")
        return 1

    print(f"# Case {args.case_id}  ({xlsx.name} / {args.sheet})\n")
    for key, value in row.items():
        print(f"- **{key}**: {value if value not in (None, '') else '-'}")

    doc_ids = [d for d in re.split(r"[,;\s]+", str(row.get("Doc IDs") or "")) if d and d != "-"]
    if not doc_ids:
        print("\nKhông có Doc ID kỳ vọng (case không cần tài liệu: social/off_topic/unanswerable/web).")
        return 0

    official = _load_manifest(dataset / "official" / "manifest.csv")
    demo = _load_manifest(dataset / "demo" / "demo_manifest.csv")
    evidence = str(row.get("Evidence") or "")
    terms = _answer_terms(str(row.get("Đáp án kỳ vọng") or "")) + args.find

    for doc_id in doc_ids:
        print(f"\n## Tài liệu kỳ vọng {doc_id}")
        meta = official.get(doc_id)
        if meta is None:
            print("- Không có trong official/manifest.csv")
            continue
        demo_meta = demo.get(doc_id, {})
        for key in ("file_name", "unit", "department_id", "is_public", "access_level",
                    "quality", "pages", "text_chars", "local_path", "source_url"):
            print(f"- {key}: {meta.get(key, '-')}")
        if demo_meta:
            print(f"- demo_is_public: {demo_meta.get('demo_is_public')}  "
                  f"demo_access_level: {demo_meta.get('demo_access_level') or '-'}")
        if meta.get("quality") in {"scanned", "garbled_ocr", "broken_encoding", "no_diacritics"}:
            print(f"- ⚠ quality={meta['quality']}: ingest không OCR, text-layer có thể rỗng/hỏng "
                  "→ chunk có thể rỗng hoặc rác.")
        pdf = dataset / "official" / meta.get("local_path", "")
        print(f"- pdf: {pdf}")
        if pdf.is_file():
            _locate_evidence(pdf, evidence, args.context, terms)
        else:
            print(f"- Không thấy file PDF tại {pdf}")
    return 0


def _find_xlsx() -> Path:
    files = sorted(glob.glob(DEFAULT_XLSX_GLOB))
    if not files:
        sys.exit(f"Không thấy file xlsx theo {DEFAULT_XLSX_GLOB}; truyền --xlsx")
    return Path(files[0])


def _find_row(xlsx: Path, sheet: str, case_id: str) -> dict | None:
    import openpyxl

    workbook = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    # Tên sheet/file có thể lưu ở dạng Unicode tổ hợp (NFD) nên so sánh sau khi chuẩn hoá.
    target = _nfc(sheet)
    worksheet = next((ws for ws in workbook.worksheets if _nfc(ws.title) == target), None)
    if worksheet is None:
        sys.exit(f"Không có sheet '{sheet}'. Có: {[ws.title for ws in workbook.worksheets]}")
    rows = worksheet.iter_rows(values_only=True)
    header = [str(h).strip() if h is not None else "" for h in next(rows)]
    for values in rows:
        if values and str(values[0]).strip().lower() == case_id.lower():
            return dict(zip(header, values))
    return None


def _load_manifest(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {r["file_id"]: r for r in csv.DictReader(handle)}


def _locate_evidence(pdf: Path, evidence: str, context: int, terms: list[str]) -> None:
    import pymupdf

    with pymupdf.open(pdf) as doc:
        pages = [page.get_text() for page in doc]
    total = sum(len(p.strip()) for p in pages)
    print(f"- text-layer: {len(pages)} trang, {total} ký tự")
    if total == 0:
        print("- ⚠ PDF không có text-layer → ingest (pymupdf, không OCR) sẽ ra chunk rỗng.")
        return
    _report_terms(pages, terms)
    if not evidence.strip():
        return

    needle = _norm(evidence)
    best = (0.0, 0, 0, 0)
    for page_no, text in enumerate(pages, 1):
        hay = _norm(text)
        exact = hay.find(needle)
        if exact >= 0:
            best = (1.0, page_no, exact, exact + len(needle))
            break
        ratio, start, end = _fuzzy_window(needle, hay)
        if ratio > best[0]:
            best = (ratio, page_no, start, end)

    ratio, page_no, start, end = best
    if ratio < 0.5:
        print(f"- ⚠ Không tìm thấy evidence trong PDF (độ khớp tốt nhất {ratio:.2f}). "
              "Evidence có thể được LLM diễn đạt lại hoặc nằm trong ảnh/bảng.")
        return
    hay = _norm(pages[page_no - 1])
    snippet = hay[max(0, start - context // 2): end + context // 2]
    print(f"- Evidence nằm ở **trang {page_no}** (độ khớp {ratio:.2f}). Đoạn quanh evidence:")
    print("\n```text\n" + snippet + "\n```")


def _answer_terms(answer: str) -> list[str]:
    """Số liệu kèm đơn vị và cụm trong ngoặc kép của đáp án kỳ vọng — evidence
    thường chỉ phủ một ý, còn các ý khác cần biết nằm trang nào và có giá trị
    nào mâu thuẫn trong cùng tài liệu."""

    found = re.findall(r"\d[\d.,]*\s*(?:%|[^\W\d_]+)?", answer)
    found += re.findall(r"[\"“”]([^\"“”]{3,60})[\"“”]", answer)
    seen: list[str] = []
    for term in (t.strip() for t in found):
        if len(term) >= 2 and term.lower() not in (s.lower() for s in seen):
            seen.append(term)
    return seen


def _report_terms(pages: list[str], terms: list[str]) -> None:
    if not terms:
        return
    normalized = [_norm(p) for p in pages]
    print("- Vị trí các số liệu/cụm của đáp án kỳ vọng trong PDF:")
    for term in terms:
        needle = _norm(term)
        hits = [i for i, text in enumerate(normalized, 1) if needle in text]
        print(f"    - \"{term}\": " + (f"trang {', '.join(map(str, hits))}" if hits else "⚠ không thấy"))
    numbers = re.findall(r"\d+\s*(phút|giờ|ngày|tuần|tháng|năm|tín chỉ|%|đồng|triệu)", " ".join(terms), re.I)
    for unit in {u.lower() for u in numbers}:
        values = sorted({m for text in normalized for m in re.findall(rf"\b(\d+)\s*{re.escape(unit)}", text)})
        if len(values) > 1:
            print(f"    - ⚠ Tài liệu có nhiều giá trị \"… {unit}\": {', '.join(values)} — chunk sai đoạn sẽ cho số sai.")


def _search_manifest(dataset: Path, fragment: str) -> int:
    stem = re.sub(r"^[0-9a-f-]{8,}_", "", Path(fragment).name, flags=re.I)
    stem = _norm(re.sub(r"\.pdf$", "", stem, flags=re.I))
    rows = _load_manifest(dataset / "official" / "manifest.csv").values()
    hits = [r for r in rows if stem in _norm(r.get("file_name", "")) or stem in r.get("file_id", "")]
    if not hits:
        print(f"Không có tài liệu nào trong manifest khớp '{stem}' — có thể là file upload ngoài bộ dataset.")
        return 1
    for r in hits:
        print(f"- {r['file_id']}  {r['file_name']}  quality={r['quality']}  "
              f"pdf={dataset / 'official' / r['local_path']}")
    return 0


def _fuzzy_window(needle: str, hay: str) -> tuple[float, int, int]:
    if not hay:
        return 0.0, 0, 0
    width = len(needle)
    step = max(1, width // 4)
    best = (0.0, 0, 0)
    for start in range(0, max(1, len(hay) - width + 1), step):
        window = hay[start:start + width]
        ratio = difflib.SequenceMatcher(None, needle, window, autojunk=False).ratio()
        if ratio > best[0]:
            best = (ratio, start, start + width)
    return best


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", _nfc(text)).lower()


if __name__ == "__main__":
    raise SystemExit(main())
