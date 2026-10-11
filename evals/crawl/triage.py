"""Sort downloaded PDFs after `download`: drop broken ones, set scans aside.

- Broken (PyMuPDF can't open it, or 0 pages): file deleted, row moved from
  `manifest.csv` to `download_errors.csv` with reason `unreadable`, so a later
  `download` run won't fetch it again.
- Needs OCR: file moved from `files/<unit>/` to `scanned_pdf/<unit>/`, row
  kept in the manifest with the new `local_path` and the reason in `quality`:
  - `scanned`         - image only (< `SCANNED_CHARS_PER_PAGE` chars per page);
  - `garbled_ocr`     - a scanner's own OCR layer, e.g. "T6ng C6ng ty";
  - `no_diacritics`   - Vietnamese OCR'd without diacritics;
  - `broken_encoding` - font without a Unicode map, text is control chars.
  The last three have a text layer, so the production parser neither flags
  nor re-OCRs them - it would embed the garbage. OCR from the page image
  (Tesseract `vie`, built into the parser) is a separate decision.
- Everything else stays in `files/` with `quality=ok`.

Idempotent, and re-checks every non-scan row on each run: a file wrongly set aside
by an older heuristic moves back to `files/`.

Usage:
    python -m evals.crawl.triage --dataset ../unisage-gateway/dataset/official
"""

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from evals.crawl.download import DownloadState, is_probably_scanned

FILES_DIR = "files"
SCANNED_DIR = "scanned_pdf"


QUALITY_PAGES = 5
MIN_LETTERS = 200

VIETNAMESE_MARKED = frozenset("ăâđêôơưàảãáạằẳẵắặầẩẫấậèẻẽéẹềểễếệìỉĩíịòỏõóọồổỗốộờởỡớợùủũúụừửữứựỳỷỹýỵ")
# Common Vietnamese words as they come out of an OCR without diacritics.
BARE_VIETNAMESE_WORDS = frozenset(
    {"cua", "va", "cac", "duoc", "trong", "theo", "khong", "nhung", "nam", "hoc",
     "truong", "sinh", "vien", "dao", "tao", "viec", "thong", "quy", "dinh"}
)  # fmt: skip
_WORD = re.compile(r"\w+")
_DIGIT_INSIDE_WORD = re.compile(r"[^\W\d_]\d|\d[^\W\d_]")
_CODE_LIKE = re.compile(r"[A-Z0-9-]+")  # "PFCE155", "QD-DHCN": legitimately mixed


def text_quality(text: str) -> str:
    """'ok' | 'garbled_ocr' | 'no_diacritics' | 'broken_encoding' for extracted text.

    Thresholds were tuned on the 30-09-2026 crawl by reading samples of each
    group; they are heuristics, not guarantees.
    """

    if not text:
        return "ok"
    control = sum(1 for c in text if (ord(c) < 32 and c not in "\n\r\t") or 0x80 <= ord(c) <= 0x9F)
    if control / len(text) > 0.02:
        return "broken_encoding"
    letters = [c for c in text.lower() if c.isalpha()]
    if len(letters) < MIN_LETTERS:
        return "ok"
    words = _WORD.findall(text)
    marked = sum(1 for c in letters if c in VIETNAMESE_MARKED) / len(letters)
    # Real Vietnamese is dense with diacritics; a broken scanner OCR layer has
    # almost none. Requiring both keeps well-formed documents full of class and
    # student codes ("DHHO18B", "2026/ĐH-113") out of this bucket.
    mixed = sum(1 for w in words if _DIGIT_INSIDE_WORD.search(w) and not _CODE_LIKE.fullmatch(w))
    if mixed / len(words) > 0.03 and marked < 0.05:
        return "garbled_ocr"
    # All-caps words are skipped: name lists ("NGUYEN NGOC DINH") are legitimately
    # unaccented, while an OCR that dropped diacritics mangles running prose.
    bare = sum(1 for w in words if not w.isupper() and w.lower() in BARE_VIETNAMESE_WORDS)
    if marked < 0.02 and bare / len(words) > 0.03:
        return "no_diacritics"
    return "ok"


def pdf_text_quality(path: Path) -> str:
    try:
        document = pymupdf.open(path)
    except Exception:  # pages>0 means download could open it; keep rather than guess
        return "ok"
    try:
        pages = range(min(QUALITY_PAGES, document.page_count))
        text = " ".join(document[i].get_text() for i in pages)
    finally:
        document.close()
    return text_quality(text)


@dataclass
class TriageResult:
    removed_broken: int = 0
    kept_text: int = 0
    restored: int = 0
    moved: dict[str, int] = field(default_factory=dict)


def triage(dataset: Path) -> TriageResult:
    state = DownloadState(dataset)
    result = TriageResult()
    kept: list[dict[str, str]] = []
    for row in state.manifest:
        path = dataset / row["local_path"]
        pages = int(row["pages"] or 0)
        if pages == 0:
            path.unlink(missing_ok=True)
            state.errors.append(
                {
                    "source_url": row["source_url"],
                    "source_page": row["source_page"],
                    "unit": row["unit"],
                    "reason": "unreadable",
                }
            )
            result.removed_broken += 1
            continue
        relative = Path(row["local_path"])
        if is_probably_scanned(pages, int(row["text_chars"])):
            row["quality"] = "scanned"
        else:
            # Re-checked on every run, including files set aside earlier, so a
            # tightened heuristic moves false positives back into `files/`.
            row["quality"] = pdf_text_quality(path)
        wanted_dir = FILES_DIR if row["quality"] == "ok" else SCANNED_DIR
        if relative.parts[0] != wanted_dir:
            target = Path(wanted_dir, *relative.parts[1:])
            (dataset / target).parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                path.replace(dataset / target)
            row["local_path"] = target.as_posix()
            if wanted_dir == FILES_DIR:
                result.restored += 1
            else:
                result.moved[row["quality"]] = result.moved.get(row["quality"], 0) + 1
        if wanted_dir == FILES_DIR:
            result.kept_text += 1
        kept.append(row)
    state.manifest = kept
    state.save()
    for root in (FILES_DIR, SCANNED_DIR):
        for directory in (dataset / root).glob("*/"):
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    result = triage(args.dataset)
    print(
        f"removed broken: {result.removed_broken}  moved to {SCANNED_DIR}/: {result.moved}  "
        f"moved back to {FILES_DIR}/: {result.restored}  text PDFs in {FILES_DIR}/: "
        f"{result.kept_text}"
    )


if __name__ == "__main__":
    main()
