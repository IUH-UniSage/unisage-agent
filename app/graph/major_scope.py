"""Major (ngành) scope check between the question and the retrieved documents, after the
cohort and programme checks (app/graph/cohort_scope.py, app/graph/program_scope.py) and
before generation.

A lecturer without access to the KTPM thesis rules asked for the KTPM thesis weights; the only
document retrieved was the public thesis plan of another major (Quản trị du lịch, khách sạn,
nhà hàng), whose weights happen to be the same, and the answer gave them as KTPM's
(25/11/2026). Nothing told generation that the document is about another major.

The major is read from a document's title ("... ngành <X> khóa ...") and from the standalone
questions (a major named by a retrieved document, or a known abbreviation / full name such as
"KTPM" / "kỹ thuật phần mềm"). A document whose title names no major is neutral and never
dropped. Then:
- the asked major has a document: other majors' documents are dropped;
- it has none: generation is told to say first that there is no document for that major and
  which majors the documents found are for, then answer naming the major of every figure;
- the question names no major but documents do: generation names the major of every figure.
"""

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from app.graph.cohort_scope import ONLY_WHAT_WAS_ASKED
from app.rag.prompting.citations import source_title
from app.schemas.retrieval import RetrievedChunk

# Folded (lower case, no diacritics) abbreviation or short name -> folded full name. Only
# unambiguous ones:
# "KT" (kế toán / kinh tế / kỹ thuật) is left out on purpose.
_ABBREVIATIONS = {
    "ktpm": "ky thuat phan mem",
    "cntt": "cong nghe thong tin",
    "khmt": "khoa hoc may tinh",
    "httt": "he thong thong tin",
    "qtkd": "quan tri kinh doanh",
    "tcnh": "tai chinh ngan hang",
    "cntp": "cong nghe thuc pham",
    "qtdl": "quan tri du lich",
    "tmdt": "thuong mai dien tu",
    # Short names users type for a longer title.
    "co khi": "cong nghe ky thuat co khi",
}
# Full names recognised in a question even when no retrieved document is about them.
_DISPLAY = {
    "ky thuat phan mem": "Kỹ thuật phần mềm",
    "cong nghe thong tin": "Công nghệ thông tin",
    "khoa hoc may tinh": "Khoa học máy tính",
    "he thong thong tin": "Hệ thống thông tin",
    "quan tri kinh doanh": "Quản trị kinh doanh",
    "tai chinh ngan hang": "Tài chính - Ngân hàng",
    "cong nghe thuc pham": "Công nghệ thực phẩm",
    "quan tri du lich": "Quản trị du lịch",
    "thuong mai dien tu": "Thương mại điện tử",
    "cong nghe ky thuat co khi": "Công nghệ kỹ thuật cơ khí",
    "ke toan": "Kế toán",
}

# "ngành <X>" in a title, up to what follows a major there: cohort, year, programme, a
# bracket or a comma. Applied to the folded title.
_TITLE_MAJOR = re.compile(
    r"\bnganh\s+(.+?)(?=\s+(?:khoa\b|k\d{2}\b|nam\b|hk\d|hoc ky|dot\b|chat luong cao|tang cuong"
    r"|tien tien|chinh quy|dai hoc|cao dang|\d{4})|\s*[(),;:]|$)"
)


@dataclass(frozen=True)
class MajorScope:
    chunks: list[RetrievedChunk]
    # Per-turn instruction for generation ("" when none is needed).
    notice: str = ""
    asked: str | None = None
    available: tuple[str, ...] = ()
    dropped: int = 0


def fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    plain = "".join(char for char in decomposed if not unicodedata.combining(char)).lower()
    # A dash inside a name ("Tài chính - Ngân hàng") is a space, as users type it.
    return re.sub(r"\s+", " ", re.sub(r"\s*[-\u2013\u2014]\s*", " ", plain)).strip()


def document_major(source: str) -> tuple[str, str] | None:
    """(folded, display) major of a document's title, or None when it names none."""

    title = source_title(source)
    folded = fold(title)
    match = _TITLE_MAJOR.search(folded)
    if match is None:
        return None
    key = match.group(1).strip()
    if len(key) < 3:
        return None
    return key, _display(title, key) or key


def _display(title: str, key: str) -> str:
    """The words of the original title that fold to `key` (dash tokens skipped)."""

    words = title.split()
    folded = [fold(word).strip(",;:.()") for word in words]
    wanted = key.split()
    for start in range(len(words)):
        picked: list[str] = []
        index, matched = start, 0
        while index < len(words) and matched < len(wanted):
            if folded[index] == "":
                picked.append(words[index])
            elif folded[index] == wanted[matched]:
                picked.append(words[index])
                matched += 1
            else:
                break
            index += 1
        if matched == len(wanted):
            return " ".join(picked).strip(" ,;:.")
    return ""


def _same(asked: str, document: str) -> bool:
    """A question's major and a document's are the same when one name contains the other
    ("quan tri du lich" vs "quan tri du lich, khach san, nha hang" cut at the comma)."""

    return asked in document or document in asked


def asked_majors(question: str, available: Sequence[str]) -> set[str]:
    """Folded majors the question names: a retrieved document's major, a known full name, or
    a known abbreviation."""

    folded = fold(question)
    found = {major for major in available if re.search(rf"\b{re.escape(major)}\b", folded)}
    found |= {name for name in _DISPLAY if re.search(rf"\b{re.escape(name)}\b", folded)}
    found |= {name for abbr, name in _ABBREVIATIONS.items() if re.search(rf"\b{abbr}\b", folded)}
    # "quan tri du lich" and "quan tri du lich, khach san..." are one major.
    return {major for major in found if not any(m != major and major in m for m in found)}


def check_major_scope(
    questions: Sequence[str], chunks: Sequence[RetrievedChunk], user_message: str | None = None
) -> MajorScope:
    """`questions`: the turn's standalone questions; `user_message`: the user's own words,
    which decide when they name a major - a rewrite can add one (the standalone form of
    "diem khoa luan ktpm …" also named "Hệ thống thông tin", 25/11/2026). Questions naming
    different majors (a comparison) are left alone."""

    kept = list(chunks)
    majors = [document_major(chunk.source) for chunk in kept]
    display = dict(m for m in majors if m is not None)
    available = tuple(dict.fromkeys(key for key, _ in (m for m in majors if m is not None)))
    if not questions or not available:
        return MajorScope(chunks=kept)
    named = asked_majors(user_message, available) if user_message else set()
    if not named:
        for question in questions:
            named |= asked_majors(question, available)
    labels = tuple(display[key] for key in available)
    if len(named) > 1:
        return MajorScope(chunks=kept, asked=" | ".join(sorted(named)), available=labels)
    if not named:
        return MajorScope(chunks=kept, notice=_unnamed_notice(labels), available=labels)
    asked = named.pop()
    asked_label = _DISPLAY.get(asked) or next(
        (display[key] for key in available if _same(asked, key)), asked
    )
    if any(_same(asked, key) for key in available):
        matching = [
            chunk
            for chunk, major in zip(kept, majors, strict=True)
            if major is None or _same(asked, major[0])
        ]
        return MajorScope(
            chunks=matching,
            asked=asked_label,
            available=labels,
            dropped=len(kept) - len(matching),
        )
    return MajorScope(
        chunks=kept,
        notice=_missing_notice(asked_label, labels),
        asked=asked_label,
        available=labels,
    )


def _missing_notice(asked: str, available: Sequence[str]) -> str:
    found = ", ".join(f"ngành {name}" for name in available)
    return (
        "\n\n## PHẠM VI NGÀNH (bắt buộc)\n"
        f"Người dùng hỏi về ngành {asked} nhưng KHÔNG có tài liệu nào ghi ngành này; tài liệu "
        f"tìm được có ghi ngành thuộc: {found}.\n"
        f"- Câu đầu: nói rõ tài liệu tra cứu được chưa có văn bản cho ngành {asked}, chỉ có của "
        f"{found} (thiếu là ở tài liệu tra cứu được, không nói Nhà trường không có).\n"
        "- Sau đó trình bày thông tin được hỏi từ những tài liệu CÓ nó, mỗi số liệu ghi rõ ngành "
        f"của tài liệu nguồn; KHÔNG gọi đó là thông tin của ngành {asked}.\n"
        f"{ONLY_WHAT_WAS_ASKED}\n"
        f"- Cuối: gợi ý xác nhận với Khoa/Bộ môn quản lý ngành {asked}."
    )


def _unnamed_notice(available: Sequence[str]) -> str:
    found = ", ".join(f"ngành {name}" for name in available)
    return (
        "\n\n## PHẠM VI NGÀNH (bắt buộc)\n"
        f"Câu hỏi không nêu ngành; tài liệu tìm được có ghi ngành thuộc: {found}. Ghi rõ ngành "
        "của từng thông tin đưa ra, không trình bày như quy định chung của mọi ngành.\n"
        f"{ONLY_WHAT_WAS_ASKED}"
    )
