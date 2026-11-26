"""Course row lookup: the row of the course a question names, read from the whole table of
the programme documents (chương trình đào tạo, kế hoạch đào tạo) already found.

A programme document is one wide table cut into ~20 chunks of rows. The search and the
cross-encoder rank those chunks by how the question reads against them, and a programme
whose title shares the question's words wins over the row asked for: "Tiếng Anh 2 ... cơ khí
khóa 22" found the "tăng cường tiếng Anh" programmes, and the standard programme's chunk
holding the Tiếng Anh 2 row came 28th in the search and 50th after the cross-encoder, which
keeps 8 (25/11/2026). Sending whole programmes to generation costs 13 000 - 50 000 tokens a
question; this reads their chunks in code instead (no LLM) and adds only the chunks holding
the asked course's row - and the rows of the courses it names as prerequisites (`015253(b)`).

The course is the classifier's `hoc_phan` (named as the documents name it), or a course
name of those tables written in the user's message or the standalone questions (accents
ignored). Nothing matches → nothing changes. Only documents the caller can see are read: the
programme documents come from this turn's candidates and are fetched with the caller's filter.
"""

import logging
import re
from collections.abc import Awaitable, Callable, Iterator, Sequence
from dataclasses import dataclass, field

from app.graph.cohort_scope import asked_cohort, document_cohort
from app.graph.major_scope import asked_majors, document_major, fold
from app.graph.program_scope import STANDARD, document_program
from app.rag.prompting.citations import source_title
from app.schemas.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

# Programme documents read at most (first ones among the candidates), chunks fetched per
# document at most, and chunks added at most.
MAX_DOCUMENTS = 6
MAX_DOCUMENT_CHUNKS = 60
MAX_ADDED = 4
COURSE_ROW_FLAG = "course_row"

_PROGRAMME_TITLE = re.compile(r"\b(chuong trinh dao tao|ke hoach dao tao|chuong trinh khung)\b")
# A course code cell: "015254" (mã môn học), "4203015254" (mã học phần), or the 7 digits
# of the older plans ("2101700", Kế hoạch đào tạo ngành Công nghệ thông tin khóa 16).
_CODE = re.compile(r"^\d{6,7}$|^\d{10}$")
# A prerequisite cell: "015253(b)", "003067(a)", several in one cell.
_USER_CODE = re.compile(r"(?<!\d)(\d{6,7}|\d{10})(?!\d)")

FetchDocuments = Callable[[list[str]], Awaitable[list[RetrievedChunk]]]
# document_id -> object key of every document (RetrievalService.document_titles_async).
Titles = Callable[[], Awaitable[dict[str, str]]]


@dataclass
class CourseLookup:
    added: list[RetrievedChunk] = field(default_factory=list)
    # Folded names (or 6-digit codes) of the courses asked about.
    courses: list[str] = field(default_factory=list)
    documents: int = 0
    # document_id -> {6-digit code: course name}, of every programme document read: names
    # the prerequisites of the summary lines.
    names: dict[str, dict[str, str]] = field(default_factory=dict)
    # The asked course as the first table naming it writes it ("Cơ lưu chất"), when exactly
    # one course was asked by name: fills `hoc_phan` when the classifier left it out.
    course_name: str | None = None


def document_id(chunk: RetrievedChunk) -> str:
    return str(chunk.metadata.get("document_id") or chunk.chunk_id.split(":", 1)[0])


def is_programme_title(source: str) -> bool:
    return bool(_PROGRAMME_TITLE.search(fold(source_title(source) if source else "")))


def is_programme_document(chunk: RetrievedChunk) -> bool:
    return is_programme_title(chunk.source)


def sibling_documents(
    candidates: Sequence[RetrievedChunk], titles: dict[str, str], texts: Sequence[str]
) -> list[str]:
    """Programme documents of the asked major for the asked cohort - or, when it has none,
    the nearest earlier cohort that has some - standard programme first. The search may
    have found only the other programmes of that major ("tăng cường tiếng Anh" for a
    "Tiếng Anh 2" question): the standard one is read too. Nothing when the turn names
    no cohort (every cohort's programme would qualify)."""

    found = {
        major[0]
        for chunk in candidates
        if is_programme_document(chunk) and (major := document_major(chunk.source))
    }
    if not found:
        return []
    # The majors the turn names (a found one, a known full name or abbreviation - "cntt");
    # none named → the found ones. A named major is looked up in the catalogue even when no
    # candidate is for it ("còn ngành cntt" after a mechanical-engineering question), never
    # replaced by the found ones: their programmes would not answer it.
    asked = asked_majors(" ".join(texts), tuple(found)) or found
    cohort = next((year for text in texts if (year := asked_cohort(text))), None)
    if cohort is None:
        return []
    pool = [
        (doc, source)
        for doc, source in titles.items()
        if is_programme_title(source)
        and (major := document_major(source))
        and any(name in major[0] or major[0] in name for name in asked)
    ]
    years = {year for _, source in pool if (year := document_cohort(source))}
    target = cohort if cohort in years else max((y for y in years if y < cohort), default=None)
    picked = [(doc, src) for doc, src in pool if target and document_cohort(src) == target]
    picked.sort(key=lambda item: (document_program(item[1]) != STANDARD, item[1]))
    return [doc for doc, _ in picked]


def programme_documents(chunks: Sequence[RetrievedChunk]) -> list[str]:
    """Ids of the programme documents among `chunks`, first seen first, at most
    MAX_DOCUMENTS."""

    found: list[str] = []
    for chunk in chunks:
        doc = document_id(chunk)
        if doc not in found and is_programme_document(chunk):
            found.append(doc)
    return found[:MAX_DOCUMENTS]


def table_rows(content: str) -> Iterator[list[str]]:
    """Cells of each table line. A merged cell keeps the parser's "<br>" ("2<br>015311",
    "4203015311<br>"): it becomes a space, and a cell ending in a code is read as that code."""

    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("|"):
            cells = [
                " ".join(cell.replace("<br>", " ").split()) for cell in line.strip("|").split("|")
            ]
            yield [_trailing_code(cell) for cell in cells]


def _trailing_code(cell: str) -> str:
    words = cell.split()
    return words[-1] if len(words) > 1 and _CODE.match(words[-1]) and words[0].isdigit() else cell


_CONDITION_KINDS = {"a": "học trước", "b": "tiên quyết", "c": "song hành"}
_CREDITS = re.compile(r"^\d+\s*\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)$")
_CONDITION = re.compile(r"(\d{6,7})\s*\(\s*([abc])\s*\)")


@dataclass(frozen=True)
class CourseRow:
    code: str  # the 6-digit course code (mã môn học)
    name: str
    # Every other cell of the row, for the summary line.
    full_code: str = ""  # mã học phần (10 digits), "" when the row has none
    credits: str = ""  # "3(3,0,6)"
    conditions: tuple[tuple[str, str], ...] = ()  # (6-digit code, a|b|c)

    @property
    def prerequisites(self) -> tuple[str, ...]:
        return tuple(code for code, _ in self.conditions)


def course_rows(content: str) -> Iterator[CourseRow]:
    """Rows that name a course: a code cell followed by a name cell (the name is the first
    cell with letters after the first code)."""

    for cells in table_rows(content):
        start = next((i for i, cell in enumerate(cells) if _CODE.match(cell)), None)
        if start is None:
            continue
        name = vietnamese_name(
            next((c for c in cells[start + 1 :] if re.search(r"[^\W\d_]", c)), "")
        )
        if not name:
            continue
        rest = cells[start + 1 :]
        yield CourseRow(
            code=cells[start][-6:] if len(cells[start]) == 10 else cells[start],
            name=name,
            full_code=next((c for c in rest if len(c) == 10 and _CODE.match(c)), ""),
            credits=next((c for c in rest if _CREDITS.match(c)), ""),
            conditions=tuple(
                (m.group(1), m.group(2)) for c in rest for m in _CONDITION.finditer(c)
            ),
        )


def vietnamese_name(cell: str) -> str:
    """The Vietnamese name of a cell the older plans write in both languages ("Khóa luận
    tốt nghiệp Capstone project" -> "Khóa luận tốt nghiệp"): an English tail of two words
    of letters or more, from a capitalised plain-ASCII word to the end, after a Vietnamese
    part. "Tiếng Anh 2", "Toán cao cấp 1 Calculus 1" and "Lập trình Java" stay whole."""

    words = cell.split()
    for start in range(1, len(words) - 1):
        head, tail = words[:start], words[start:]
        if (
            tail[0][:1].isupper()
            and all(word.isascii() for word in tail)
            and sum(1 for word in tail if re.search(r"[A-Za-z]{2}", word)) >= 2
            and not all(word.isascii() for word in head)
        ):
            return " ".join(head)
    return cell


def _phrase_in(phrase: str, text: str) -> bool:
    return bool(phrase) and re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None


def asked_courses(names: Sequence[str], course: str | None, texts: Sequence[str]) -> list[str]:
    """Folded course names (from `names`, the tables' course names) the turn asks about:
    equal to the classifier's `course`, else written in `texts`. Names of two words or more
    only ("Toán" alone is too common a word), and a name inside a longer matching one
    ("Tiếng Anh" in "Tiếng Anh 2") is dropped."""

    candidates = {fold(name) for name in names}
    candidates = {name for name in candidates if len(name.split()) >= 2}
    if course and fold(course) in candidates:
        return [fold(course)]
    folded = [fold(text) for text in texts if text]
    found = {name for name in candidates if any(_phrase_in(name, text) for text in folded)}
    return sorted(name for name in found if not any(o != name and name in o for o in found))


def asked_codes(texts: Sequence[str]) -> set[str]:
    codes: set[str] = set()
    for text in texts:
        for match in _USER_CODE.finditer(text or ""):
            code = match.group(1)
            codes.add(code[-6:] if len(code) == 10 else code)
    return codes


async def lookup_course_rows(
    candidates: Sequence[RetrievedChunk],
    present: Sequence[RetrievedChunk],
    *,
    course: str | None,
    texts: Sequence[str],
    fetch_documents: FetchDocuments,
    titles: Titles | None = None,
) -> CourseLookup:
    """The chunks (not in `present`) holding the row of the course asked about, or of a
    course it names as a prerequisite, from the programme documents among `candidates` and
    (with `titles`) their `sibling_documents`."""

    documents = programme_documents(candidates)
    if not documents:
        return CourseLookup()
    if titles is not None:
        siblings = sibling_documents(candidates, await titles(), texts)
        documents = list(dict.fromkeys([*siblings, *documents]))[:MAX_DOCUMENTS]
    chunks = await fetch_documents(documents)
    rows_of = {chunk.chunk_id: list(course_rows(chunk.content)) for chunk in chunks}
    names = asked_courses(
        [row.name for rows in rows_of.values() for row in rows], course, [course or "", *texts]
    )
    codes = asked_codes(texts)
    if not names and not codes:
        return CourseLookup(documents=len(documents))
    # (chunk, is a prerequisite row): the asked rows of every document go before the
    # prerequisite rows, so MAX_ADDED cuts prerequisites first.
    matched: dict[str, tuple[RetrievedChunk, bool]] = {}
    prerequisites: dict[str, set[str]] = {}
    for chunk in chunks:
        for row in rows_of[chunk.chunk_id]:
            if fold(row.name) in names or (row.code and row.code in codes):
                matched[chunk.chunk_id] = (chunk, False)
                prerequisites.setdefault(document_id(chunk), set()).update(row.prerequisites)
    for chunk in chunks:
        wanted = prerequisites.get(document_id(chunk), set())
        if chunk.chunk_id not in matched and any(
            row.code in wanted for row in rows_of[chunk.chunk_id]
        ):
            matched[chunk.chunk_id] = (chunk, True)
    have = {chunk.chunk_id for chunk in present}
    order = {doc: index for index, doc in enumerate(documents)}
    ranked = sorted(
        matched.values(), key=lambda item: (item[1], order.get(document_id(item[0]), len(order)))
    )
    added = [
        chunk.model_copy(update={"metadata": {**chunk.metadata, COURSE_ROW_FLAG: True}})
        for chunk, _ in ranked
        if chunk.chunk_id not in have
    ][:MAX_ADDED]
    by_document: dict[str, dict[str, str]] = {}
    for chunk in chunks:
        for row in rows_of[chunk.chunk_id]:
            by_document.setdefault(document_id(chunk), {}).setdefault(row.code, row.name)
    display = (
        next(
            (row.name for rows in rows_of.values() for row in rows if fold(row.name) == names[0]),
            None,
        )
        if len(names) == 1
        else None
    )
    return CourseLookup(
        added=added,
        courses=names or sorted(codes),
        documents=len(documents),
        names=by_document,
        course_name=display,
    )


def course_summary(chunks: Sequence[RetrievedChunk], lookup: CourseLookup) -> str:
    """Prompt section: the asked courses' rows of `chunks` (the final context, numbered
    [1..n] in this order) written out by code - name, codes, credits and each condition
    with its kind and the named course. The table is 17 columns wide and the model read
    "Tiếng Anh 2 ... 015253(b)" as Tiếng Anh 2 being Tiếng Anh 1's prerequisite
    (25/11/2026). "" when no row of an asked course is in the context."""

    if not lookup.courses:
        return ""
    lines: list[str] = []
    seen: set[tuple[str, str]] = set()
    for index, chunk in enumerate(chunks, start=1):
        doc = document_id(chunk)
        for row in course_rows(chunk.content):
            asked = fold(row.name) in lookup.courses or row.code in lookup.courses
            if not asked or (doc, row.code) in seen:
                continue
            seen.add((doc, row.code))
            names = lookup.names.get(doc, {})
            # The kinds the row has, then the missing ones in one phrase: asked for "tiên
            # quyết" with only a "học trước (a)" written, the answer read "Học phần tiên
            # quyết ... là học trước môn Toán cao cấp 1" (25/11/2026) - the missing kind must
            # be said, without a "không có" per kind the answer then copied word for word.
            present = [
                f"{label} ({kind}): "
                + ", ".join(
                    code + (f" {names[code]}" if code in names else "")
                    for code, of_kind in row.conditions
                    if of_kind == kind
                )
                for kind, label in _CONDITION_KINDS.items()
                if any(of_kind == kind for _, of_kind in row.conditions)
            ]
            missing = [
                label
                for kind, label in _CONDITION_KINDS.items()
                if not any(of_kind == kind for _, of_kind in row.conditions)
            ]
            if present and missing:
                conditions = "; ".join([*present, f"không ghi {', '.join(missing)}"])
            elif present:
                conditions = "; ".join(present)
            else:
                conditions = "không ghi điều kiện học trước, tiên quyết hay song hành"
            lines.append(
                f"- [{index}] {source_title(chunk.source)}: học phần {row.name} (mã môn học "
                f"{row.code}"
                + (f", mã học phần {row.full_code}" if row.full_code else "")
                + ")"
                + (f", số tín chỉ {row.credits}" if row.credits else "")
                + f"; điều kiện của học phần này - {conditions}."
            )
    if not lines:
        return ""
    return (
        "\n\n## DÒNG HỌC PHẦN ĐÃ ĐỌC TỪ BẢNG (chính xác, dùng thay cho việc tự dò "
        "bảng)\nMỗi dòng là DỮ LIỆU của một học phần được hỏi, đọc từ đoạn [n] ghi ở đầu "
        "dòng (trích dẫn đúng [n] đó): diễn đạt lại thành câu văn theo phong cách trả "
        'lời, không chép nguyên dòng hay ký hiệu (a)/(b)/(c). "Điều kiện của học phần '
        'này" là học phần người học phải học trước / đạt trước / học cùng ĐỂ học học phần '
        "được hỏi - không phải ngược lại. Gọi ĐÚNG loại ghi trong dòng kể cả khi người "
        'dùng dùng từ khác: hỏi "tiên quyết" mà dòng ghi học trước (a) → nói rõ học phần '
        "đó là học phần phải HỌC TRƯỚC, không gọi là tiên quyết. Câu trả lời mở bằng đúng "
        'loại được hỏi: loại đó không ghi → nói thẳng "<học phần> không có học phần '
        '<loại được hỏi>" rồi mới nêu loại khác có ghi (VD "Cơ lưu chất không có học phần '
        'tiên quyết; học phần phải học trước là Toán cao cấp 1 (003259)").\n' + "\n".join(lines)
    )
