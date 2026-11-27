"""Fee row lookup: the rows of a fee schedule (biểu mức thu học phí) a fee question asks for,
read from the whole schedule in code (no LLM).

A fee schedule is one table cut into a handful of chunks, its rows named by training system
(Đại học chính quy, Cao học...), intake ("Khóa tuyển sinh năm học 2022-2023", "... 2021-2022
trở về trước") and group (Khối Công nghệ, Khối Kinh tế, Ngành Dược) - never by major. A
question naming a major and a cohort ("ngành CNTT khóa K18") reads like the programmes'
titles ("... khóa 18 năm học 2022-2023"), so the search and the cross-encoder keep those and
drop the schedule's chunk with the asked row: the K18 answer took the "2021-2022 trở về
trước" row instead of "2022-2023" (27/11/2026). This reads the schedules (the ones among
the candidates, and the newest one of the catalogue) and adds the chunks holding the asked
rows, with a summary of those rows for the prompt.

The asked group comes from the question ("khối công nghệ") or from the major by convention
(engineering, technology and science majors -> Khối Công nghệ; economics, business, law,
tourism -> Khối Kinh tế; the documents themselves name no major per group except the
"Ngôn ngữ Anh, Quản lý đất đai, Tài nguyên thiên nhiên" rows and Ngành Dược). Unknown group
-> every group's row of the asked intake. Nothing asked about a fee -> nothing changes.
"""

import logging
import re
from collections.abc import Awaitable, Callable, Iterator, Sequence
from dataclasses import dataclass, field

from app.graph.cohort_scope import asked_cohort
from app.graph.course_lookup import is_programme_document
from app.graph.major_scope import asked_majors, fold
from app.rag.prompting.citations import source_title
from app.schemas.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

MAX_DOCUMENTS = 3
MAX_DOCUMENT_CHUNKS = 40
MAX_ADDED = 4
FEE_ROW_FLAG = "fee_row"

FetchDocuments = Callable[[Sequence[str]], Awaitable[Sequence[RetrievedChunk]]]
Titles = Callable[[], Awaitable[dict[str, str]]]

# A question about an amount of tuition (not its deadline, payment channel or exemption).
_FEE_QUESTION = re.compile(r"\b(hoc phi|muc thu|don gia (01 |1 |mot )?tin chi|tien hoc)\b")
_NOT_AN_AMOUNT = re.compile(r"\b(han (dong|nop|chot)|thoi han|cach (dong|nop)|mien|giam|hoan)\b")
_FEE_TITLE = re.compile(r"\b(hoc phi|muc thu)\b")
# A deadline notice or an exemption rule names "học phí" too, without a schedule.
_NOT_A_SCHEDULE = re.compile(r"\b(mien|giam|han (dong|nop)|thong bao|hoan)\b")

# Training systems, most specific first (an intensive-English row also says "chính quy").
_SYSTEMS = (
    ("tang cuong tieng anh", re.compile(r"\btang cuong tieng anh\b")),
    ("lien thong / van bang 2", re.compile(r"\b(lien thong|van bang 2|van bang hai)\b")),
    ("vua lam vua hoc", re.compile(r"\b(vua lam vua hoc|tu xa)\b")),
    ("nghien cuu sinh", re.compile(r"\b(nghien cuu sinh|tien si)\b")),
    ("cao hoc", re.compile(r"\b(cao hoc|thac si)\b")),
    ("dai hoc chinh quy", re.compile(r"\b(dai hoc chinh quy|chinh quy|dai hoc)\b")),
)
_BRANCH = re.compile(r"\b(phan hieu|quang ngai)\b")

# Majors by group, by convention (no document lists them).
_SPECIAL_ECONOMICS = re.compile(
    r"\b(ngon ngu anh|nn anh|quan ly dat dai|ql dat dai|tai nguyen thien nhien|tntn)\b"
)
_PHARMACY = re.compile(r"\b(nganh duoc|duoc hoc|duoc si)\b")
_ECONOMICS = re.compile(
    r"\b(kinh te|ke toan|kiem toan|quan tri|tai chinh|ngan hang|marketing|kinh doanh|thuong mai"
    r"|luat|du lich|khach san|nha hang|logistics|ngon ngu)\b"
)
_TECHNOLOGY = re.compile(
    r"\b(cong nghe|ky thuat|khoa hoc may tinh|khoa hoc du lieu|he thong thong tin|tri tue nhan tao"
    r"|phan mem|moi truong|hoa hoc|thuc pham|sinh hoc|det may|may thoi trang|co khi|dien tu|o to"
    r"|nhiet|xay dung|robot|iot)\b"
)
GROUPS = ("khoi cong nghe", "khoi kinh te", "khoi kinh te dac biet", "nganh duoc")
_GROUP_IN_ROW = (
    ("nganh duoc", re.compile(r"\b(nganh duoc|duoc hoc)\b")),
    ("khoi kinh te dac biet", re.compile(r"\b(ngon ngu anh|nn anh)\b")),
    ("khoi kinh te", re.compile(r"\b(khoi kinh te|khoi kt|kt)\b")),
    ("khoi cong nghe", re.compile(r"\b(khoi cong nghe|khoi cn|cn|cong nghe)\b")),
)
_YEARS = re.compile(r"\b(20\d\d)\s+(20\d\d)\b")
_BEFORE = re.compile(r"\btro ve truoc\b")
_NUMBER = re.compile(r"\d{1,3}(?:\.\d{3})+|\d+k")


@dataclass
class FeeRow:
    document_id: str
    chunk_id: str
    path: str  # breadcrumb + row label, as written
    folded: str
    values: list[tuple[str, str]]  # (column name, value)
    system: str | None
    intake: tuple[int, bool] | None  # (start year, "trở về trước")
    groups: set[str]


@dataclass
class FeeLookup:
    added: list[RetrievedChunk] = field(default_factory=list)
    rows: list[FeeRow] = field(default_factory=list)
    documents: int = 0
    # What was asked, for the summary: system, intake year, groups (empty = every group).
    system: str | None = None
    system_asked: bool = False
    intake: int | None = None
    groups: set[str] = field(default_factory=set)
    # The intake was asked and the schedule has neither its row nor a merged row holding it.
    missing_intake: bool = False


def document_id(chunk: RetrievedChunk) -> str:
    return str(chunk.metadata.get("document_id") or chunk.chunk_id.split(":", 1)[0])


def is_fee_question(texts: Sequence[str]) -> bool:
    folded = fold(" ".join(texts))
    return bool(_FEE_QUESTION.search(folded)) and not _NOT_AN_AMOUNT.search(folded)


def asked_system(texts: Sequence[str]) -> str | None:
    folded = fold(" ".join(texts))
    for name, pattern in _SYSTEMS:
        if pattern.search(folded):
            return name
    return None


def groups_of_text(text: str) -> set[str]:
    """Groups a question asks about: named ("khối công nghệ") or from its major."""

    folded = fold(text)
    named = set()
    if re.search(r"\bkhoi cong nghe\b", folded):
        named.add("khoi cong nghe")
    if re.search(r"\bkhoi kinh te\b", folded):
        named.add("khoi kinh te")
    if named:
        return named
    majors = " ".join(asked_majors(text, [])) + " " + folded
    if _PHARMACY.search(majors):
        return {"nganh duoc"}
    if _SPECIAL_ECONOMICS.search(majors):
        return {"khoi kinh te dac biet"}
    if not asked_majors(text, []) and not re.search(r"\bnganh\b", folded):
        return set()
    economics = bool(_ECONOMICS.search(majors))
    technology = bool(_TECHNOLOGY.search(majors))
    if economics != technology:
        return {"khoi kinh te"} if economics else {"khoi cong nghe"}
    return set()  # no major, or one the convention cannot place: every group


def _row_groups(folded: str) -> set[str]:
    groups = {name for name, pattern in _GROUP_IN_ROW if pattern.search(folded)}
    if "khoi kinh te dac biet" in groups:
        groups.discard("khoi kinh te")
    if "nganh duoc" in groups:
        groups &= {"nganh duoc"}
    return groups


def _row_system(folded: str) -> str | None:
    for name, pattern in _SYSTEMS:
        if pattern.search(folded):
            return name
    return None


def fee_rows(chunk: RetrievedChunk) -> Iterator[FeeRow]:
    """Rows of a fee table chunk, each with the breadcrumb above it ("[A ... > 3 Đại học chính
    quy > 3.4 Khóa tuyển sinh năm học 2022-2023]") and its header's column names."""

    header: list[str] = []
    crumb = ""
    for raw in chunk.content.split("\n"):
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            crumb = line[1:-1]
            continue
        if not line.startswith("|"):
            continue
        # "\\|" is a pipe inside a cell, not a cell border.
        cells = [
            " ".join(cell.replace("\\|", "|").replace("<br>", " ").split())
            for cell in re.split(r"(?<!\\)\|", line.strip("|"))
        ]
        if all(set(cell) <= set("-: ") for cell in cells):
            continue
        if not header:
            header = cells
            continue
        label_at = next(
            (i for i, cell in enumerate(cells) if re.search(r"[A-Za-zÀ-ỹ]{2}", cell)), None
        )
        if label_at is None:
            continue
        values = [
            (header[i].split(">")[-1].strip() if i < len(header) else "", cell)
            for i, cell in enumerate(cells)
            if i > label_at and _NUMBER.search(cell)
        ]
        path = f"{crumb} > {cells[label_at]}" if crumb else cells[label_at]
        folded = fold(path)
        years = _YEARS.findall(folded)
        intake = (int(years[-1][0]), bool(_BEFORE.search(folded))) if years else None
        yield FeeRow(
            document_id=document_id(chunk),
            chunk_id=chunk.chunk_id,
            path=path,
            folded=folded,
            values=values,
            system=_row_system(folded),
            intake=intake,
            groups=_row_groups(fold(cells[label_at])) or _row_groups(folded),
        )


def fee_documents(candidates: Sequence[RetrievedChunk], titles: dict[str, str] | None) -> list[str]:
    """Fee schedules among the candidates (by title) first, then the newest of the catalogue
    (the latest school year in the title)."""

    found: list[str] = []
    for chunk in candidates:
        doc = document_id(chunk)
        title = fold(source_title(chunk.source))
        if doc not in found and _FEE_TITLE.search(title) and not _NOT_A_SCHEDULE.search(title):
            found.append(doc)
    if titles:
        catalogue = [
            (max((int(a) for a, _ in _YEARS.findall(fold(title))), default=0), doc)
            for doc, title in titles.items()
            if _FEE_TITLE.search(fold(source_title(title)))
            and not _NOT_A_SCHEDULE.search(fold(source_title(title)))
        ]
        found += [doc for _, doc in sorted(catalogue, reverse=True) if doc not in found]
    return found[:MAX_DOCUMENTS]


def _matches(row: FeeRow, system: str, branch: bool, intake: int | None, groups: set[str]) -> bool:
    if row.system != system or bool(_BRANCH.search(row.folded)) != branch or not row.values:
        return False
    if intake is not None and row.intake is not None:
        start, before = row.intake
        if not (start == intake or (before and intake <= start)):
            return False
    elif intake is not None:
        return False
    return not groups or not row.groups or bool(groups & row.groups)


async def lookup_fee_rows(
    candidates: Sequence[RetrievedChunk],
    present: Sequence[RetrievedChunk],
    *,
    texts: Sequence[str],
    fetch_documents: FetchDocuments,
    titles: Titles | None = None,
) -> FeeLookup:
    if not is_fee_question(texts):
        return FeeLookup()
    documents = fee_documents(candidates, await titles() if titles is not None else None)
    if not documents:
        return FeeLookup()
    chunks = await fetch_documents(documents)
    rows = [row for chunk in chunks for row in fee_rows(chunk)]
    system = asked_system(texts)
    joined = " ".join(texts)
    intake = next((year for year in map(asked_cohort, texts) if year is not None), None)
    groups = set().union(*(groups_of_text(text) for text in texts))
    branch = bool(_BRANCH.search(fold(joined)))
    chosen_system = system or "dai hoc chinh quy"
    # The asked intake's own row wins over a merged "trở về trước" row holding it.
    matched = [row for row in rows if _matches(row, chosen_system, branch, intake, groups)]
    if intake is not None and any(row.intake == (intake, False) for row in matched):
        matched = [row for row in matched if row.intake == (intake, False)]
    elif intake is not None and matched:
        # Only the nearest merged row (the smallest start year still holding the intake).
        nearest = min(row.intake[0] for row in matched if row.intake)
        matched = [row for row in matched if row.intake and row.intake[0] == nearest]
    missing = (
        intake is not None
        and not matched
        and any(row.system == chosen_system and row.intake for row in rows)
    )
    have = {chunk.chunk_id for chunk in present}
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    added: list[RetrievedChunk] = []
    for row in matched:
        if row.chunk_id in have or any(chunk.chunk_id == row.chunk_id for chunk in added):
            continue
        chunk = by_id[row.chunk_id]
        added.append(chunk.model_copy(update={"metadata": {**chunk.metadata, FEE_ROW_FLAG: True}}))
    return FeeLookup(
        added=added[:MAX_ADDED],
        rows=matched,
        documents=len(documents),
        system=chosen_system,
        system_asked=system is not None,
        intake=intake,
        groups=groups,
        missing_intake=missing,
    )


def without_programmes(chunks: Sequence[RetrievedChunk], lookup: FeeLookup) -> list[RetrievedChunk]:
    """`chunks` without the programme documents (chương trình / kế hoạch đào tạo) once the
    schedule's rows were found: they hold no fee, and their standard and intensive-English
    programmes made the program check ask which programme and the answer open with "thuộc
    chương trình tăng cường tiếng Anh" for a standard row (27/11/2026)."""

    if not lookup.rows:
        return list(chunks)
    return [chunk for chunk in chunks if not is_programme_document(chunk)]


_GROUP_LABEL = {
    "khoi cong nghe": "Khối Công nghệ",
    "khoi kinh te": "Khối Kinh tế",
    "khoi kinh te dac biet": "Khối Kinh tế (Ngôn ngữ Anh, Quản lý đất đai, Tài nguyên thiên nhiên)",
    "nganh duoc": "Ngành Dược",
}


def fee_summary(chunks: Sequence[RetrievedChunk], lookup: FeeLookup) -> str:
    """Prompt section: the asked rows of the final context (numbered [1..n] in this order),
    written out by code with their path and values. "" when no fee question was looked up."""

    if not lookup.documents or (not lookup.rows and not lookup.missing_intake):
        return ""
    wanted = {(row.chunk_id, row.path) for row in lookup.rows}
    lines: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        for row in fee_rows(chunk):
            if (chunk.chunk_id, row.path) not in wanted:
                continue
            wanted.discard((chunk.chunk_id, row.path))
            values = "; ".join(f"{name or 'mức thu'}: {value}" for name, value in row.values)
            lines.append(f"- [{index}] {source_title(chunk.source)}: {row.path} - {values}")
    asked = []
    if lookup.intake is not None:
        asked.append(
            f'khóa nhập học {lookup.intake} = dòng "Khóa tuyển sinh năm học '
            f'{lookup.intake}-{lookup.intake + 1}"'
        )
    if lookup.groups:
        asked.append(", ".join(_GROUP_LABEL[g] for g in sorted(lookup.groups)))
    if not lookup.system_asked:
        asked.append("câu hỏi không nêu hệ: dòng Đại học chính quy")
    head = (
        "\n\n## DÒNG BIỂU HỌC PHÍ ĐÃ ĐỌC TỪ BẢNG\n"
        "Code đọc nguyên biểu học phí và chọn các dòng dưới đây theo câu hỏi ("
        + "; ".join(asked)
        + "). Đây là DỮ LIỆU để trả lời: diễn đạt lại thành câu, trích dẫn đúng [n] của dòng, "
        "nói rõ khóa tuyển sinh và khối của dòng. Khối suy từ ngành theo quy ước chung "
        "(văn bản không liệt kê ngành theo khối): nêu ngắn rằng ngành đó thuộc khối này."
    )
    if lookup.missing_intake and not lines:
        return (
            head + f"\n- Biểu học phí không có dòng của khóa tuyển sinh năm học {lookup.intake}-"
            f'{lookup.intake + 1} (hay dòng gộp "trở về trước" chứa khóa này) cho hệ được hỏi: '
            "nói chưa tìm thấy, không lấy số của dòng khác."
        )
    if any(row.intake and row.intake[1] for row in lookup.rows) and lookup.intake is not None:
        head += (
            f' Biểu không có dòng riêng cho khóa nhập học {lookup.intake}: dòng gộp "trở về '
            'trước" dưới đây là dòng áp dụng.'
        )
    return head + "\n" + "\n".join(lines) if lines else ""
