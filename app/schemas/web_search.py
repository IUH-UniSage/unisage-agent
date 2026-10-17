from typing import Literal, get_args

from pydantic import BaseModel, Field

# What a web search is about, set by MessageClassificationNode. It picks the query template
# and the university sites tried first (app/rag/web_query.py), how long a result is cached,
# and whether the search starts before retrieval (CHAT_WEB_SEARCH_SPECULATIVE_TOPICS).
WebTopic = Literal[
    "hoc_phi",
    "tuyen_sinh",
    "lich_thong_bao",
    "lien_he_co_so",
    "quy_che_thu_tuc",
    "chuong_trinh_dao_tao",
    "khac",
]
WEB_TOPICS: tuple[str, ...] = get_args(WebTopic)


class WebSlots(BaseModel):
    """What the web search of one advisory task looks for, as slots the classifier fills
    and code turns into a keyword query - never free text the model writes itself, so the
    same question always becomes the same query (and cache key)."""

    # The first one picks the query template; a MULTI task may span several.
    topics: list[WebTopic] = Field(min_length=1)
    subject: str
    bac_dao_tao: str | None = None
    he_dao_tao: str | None = None
    nganh_hoc: str | None = None
    co_so: str | None = None
    # Only when the user named one ("2025-2026"); otherwise the current academic year.
    nam_hoc: str | None = None


class WebSearchResult(BaseModel):
    """One web page returned by the web search provider for a sub-query.

    `content` is the provider's relevance-ranked snippet of the page, not the
    full page text."""

    title: str
    url: str
    content: str
    score: float = Field(ge=0.0, le=1.0)
    # The academic year the page is about ("2026-2027"), read from its title, URL or
    # content - None when none is written there.
    academic_year: str | None = None
