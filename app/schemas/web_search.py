from pydantic import BaseModel, Field


class WebSearchResult(BaseModel):
    """One web page returned by the web search provider for a sub-query.

    `content` is the provider's relevance-ranked snippet of the page, not the
    full page text."""

    title: str
    url: str
    content: str
    score: float = Field(ge=0.0, le=1.0)
