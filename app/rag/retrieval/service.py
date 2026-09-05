from app.core.config import settings
from app.rag.retrieval.hybrid import hybrid_score
from app.schemas.retrieval import RetrievedChunk


class RetrievalService:
    """Retrieve academic context with a metadata-filtered fallback corpus."""

    _CORPUS = (
        RetrievedChunk(
            chunk_id="demo-001",
            content="Sinh viên tra cứu quy chế đào tạo theo chương trình và khóa học của mình.",
            source="Quy chế đào tạo mẫu",
            faculty="GLOBAL",
            score=0.0,
            metadata={"min_user_level": 1},
        ),
        RetrievedChunk(
            chunk_id="demo-002",
            content="Thông tin học vụ cần được đối chiếu với thông báo chính thức của nhà trường.",
            source="Sổ tay học vụ mẫu",
            faculty="GLOBAL",
            score=0.0,
            metadata={"min_user_level": 1},
        ),
        RetrievedChunk(
            chunk_id="demo-003",
            content="Tai lieu noi bo chi danh cho can bo quan ly hoc vu.",
            source="Huong dan quan tri mau",
            faculty="FIT",
            score=0.0,
            metadata={"min_user_level": 2},
        ),
    )

    def retrieve(
        self,
        query: str,
        *,
        user_faculty: str,
        user_level: int,
        limit: int | None = None,
    ) -> list[RetrievedChunk]:
        """Apply faculty visibility before ranking and returning context.

        `limit` defaults to `settings.RETRIEVAL_MAX_CHUNKS` (T1.9) rather
        than a hardcoded constant, per plan.md. Deliberately does not filter
        by `department_access`/permission - that is Phase 4 scope, not this
        phase.
        """

        effective_limit = limit if limit is not None else settings.RETRIEVAL_MAX_CHUNKS
        visible = [
            chunk.model_copy(update={"score": hybrid_score(query, chunk.content)})
            for chunk in self._CORPUS
            if chunk.faculty in {"GLOBAL", user_faculty}
            and int(chunk.metadata.get("min_user_level", 1)) <= user_level
        ]
        ranked = sorted(visible, key=lambda chunk: chunk.score, reverse=True)
        return ranked[:effective_limit]
