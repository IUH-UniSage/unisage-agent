from pathlib import Path


def load_text_file(path: Path) -> str:
    """Load a UTF-8 text document for local ingestion jobs."""

    return path.read_text(encoding="utf-8")
