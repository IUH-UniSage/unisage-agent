"""Assign simulated access labels to `manifest.csv` (Task 6).

Fills `department_id` (department code from `label_map.yaml`), `is_public` and
`access_level`, following the Qdrant visibility rule and the backend's
`documents_public_access_level_check`: a public document has no access level.

Within each (department, `quality == ok`) group, `private_ratio` of the documents
- picked with a fixed seed, so re-running gives the same labels - become
private with a level from `private_levels`. Grouping by quality keeps private
documents among the ones the evaluation actually ingests. Rows marked
`label_source=manual` are never touched, so hand corrections survive re-runs.

Usage:
    python -m evals.label --dataset ../unisage-gateway/dataset/official
"""

import argparse
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from evals.crawl.download import DownloadState

DEFAULT_MAP = Path(__file__).with_name("label_map.yaml")


class LabelMapError(Exception):
    pass


@dataclass(frozen=True)
class UnitRule:
    department: str
    force_public: bool


@dataclass(frozen=True)
class LabelMap:
    seed: int
    private_ratio: float
    private_levels: tuple[int, ...]
    units: dict[str, UnitRule]


def load_label_map(path: Path) -> LabelMap:
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    known = set(raw["known_departments"])
    units = {
        unit: UnitRule(str(rule["department"]), bool(rule.get("force_public", False)))
        for unit, rule in raw["units"].items()
    }
    unknown = sorted({rule.department for rule in units.values()} - known)
    if unknown:
        raise LabelMapError(f"department codes not seeded by the backend: {unknown}")
    return LabelMap(
        seed=int(raw["seed"]),
        private_ratio=float(raw["private_ratio"]),
        private_levels=tuple(int(level) for level in raw["private_levels"]),
        units=units,
    )


def private_count(size: int, ratio: float) -> int:
    if size < 2:
        return 0
    return max(1, math.floor(size * ratio + 0.5))


def label_rows(rows: list[dict[str, str]], label_map: LabelMap) -> None:
    """Label `rows` in place."""

    missing = sorted({row["unit"] for row in rows} - label_map.units.keys())
    if missing:
        raise LabelMapError(f"units without a rule in label_map.yaml: {missing}")

    groups: dict[tuple[str, bool], list[dict[str, str]]] = {}
    for row in rows:
        if row.get("label_source") == "manual":
            continue
        rule = label_map.units[row["unit"]]
        row["department_id"] = rule.department
        row["is_public"] = "true"
        row["access_level"] = ""
        row["label_source"] = "auto"
        if not rule.force_public:
            key = (rule.department, row.get("quality") == "ok")
            groups.setdefault(key, []).append(row)

    for (department, is_ok), group in sorted(groups.items()):
        rng = random.Random(f"{label_map.seed}:{department}:{is_ok}")
        group.sort(key=lambda row: row["file_id"])
        for row in rng.sample(group, private_count(len(group), label_map.private_ratio)):
            row["is_public"] = "false"
            row["access_level"] = str(rng.choice(label_map.private_levels))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--label-map", type=Path, default=DEFAULT_MAP)
    args = parser.parse_args()

    state = DownloadState(args.dataset)
    label_rows(state.manifest, load_label_map(args.label_map))
    state.save()

    ok = [row for row in state.manifest if row.get("quality") == "ok"]
    private = [row for row in ok if row["is_public"] == "false"]
    print(f"labelled {len(state.manifest)} rows; quality=ok: {len(ok)}, private: {len(private)}")
    print(
        "private access levels (ok):", sorted(Counter(r["access_level"] for r in private).items())
    )
    print("departments (ok):", len({row["department_id"] for row in ok}))


if __name__ == "__main__":
    main()
