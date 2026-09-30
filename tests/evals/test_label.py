from pathlib import Path

import pytest

from evals.label import (
    DEFAULT_MAP,
    LabelMap,
    LabelMapError,
    UnitRule,
    label_rows,
    load_label_map,
    private_count,
)

LABEL_MAP = LabelMap(
    seed=1,
    private_ratio=0.3,
    private_levels=(1, 2, 3, 4),
    units={
        "Phòng Đào tạo": UnitRule("PHONG_DAO_TAO", force_public=False),
        "Tuyển sinh": UnitRule("PHONG_DAO_TAO", force_public=True),
        "Khoa X": UnitRule("KHOA_CNTT", force_public=False),
    },
)


def _rows(unit: str, count: int, quality: str = "ok") -> list[dict[str, str]]:
    return [
        {"file_id": f"{unit}-{quality}-{i:03d}", "unit": unit, "quality": quality}
        for i in range(count)
    ]


def test_public_rows_never_carry_an_access_level():
    rows = _rows("Phòng Đào tạo", 20) + _rows("Tuyển sinh", 5) + _rows("Khoa X", 1)
    label_rows(rows, LABEL_MAP)

    for row in rows:
        assert row["label_source"] == "auto"
        if row["is_public"] == "true":
            assert row["access_level"] == ""
        else:
            assert row["access_level"] in {"1", "2", "3", "4"}
    daotao = [row for row in rows if row["unit"] == "Phòng Đào tạo"]
    assert sum(row["is_public"] == "false" for row in daotao) == 6
    assert all(row["is_public"] == "true" for row in rows if row["unit"] == "Tuyển sinh")
    assert all(
        row["is_public"] == "true" for row in rows if row["unit"] == "Khoa X"
    )  # only one doc
    assert {row["department_id"] for row in rows if row["unit"] == "Tuyển sinh"} == {
        "PHONG_DAO_TAO"
    }


def test_private_share_is_per_quality_group():
    rows = _rows("Phòng Đào tạo", 10, "ok") + _rows("Phòng Đào tạo", 10, "scanned")
    label_rows(rows, LABEL_MAP)
    ok_private = [r for r in rows if r["quality"] == "ok" and r["is_public"] == "false"]
    assert len(ok_private) == 3


def test_labels_are_deterministic_and_keep_manual_rows():
    first = _rows("Phòng Đào tạo", 12)
    second = [dict(row) for row in first]
    second[0].update(label_source="manual", department_id="PHONG_CTSV", is_public="true")
    label_rows(first, LABEL_MAP)
    label_rows(second, LABEL_MAP)

    assert second[0]["department_id"] == "PHONG_CTSV"
    assert second[0]["label_source"] == "manual"
    again = [dict(row, label_source="") for row in first]
    label_rows(again, LABEL_MAP)
    assert [(r["is_public"], r["access_level"]) for r in again] == [
        (r["is_public"], r["access_level"]) for r in first
    ]


def test_unknown_unit_is_rejected():
    with pytest.raises(LabelMapError, match="Khoa Y"):
        label_rows(_rows("Khoa Y", 2), LABEL_MAP)


@pytest.mark.parametrize(("size", "expected"), [(0, 0), (1, 0), (2, 1), (5, 2), (10, 3), (100, 30)])
def test_private_count(size, expected):
    assert private_count(size, 0.3) == expected


def test_shipped_label_map_only_uses_seeded_departments():
    label_map = load_label_map(DEFAULT_MAP)
    assert label_map.units


def test_label_map_rejects_unseeded_department(tmp_path: Path):
    path = tmp_path / "map.yaml"
    path.write_text(
        "seed: 1\nprivate_ratio: 0.3\nprivate_levels: [1]\nknown_departments: [A]\n"
        "units:\n  X: {department: B}\n",
        encoding="utf-8",
    )
    with pytest.raises(LabelMapError, match="B"):
        load_label_map(path)
