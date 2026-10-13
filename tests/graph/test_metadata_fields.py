from datetime import date

from app.calculation.formulas import FORMULAS
from app.graph.clarification_round import slug
from app.graph.metadata_fields import (
    apply_updates,
    canonical_field,
    canonical_params,
    canonical_profile,
    field_scope,
    is_profile_field,
    normalize_value,
    registered_name,
)
from app.rag.prompting.loader import get_known_metadata_fields

_TODAY = date(2026, 10, 13)


def test_every_formula_parameter_is_a_registered_field() -> None:
    registered = {entry["field"] for entry in get_known_metadata_fields()}
    for formula in FORMULAS.values():
        for spec in formula.params:
            assert spec.name in registered, (formula.formula_id, spec.name)
            assert canonical_field(spec.name) == spec.name


def test_the_fee_schedule_context_params_are_profile_fields() -> None:
    assert is_profile_field("khoa_nhap_hoc")
    assert is_profile_field("khoi_nganh")
    assert not is_profile_field("tc_lt")


def test_no_alias_names_two_fields_or_shadows_a_field() -> None:
    entries = get_known_metadata_fields()
    fields = {entry["field"] for entry in entries}
    seen: dict[str, str] = {}
    for entry in entries:
        assert entry["scope"] in {"profile", "calculation"}
        for alias in entry["key_aliases"]:
            key = slug(alias)
            assert key not in fields, key
            assert seen.setdefault(key, entry["field"]) == entry["field"], key


def test_key_aliases_resolve_to_one_name() -> None:
    assert canonical_field("khoa_tuyen_sinh") == "khoa_nhap_hoc"
    assert canonical_field("so_tc_lt") == "tc_lt"
    assert canonical_field("Training Type") == "he_dao_tao"
    assert canonical_field("da_nop") == "so_tien_da_nop"
    assert canonical_field("truong_moi_la") == "truong_moi_la"


def test_unknown_fields_take_the_scope_of_where_they_were_asked() -> None:
    assert field_scope("truong_moi_la", "profile") == "profile"
    assert field_scope("truong_moi_la", "calculation") == "calculation"
    assert field_scope("so_tc_lt", "profile") == "calculation"


def test_profile_values_are_stored_normalised() -> None:
    assert normalize_value("khoa_tuyen_sinh", "kỳ 9", _TODAY) == "2022"
    assert normalize_value("khoa_nhap_hoc", "không rõ", _TODAY) is None
    assert normalize_value("nganh_hoc", "  Công nghệ   thông tin ", _TODAY) == "Công nghệ thông tin"
    assert normalize_value("nganh_hoc", "", _TODAY) is None


def test_apply_updates_touches_only_profile_fields_and_reports_changes() -> None:
    profile = {"khoa_nhap_hoc": "2022", "nganh_hoc": "Kế toán"}
    merged, changed = apply_updates(
        profile,
        {"khoa_tuyen_sinh": "2023", "nganh_hoc": None, "tc_lt": 15, "he_dao_tao": "Chính quy"},
        _TODAY,
    )
    assert merged == {"khoa_nhap_hoc": "2023", "he_dao_tao": "Chính quy"}
    assert changed == {"khoa_nhap_hoc": "2023", "nganh_hoc": None, "he_dao_tao": "Chính quy"}
    assert profile == {"khoa_nhap_hoc": "2022", "nganh_hoc": "Kế toán"}


def test_an_update_that_changes_nothing_is_not_reported() -> None:
    _, changed = apply_updates({"khoa_nhap_hoc": "2022"}, {"khoa_nhap_hoc": "K18"}, _TODAY)
    assert changed == {}


def test_canonical_params_keep_the_first_value_on_a_clash() -> None:
    assert canonical_params({"so_tc_lt": 10, "tc_lt": 12, "gk": 7}) == {"tc_lt": 10, "gk": 7}


def test_a_stored_profile_reads_back_under_registry_names() -> None:
    # A row migrated from the per-conversation `confirmed_metadata` (aliases, raw intake).
    stored = {
        "cohort": "K18",
        "nganh": " CNTT ",
        "khoa_nhap_hoc": "2020",
        "ghi_chu": "",
        "lop": "x",
    }
    assert canonical_profile(stored, _TODAY) == {
        "khoa_nhap_hoc": "2022",
        "nganh_hoc": "CNTT",
        "lop": "x",
    }


def test_registered_name_keeps_an_unregistered_label_as_written() -> None:
    assert registered_name("so_tc_lt") == "tc_lt"
    label = "Số tín chỉ lý thuyết của một năm học"
    assert registered_name(label) == label
    assert registered_name("Số tín chỉ lý thuyết") == "tc_lt"
