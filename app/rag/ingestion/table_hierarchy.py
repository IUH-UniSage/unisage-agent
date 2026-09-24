"""Infer each table row's ancestors (its parent rows, root first).

A hierarchical table (sections > groups > items) is flattened row by row; if a
chunk keeps only the leaf row, its parents are lost. This module reads the
hierarchy back from structural evidence only - nothing here knows any word of
any document:

- numbering: an ordinal column (`A.`, `1`, `1.1`, `a)`, `II`) or an ordinal
  prefix in the label, ranked by the order in which each *style* first appears
  (a stack: a style already on the stack pops back to its level, a new style
  pushes a deeper one);
- indentation: rank of the label's left edge (PDF geometry);
- merged label cell, bold/larger font, taller row (PDF geometry);
- numeric mask: which numeric columns a row fills - within one scope, if the
  first two unnumbered rows fill different columns, the first row's pattern is
  the parent pattern.

Each signal votes for the depth of the current row given the previous row's
depth; weights, thresholds and the tie rule are in `HIERARCHY_SCORING`
(deterministic - see AD9). A row whose winning depth is not clear enough gets
NO ancestors and a `hierarchy_uncertain` warning, and never disturbs the rows
around it: no guess is better than a wrong breadcrumb.
"""

import re
import statistics
from dataclasses import dataclass, field, replace

from app.rag.ingestion.canonical_table import TableBlock, TableRow


@dataclass(frozen=True)
class HierarchyScoring:
    weight_numbering: float = 0.35
    weight_indent: float = 0.25
    weight_span: float = 0.15
    weight_font: float = 0.10
    weight_mask: float = 0.10
    weight_height: float = 0.05
    # Total weight of the signals that have data for a row must reach this,
    # else the row is left alone (no votes are guessed). Indentation alone
    # (0.25) is enough: two or more indent clusters are real evidence.
    min_present_weight: float = 0.25
    assign_threshold: float = 0.60
    # The best depth must beat the runner-up by this share of the present
    # weight; closer (or exactly equal) is "no decision".
    tie_gap: float = 0.10
    indent_cluster_tolerance: float = 2.0  # points
    font_size_ratio: float = 1.05
    height_ratio: float = 1.3
    height_max_parent_share: float = 0.5
    ordinal_column_min_share: float = 0.6
    ordinal_column_min_count: int = 2
    ordinal_column_max_length: int = 8
    ordinal_prefix_min_share: float = 0.3
    ordinal_prefix_min_count: int = 3
    # Group rows (a label in a column LEFT of the label column while the label
    # cell itself is empty): how many make a hierarchy, and the most that can
    # be groups before the table is more likely just sparse.
    group_row_min_count: int = 2
    group_row_max_share: float = 0.6


HIERARCHY_SCORING = HierarchyScoring()

_NUMERIC = re.compile(r"^[\d.,\s%+-]+$")
_DOTTED = re.compile(r"^\(?(\d{1,2}(?:\.\d{1,2})+)[.)]?$")
_NUMBER = re.compile(r"^\(?(\d{1,3})[.)]?$")
_ROMAN_UPPER = re.compile(r"^\(?([IVXLCDM]{2,})[.)]?$")
_ROMAN_LOWER = re.compile(r"^\(?([ivxlcdm]{2,})[.)]?$")
_LETTER_UPPER = re.compile(r"^\(?([A-Z])[.)]?$")
_LETTER_LOWER = re.compile(r"^\(?([a-z])[.)]?$")
_PREFIX = re.compile(r"^(\S{1,12}?)\s+(\S.*)$", re.DOTALL)
_ROMAN_CHARS = set("IVXLCDM")
_GARBLED_WARNING = "garbled_text_raw_kept"
_GEOMETRY_SIGNALS = frozenset({"span", "font", "height"})


def _parse_ordinal(text: str) -> tuple[str, str] | None:
    """(style, token) of an ordinal cell, or None. Dotted numbers use
    1-2 digit groups so money (`1.300.000`) is never taken for `1.3.0.0`."""

    stripped = text.strip()
    if not stripped:
        return None
    match = _DOTTED.match(stripped)
    if match:
        return (f"dotted{match.group(1).count('.') + 1}", stripped)
    if _NUMBER.match(stripped):
        return ("number", stripped)
    if _ROMAN_UPPER.match(stripped):
        return ("roman_upper", stripped)
    if _ROMAN_LOWER.match(stripped):
        return ("roman_lower", stripped)
    if _LETTER_UPPER.match(stripped):
        return ("upper", stripped)
    if _LETTER_LOWER.match(stripped):
        return ("lower", stripped)
    return None


def _split_prefix(label: str) -> tuple[tuple[str, str] | None, str]:
    """An ordinal glued to the front of a label (`3.2 Some title`); requires
    punctuation or a multi-part number so plain words/quantities are safe."""

    match = _PREFIX.match(label.strip())
    if not match:
        return None, label.strip()
    token, rest = match.group(1), match.group(2)
    parsed = _parse_ordinal(token)
    if parsed is None:
        return None, label.strip()
    style, _ = parsed
    bare_number = style == "number" and not token.endswith((".", ")"))
    bare_letter = style in ("upper", "lower") and not token.endswith((".", ")"))
    if bare_number or bare_letter:
        return None, label.strip()
    return parsed, rest.strip()


def _clean(text: str) -> str:
    return " ".join(re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE).split())


@dataclass(frozen=True)
class _Layout:
    label_col: int
    ordinal_col: int | None
    numeric_cols: list[int]
    # row index -> group label, for rows that name a group instead of an item
    group_labels: dict[int, str] = field(default_factory=dict)


def _is_numeric(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and bool(_NUMERIC.match(stripped)) and any(c.isdigit() for c in stripped)


def _detect_layout(
    rows: list[TableRow], column_count: int, scoring: HierarchyScoring
) -> _Layout | None:
    def column(index: int) -> list[str]:
        return [row.cells[index].strip() for row in rows if index < len(row.cells)]

    text_weight = [
        sum(len(cell) for cell in column(index) if cell and not _is_numeric(cell))
        for index in range(column_count)
    ]
    if not any(text_weight):
        return None
    label_col = max(range(column_count), key=lambda index: text_weight[index])

    group_labels = _find_group_rows(rows, label_col, scoring)

    ordinal_col: int | None = None
    # With group rows the running number in the first column restarts inside
    # every group - it is an item counter, not a hierarchy.
    for index in range(0 if group_labels else label_col):
        cells = [cell for cell in column(index) if cell]
        if len(cells) < scoring.ordinal_column_min_count:
            continue
        if any(len(cell) > scoring.ordinal_column_max_length for cell in cells):
            continue
        matches = sum(1 for cell in cells if _parse_ordinal(cell) is not None)
        if matches / len(cells) >= scoring.ordinal_column_min_share:
            ordinal_col = index
            break

    skip = {label_col} | ({ordinal_col} if ordinal_col is not None else set())
    numeric_cols = []
    for index in range(column_count):
        if index in skip:
            continue
        cells = [cell for cell in column(index) if cell]
        if cells and sum(1 for cell in cells if _is_numeric(cell)) / len(cells) >= 0.5:
            numeric_cols.append(index)
    return _Layout(label_col, ordinal_col, numeric_cols, group_labels)


def _find_group_rows(
    rows: list[TableRow], label_col: int, scoring: HierarchyScoring
) -> dict[int, str]:
    """Rows that name a group rather than an item: the label cell is empty while
    a column to its left holds text that is neither a number nor an ordinal
    (`HỌC KỲ 1`, `Nhóm 2` sitting in the first column of a course list). Purely
    positional - no word is looked for. Fewer than a couple of them, or so many
    that the table is mostly such rows, means this is not a group layout."""

    groups: dict[int, str] = {}
    for index, row in enumerate(rows):
        if label_col < len(row.cells) and row.cells[label_col].strip():
            continue
        left = [_clean(cell) for cell in row.cells[: min(label_col, len(row.cells))]]
        texts = [text for text in left if text]
        # the label may be spread over neighbouring cells (`Nhóm` | `2`)
        if any(not _is_numeric(text) and _parse_ordinal(text) is None for text in texts):
            groups[index] = " ".join(texts)
    if len(groups) < scoring.group_row_min_count or len(groups) > scoring.group_row_max_share * len(
        rows
    ):
        return {}
    return groups


_DIGITS = re.compile(r"\d+")


def _group_depths(group_labels: dict[int, str], row_count: int) -> list[int | None]:
    """Depth of each group row, from the SHAPE of its label (digits masked, so
    `Nhóm 1` and `Nhóm 2` are one kind) and where it sits:

    - a kind already open pops back to its own level (a repeat);
    - a new kind directly after another group row is its child;
    - a new kind after item rows is a sibling of the previous group row (both
      head their own run of items).
    """

    stack: list[tuple[str, int]] = []
    depths: list[int | None] = [None] * row_count
    previous_was_group = False
    for index in range(row_count):
        label = group_labels.get(index)
        if label is None:
            previous_was_group = False
            continue
        kind = _DIGITS.sub("#", label.strip())
        known = next((i for i, (seen, _d) in enumerate(stack) if seen == kind), None)
        if known is not None:
            del stack[known + 1 :]
            depth = stack[known][1]
        elif previous_was_group and stack:
            depth = stack[-1][1] + 1
            stack.append((kind, depth))
        elif stack:
            depth = stack[-1][1]
            stack[-1] = (kind, depth)
        else:
            depth = 0
            stack.append((kind, depth))
        depths[index] = depth
        previous_was_group = True
    return depths


def _ordinal_depths(parsed: list[tuple[str, str] | None]) -> list[int | None]:
    """Depth of each ordinal row by first-appearance stack of styles."""

    if any(item and item[0] == "roman_upper" for item in parsed):
        parsed = [
            ("roman_upper", item[1])
            if item and item[0] == "upper" and item[1].strip("().") in _ROMAN_CHARS
            else item
            for item in parsed
        ]
    stack: list[str] = []
    depths: list[int | None] = []
    for item in parsed:
        if item is None:
            depths.append(None)
            continue
        style = item[0]
        if style in stack:
            index = stack.index(style)
            del stack[index + 1 :]
            depths.append(index)
        else:
            stack.append(style)
            depths.append(len(stack) - 1)
    return depths


def _binary_feature(
    values: list[bool | None], numbered: list[bool], scope: list[int]
) -> list[bool | None]:
    """A True/False class per unnumbered row, kept only in scopes where the
    unnumbered rows actually contain both classes (else the class says
    nothing about hierarchy)."""

    per_scope: dict[int, set[bool]] = {}
    for value, is_numbered, scope_id in zip(values, numbered, scope, strict=True):
        if value is not None and not is_numbered:
            per_scope.setdefault(scope_id, set()).add(value)
    return [
        value
        if (value is not None and not is_numbered and per_scope.get(scope_id) == {True, False})
        else None
        for value, is_numbered, scope_id in zip(values, numbered, scope, strict=True)
    ]


def _mask_feature(
    rows: list[TableRow], layout: _Layout, numbered: list[bool], scope: list[int]
) -> list[bool | None]:
    """Parent pattern = the numeric-fill pattern of the first unnumbered row of
    a scope, but only when the next unnumbered row fills different columns."""

    if not layout.numeric_cols:
        return [None] * len(rows)
    masks = [
        tuple(
            bool(row.cells[col].strip()) if col < len(row.cells) else False
            for col in layout.numeric_cols
        )
        for row in rows
    ]
    by_scope: dict[int, list[int]] = {}
    for index, (is_numbered, scope_id) in enumerate(zip(numbered, scope, strict=True)):
        if not is_numbered:
            by_scope.setdefault(scope_id, []).append(index)
    result: list[bool | None] = [None] * len(rows)
    for indexes in by_scope.values():
        if len(indexes) < 2 or masks[indexes[0]] == masks[indexes[1]]:
            continue
        parent_mask = masks[indexes[0]]
        for index in indexes:
            result[index] = masks[index] == parent_mask
    return result


def _label_offset(row: TableRow, label_col: int) -> float | None:
    """How far the label text starts from the left edge of its OWN cell. Using
    the offset rather than the absolute x keeps a column that merely sits
    further right on part of a page from reading as a deeper level."""

    signals = row.signals
    if signals is None or label_col >= len(signals.cell_x0):
        return None
    x0 = signals.cell_x0[label_col]
    if x0 is None:
        return None
    left = signals.cell_left[label_col] if label_col < len(signals.cell_left) else None
    return x0 - (left or 0.0)


def _indent_ranks(rows: list[TableRow], label_col: int, tolerance: float) -> list[int | None]:
    x0s = [_label_offset(row, label_col) for row in rows]
    known = sorted(value for value in x0s if value is not None)
    clusters: list[float] = []
    for value in known:
        if not clusters or value - clusters[-1] > tolerance:
            clusters.append(value)
    if len(clusters) < 2:
        return [None] * len(rows)
    return [
        None if value is None else min(range(len(clusters)), key=lambda k: abs(clusters[k] - value))
        for value in x0s
    ]


def _geometry_features(
    rows: list[TableRow],
    label_col: int,
    numbered: list[bool],
    scope: list[int],
    scoring: HierarchyScoring,
) -> tuple[list[bool | None], list[bool | None], list[bool | None]]:
    """span, font and height classes (True = looks like a parent row)."""

    n = len(rows)
    span: list[bool | None] = [None] * n
    font: list[bool | None] = [None] * n
    height: list[bool | None] = [None] * n
    sizes: list[float] = []
    for row in rows:
        if row.signals is not None and label_col < len(row.signals.cell_size):
            size = row.signals.cell_size[label_col]
            if size:
                sizes.append(size)
    median_size = statistics.median(sizes) if sizes else None
    heights = [
        row.signals.height for row in rows if row.signals is not None and row.signals.height > 0
    ]
    median_height = statistics.median(heights) if heights else None

    for index, row in enumerate(rows):
        signals = row.signals
        if signals is None or label_col >= len(signals.cell_merged):
            continue
        span[index] = any(signals.cell_merged[label_col + 1 :])
        bold = signals.cell_bold[label_col] if label_col < len(signals.cell_bold) else False
        size = signals.cell_size[label_col] if label_col < len(signals.cell_size) else None
        larger = bool(median_size and size and size > median_size * scoring.font_size_ratio)
        font[index] = bold or larger
        if median_height:
            height[index] = signals.height > median_height * scoring.height_ratio

    unnumbered_total = sum(1 for is_numbered in numbered if not is_numbered)
    if unnumbered_total:
        tall = sum(
            1
            for value, is_numbered in zip(height, numbered, strict=True)
            if value and not is_numbered
        )
        if tall / unnumbered_total > scoring.height_max_parent_share:
            height = [None] * n
    return (
        _binary_feature(span, numbered, scope),
        _binary_feature(font, numbered, scope),
        _binary_feature(height, numbered, scope),
    )


def _is_group_row(row: TableRow, layout: _Layout) -> bool:
    """A row that only names something: no cell besides its ordinal and label."""

    return not any(
        cell.strip()
        for column, cell in enumerate(row.cells)
        if column not in (layout.ordinal_col, layout.label_col)
    )


def _numbering(
    rows: list[TableRow], layout: _Layout, scoring: HierarchyScoring
) -> tuple[list[tuple[str, str] | None], list[str], list[int | None]]:
    """Per row: its ordinal (style, token), its label text, and the absolute depth
    the numbering scheme gives it (None for rows outside the scheme).

    Group rows (`layout.group_labels`) head the hierarchy exactly as numbered rows
    would - each carries an absolute depth and every other row hangs below the last
    one. Otherwise the scheme comes from an ordinal column or from an ordinal glued
    to the front of the label."""

    def label_of(row: TableRow) -> str:
        return _clean(row.cells[layout.label_col]) if layout.label_col < len(row.cells) else ""

    if layout.group_labels:
        labels = [layout.group_labels.get(i) or label_of(row) for i, row in enumerate(rows)]
        return [None] * len(rows), labels, _group_depths(layout.group_labels, len(rows))

    parsed: list[tuple[str, str] | None] = []
    labels = []
    span_origin_is_group = False
    for row in rows:
        if layout.ordinal_col is not None and layout.ordinal_col < len(row.cells):
            if layout.ordinal_col not in row.inherited_cells:
                span_origin_is_group = _is_group_row(row, layout)
                parsed.append(_parse_ordinal(row.cells[layout.ordinal_col]))
            elif span_origin_is_group:
                # the ordinal was merged down from a row that only names a
                # group: the rows it covers are that group's children
                parsed.append(None)
            else:
                # merged down from a row holding values: the covered rows are
                # further items under the same number (siblings)
                parsed.append(_parse_ordinal(row.cells[layout.ordinal_col]))
            labels.append(label_of(row))
        else:
            item, rest = _split_prefix(label_of(row))
            parsed.append(item)
            labels.append(rest)
    if layout.ordinal_col is None:
        found = sum(1 for item in parsed if item is not None)
        if (
            found < scoring.ordinal_prefix_min_count
            or found / len(rows) < scoring.ordinal_prefix_min_share
        ):
            return [None] * len(rows), [label_of(row) for row in rows], [None] * len(rows)
    numbered_any = any(item is not None for item in parsed)
    return parsed, labels, _ordinal_depths(parsed) if numbered_any else [None] * len(rows)


def infer_hierarchy(table: TableBlock, scoring: HierarchyScoring = HIERARCHY_SCORING) -> TableBlock:
    rows = table.rows
    if len(rows) < 2 or not table.header_row:
        return table
    column_count = len(table.header_row)
    layout = _detect_layout(rows, column_count, scoring)
    if layout is None:
        return table

    parsed, labels, ordinal_depth = _numbering(rows, layout, scoring)
    numbering_active = any(depth is not None for depth in ordinal_depth)
    numbered = [depth is not None for depth in ordinal_depth]

    scope: list[int] = []
    current_scope = 0
    for depth in ordinal_depth:
        if depth is not None:
            current_scope += 1
        scope.append(current_scope)

    if layout.group_labels:
        # The group rows already say where every item belongs. The signals that
        # INFER parents (fill pattern, font, indentation) would only add false
        # ones here - an item with one empty cell looks like a "different" row.
        none: list[bool | None] = [None] * len(rows)
        mask = span = font = height = none
        ranks: list[int | None] = [None] * len(rows)
    else:
        mask = _mask_feature(rows, layout, numbered, scope)
        span, font, height = _geometry_features(rows, layout.label_col, numbered, scope, scoring)
        ranks = _indent_ranks(rows, layout.label_col, scoring.indent_cluster_tolerance)
    binary_signals = (
        ("span", span, scoring.weight_span),
        ("font", font, scoring.weight_font),
        ("mask", mask, scoring.weight_mask),
        ("height", height, scoring.weight_height),
    )

    def display(index: int) -> str:
        item = parsed[index]
        token = item[1] if item is not None else ""
        return " ".join(part for part in (token, labels[index]) if part)

    new_rows: list[TableRow] = []
    stack: list[tuple[int, str]] = []
    depth_previous = 0
    context_depth = -1
    last_true: dict[str, int] = {}
    last_by_rank: dict[int, int] = {}

    for index, row in enumerate(rows):
        warnings = list(row.warnings)
        confidence = row.confidence
        ancestors: list[str] = []
        decided_depth: int | None = None

        if index == 0:
            decided_depth = ordinal_depth[0] if ordinal_depth[0] is not None else 0
        else:
            tally: dict[int, float] = {}
            present = 0.0
            previous_is_ordinal = numbered[index - 1]
            if numbered[index]:
                target = ordinal_depth[index] or 0
                tally[target] = scoring.weight_numbering
                present = scoring.weight_numbering
            else:
                # A row whose text the PDF failed to encode has no trustworthy
                # geometry either (its label starts wherever the junk starts).
                trusted_geometry = _GARBLED_WARNING not in warnings
                rank_c, rank_p = ranks[index], ranks[index - 1]
                if trusted_geometry and rank_c is not None and rank_p is not None:
                    if rank_c > rank_p:
                        vote = depth_previous + 1
                    elif rank_c == rank_p:
                        vote = depth_previous
                    else:
                        vote = last_by_rank.get(rank_c, context_depth + 1)
                    tally[vote] = tally.get(vote, 0.0) + scoring.weight_indent
                    present += scoring.weight_indent
                for name, feature, weight in binary_signals:
                    class_c = feature[index]
                    if class_c is None:
                        continue
                    if not trusted_geometry and name in _GEOMETRY_SIGNALS:
                        continue
                    class_p = feature[index - 1]
                    if class_p is None:
                        vote = depth_previous + 1
                    elif class_c:
                        vote = depth_previous if class_p else last_true.get(name, context_depth + 1)
                    else:
                        vote = depth_previous + 1 if class_p else depth_previous
                    tally[vote] = tally.get(vote, 0.0) + weight
                    present += weight
                if numbering_active:
                    floor = context_depth + 1
                    if tally:
                        for depth in list(tally):
                            if depth >= floor:
                                tally[depth] += scoring.weight_numbering
                    else:
                        default = depth_previous + 1 if previous_is_ordinal else depth_previous
                        tally[max(default, floor)] = scoring.weight_numbering
                    present += scoring.weight_numbering

            if present >= scoring.min_present_weight and tally:
                ranked = sorted(tally.items(), key=lambda item: (-item[1], item[0]))
                best_depth, best_weight = ranked[0]
                second_weight = ranked[1][1] if len(ranked) > 1 else 0.0
                share = best_weight / present
                gap = (best_weight - second_weight) / present
                if share >= scoring.assign_threshold and gap >= scoring.tie_gap:
                    decided_depth = best_depth
                    confidence = min(confidence, share)
                else:
                    confidence = min(confidence, share)
                    warnings.append("hierarchy_uncertain")

        if decided_depth is not None:
            if index > 0 and decided_depth > depth_previous + 1:
                decided_depth = depth_previous + 1
                warnings.append("depth_clamped")
            decided_depth = max(decided_depth, 0)
            while stack and stack[-1][0] >= decided_depth:
                stack.pop()
            ancestors = [label for _depth, label in stack if label]
            stack.append((decided_depth, display(index)))
            depth_previous = decided_depth
            if numbered[index]:
                context_depth = decided_depth
                last_true = {}
                last_by_rank = {}
            if ranks[index] is not None:
                last_by_rank[ranks[index]] = decided_depth  # type: ignore[index]
            for name, feature, _weight in binary_signals:
                if feature[index]:
                    last_true[name] = decided_depth

        new_rows.append(replace(row, ancestors=ancestors, confidence=confidence, warnings=warnings))

    return replace(table, rows=new_rows)
