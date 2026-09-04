"""Reading a drawn table as a table.

Built from DXF fragments rather than from the reference drawing, because the
module must work on a file nobody here has seen. The two facts taken from the
real file are the shape of a cell value -- MTEXT formatting wrapped around a
number, with the backslash escaped -- and the fact that its schedule has no
header row at all, which is why no rule here may depend on one.
"""

from __future__ import annotations

from app import drawn_tables as tbl


def _dxf(entities: list[tuple[str, list[tuple[str, str]]]]) -> str:
    out = ["0", "SECTION", "2", "ENTITIES"]
    for kind, pairs in entities:
        out += ["0", kind]
        for code, value in pairs:
            out += [code, value]
    out += ["0", "ENDSEC", "0", "EOF"]
    return "\n".join(out) + "\n"


def _table(handle: str, rows: int, cols: int, cells: list[str], layer: str = "0"):
    pairs = [("5", handle), ("100", "AcDbEntity"), ("8", layer)]
    pairs += [("100", "AcDbTable"), ("91", str(rows)), ("92", str(cols))]
    for text in cells:
        pairs += [("301", "CELL_VALUE"), ("302", text)]
    return ("ACAD_TABLE", pairs)


# --- the cell value ----------------------------------------------------------


def test_formatting_is_stripped_from_a_cell():
    """A cell arrives as `{\\C7;20479}`: the number in colour 7."""
    assert tbl.clean_cell("{\\C7;20479}") == "20479"


def test_an_escaped_backslash_is_stripped_too():
    """DXF escapes the backslash, so one pass leaves `\\C7;` behind."""
    assert tbl.clean_cell("{\\\\C7;3401}") == "3401"


def test_a_plain_cell_survives_untouched():
    assert tbl.clean_cell("Plot Area") == "Plot Area"


def test_a_paragraph_break_becomes_a_space():
    assert tbl.clean_cell("first\\Psecond") == "first second"


# --- reading the file --------------------------------------------------------


def test_a_table_is_read_with_its_shape(tmp_path):
    path = tmp_path / "t.dxf"
    path.write_text(
        _dxf([_table("A1", 2, 2, ["10", "1", "20", "2"])]), encoding="utf-8"
    )
    found = tbl.read(path)
    assert len(found) == 1
    assert (found[0].handle, found[0].rows, found[0].cols) == ("A1", 2, 2)
    assert found[0].grid == [["10", "1"], ["20", "2"]]


def test_a_numeric_cell_value_is_not_mistaken_for_a_group_code(tmp_path):
    """The bug that read sixty tables as sixty empty ones.

    A parser that decides "the previous line was a code because it looked
    numeric" reads the value `58` as a code, and then reads the layer name
    `0` as the code that opens a new entity -- closing the table two lines
    after it opened. Codes and values strictly alternate.
    """
    path = tmp_path / "numeric.dxf"
    path.write_text(
        _dxf(
            [
                (
                    "ACAD_TABLE",
                    [
                        ("5", "B1"),
                        ("330", "58"),  # an owner handle that is all digits
                        ("100", "AcDbEntity"),
                        ("8", "0"),  # a layer named "0"
                        ("100", "AcDbTable"),
                        ("91", "1"),
                        ("92", "2"),
                        ("302", "{\\C7;300}"),
                        ("302", "{\\C7;7}"),
                    ],
                )
            ]
        ),
        encoding="utf-8",
    )
    found = tbl.read(path)
    assert len(found) == 1
    assert found[0].cells == ["300", "7"]
    assert found[0].cols == 2


def test_rows_and_columns_come_from_the_table_subclass_only(tmp_path):
    """Codes 91 and 92 mean other things elsewhere in the same entity."""
    path = tmp_path / "sub.dxf"
    path.write_text(
        _dxf(
            [
                (
                    "ACAD_TABLE",
                    [
                        ("5", "C1"),
                        ("100", "AcDbEntity"),
                        ("91", "999"),  # not the table's row count
                        ("100", "AcDbTable"),
                        ("91", "3"),
                        ("92", "1"),
                        ("302", "a"),
                        ("302", "b"),
                        ("302", "c"),
                    ],
                )
            ]
        ),
        encoding="utf-8",
    )
    assert tbl.read(path)[0].rows == 3


def test_several_tables_keep_their_own_cells(tmp_path):
    path = tmp_path / "many.dxf"
    path.write_text(
        _dxf(
            [
                _table("A", 1, 2, ["1", "2"]),
                _table("B", 1, 2, ["3", "4"]),
            ]
        ),
        encoding="utf-8",
    )
    found = tbl.read(path)
    assert [t.handle for t in found] == ["A", "B"]
    assert [t.cells for t in found] == [["1", "2"], ["3", "4"]]


def test_a_table_belongs_to_the_block_it_is_drawn_in(tmp_path):
    path = tmp_path / "block.dxf"
    body = ["0", "SECTION", "2", "BLOCKS", "0", "BLOCK", "2", "*Paper_Space73"]
    body += ["0", "ACAD_TABLE", "5", "T1", "100", "AcDbTable", "91", "1", "92", "1"]
    body += ["302", "x", "0", "ENDBLK", "0", "ENDSEC", "0", "EOF"]
    path.write_text("\n".join(body) + "\n", encoding="utf-8")
    assert tbl.read(path)[0].block == "*Paper_Space73"


def test_a_file_with_no_tables_yields_none(tmp_path):
    path = tmp_path / "plain.dxf"
    path.write_text(_dxf([("LINE", [("5", "1"), ("8", "0")])]), encoding="utf-8")
    assert tbl.read(path) == []


def test_the_symbol_table_section_is_not_read_as_data(tmp_path):
    """`TABLE` in DXF opens the layer dictionary, not a drawn table."""
    path = tmp_path / "symbols.dxf"
    body = ["0", "SECTION", "2", "TABLES", "0", "TABLE", "2", "LAYER"]
    body += ["0", "LAYER", "2", "VL2", "0", "ENDTAB", "0", "ENDSEC", "0", "EOF"]
    path.write_text("\n".join(body) + "\n", encoding="utf-8")
    assert tbl.read(path) == []
    assert "TABLE" not in tbl.TABLE_TYPES


def test_a_missing_column_count_does_not_invent_a_shape(tmp_path):
    """A wrong width silently pairs the wrong values together."""
    path = tmp_path / "noshape.dxf"
    path.write_text(
        _dxf(
            [
                (
                    "ACAD_TABLE",
                    [("5", "D"), ("100", "AcDbTable"), ("302", "a"), ("302", "b")],
                )
            ]
        ),
        encoding="utf-8",
    )
    assert tbl.read(path)[0].grid == [["a", "b"]]


# --- reading a number --------------------------------------------------------


def test_numbers_a_draughtsman_types_are_read():
    assert tbl.as_number("300") == 300
    assert tbl.as_number("1,234") == 1234
    assert tbl.as_number("1,234.5") == 1234.5
    assert tbl.as_number("12.5 m2") == 12.5
    assert tbl.as_number("  42  ") == 42


def test_a_comma_decimal_is_not_read_as_thousands():
    assert tbl.as_number("12,5") == 12.5


def test_text_is_not_a_number():
    assert tbl.as_number("Plot Area") is None
    assert tbl.as_number("") is None
    assert tbl.as_number("-") is None


# --- what a column is --------------------------------------------------------


def _grid(pairs: list[tuple[str, str]]) -> list[list[str]]:
    return [list(pair) for pair in pairs]


def test_a_dense_run_of_distinct_integers_is_an_identifier():
    grid = _grid([(str(300 + i % 3), str(i)) for i in range(1, 60)])
    columns = tbl.describe_columns(grid)
    assert columns[1].kind == "identifier"
    assert columns[0].kind == "measure"


def test_a_repeated_value_is_not_an_identifier():
    grid = _grid([("5", "1"), ("5", "1"), ("5", "2")])
    assert tbl.describe_columns(grid)[1].kind == "measure"


def test_a_sparse_run_is_not_an_identifier():
    """1, 500, 9000 are distinct integers and are not a numbering."""
    grid = _grid([("5", "1"), ("6", "500"), ("7", "9000")])
    assert tbl.describe_columns(grid)[1].kind == "measure"


def test_fractional_values_are_never_an_identifier():
    grid = _grid([("1.5", "x"), ("2.5", "y"), ("3.5", "z")])
    assert tbl.describe_columns(grid)[0].kind == "measure"


def test_a_column_of_words_is_text():
    grid = _grid([("Villa", "1"), ("Mosque", "2"), ("School", "3")])
    columns = tbl.describe_columns(grid)
    assert columns[0].kind == "text"
    assert columns[0].numeric == 0


def test_every_column_says_why_it_was_called_that():
    grid = _grid([(str(i * 10), str(i)) for i in range(1, 30)])
    for column in tbl.describe_columns(grid):
        assert column.reason


def test_a_single_row_cannot_establish_a_numbering():
    """Two points make a run; one makes nothing."""
    assert tbl.describe_columns(_grid([("5", "1")]))[1].kind == "measure"


# --- merging the parts of one schedule ---------------------------------------


def test_tables_of_the_same_width_merge_in_order(tmp_path):
    path = tmp_path / "split.dxf"
    path.write_text(
        _dxf([_table("A", 1, 2, ["10", "1"]), _table("B", 1, 2, ["20", "2"])]),
        encoding="utf-8",
    )
    assert tbl.merge_grids(tbl.read(path)) == [["10", "1"], ["20", "2"]]


def test_a_table_of_a_different_width_is_left_out(tmp_path):
    """A different shape is a different thing, not a continuation."""
    path = tmp_path / "mixed.dxf"
    path.write_text(
        _dxf(
            [
                _table("A", 2, 2, ["10", "1", "20", "2"]),
                _table("B", 1, 3, ["x", "y", "z"]),
            ]
        ),
        encoding="utf-8",
    )
    merged = tbl.merge_grids(tbl.read(path))
    assert merged == [["10", "1"], ["20", "2"]]


def test_merging_nothing_is_not_an_error():
    assert tbl.merge_grids([]) == []
