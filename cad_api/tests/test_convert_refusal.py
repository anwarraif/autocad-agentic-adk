"""What the pipeline says when a drawing cannot be converted.

These tests exist because of Sedra. `SEDRA 2A NHD-G 2.dwg` failed to decode
with LibreDWG and the recorded diagnosis was "the file is probably corrupt in
transfer" -- which was wrong. The file is healthy and ODA converts it in 39 s
(docs/SEDRA-READABILITY-REPORT.md). What made a wrong diagnosis possible is
that `_convert_with_oda` returned a bare `None` for every kind of failure, so
no message could say WHICH converter had failed, or why.

The contract asserted here is therefore not cosmetic. A refusal carries four
parts -- the file, every converter tried and what it said, and the next human
step -- and a half-written DXF is never accepted as a good one.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app import convert


COMPLETE_DXF = "  0\nSECTION\n  2\nHEADER\n  0\nENDSEC\n  0\nEOF\n"

#: The same file with its tail lost, which is what a converter killed midway
#: leaves behind. It parses as a valid prefix, which is exactly the danger:
#: nothing downstream can tell it from a drawing that is genuinely small.
TRUNCATED_DXF = "  0\nSECTION\n  2\nHEADER\n  0\nENDSEC\n"


def _only(name: str):
    """A `shutil.which` that finds one named binary and nothing else."""

    def which(binary: str) -> str | None:
        return f"/usr/bin/{binary}" if binary == name else None

    return which


def _both(binary: str) -> str:
    """A `shutil.which` that finds every binary asked for, xvfb-run included."""
    return f"/usr/bin/{binary}"


# ---------------------------------------------------------------------------
# The truncation rule itself
# ---------------------------------------------------------------------------


def test_a_complete_dxf_is_not_called_truncated(tmp_path: Path):
    target = tmp_path / "good.dxf"
    target.write_text(COMPLETE_DXF)
    assert convert._truncation_reason(target) is None


def test_a_dxf_without_its_eof_marker_is_truncated(tmp_path: Path):
    """The failure mode a size check alone can never catch."""
    target = tmp_path / "half.dxf"
    target.write_text(TRUNCATED_DXF)

    reason = convert._truncation_reason(target)

    assert reason is not None
    assert "truncated" in reason
    # The size is quoted so an operator can compare it with the sender's.
    assert str(len(TRUNCATED_DXF)) in reason


def test_an_empty_file_is_reported_as_empty_not_as_truncated(tmp_path: Path):
    target = tmp_path / "nothing.dxf"
    target.write_bytes(b"")
    assert convert._truncation_reason(target) == "the converter wrote an empty file"


def test_binary_dxf_is_exempt_from_the_eof_rule(tmp_path: Path):
    """Binary DXF carries no textual EOF; the rule would fail a good file."""
    target = tmp_path / "binary.dxf"
    target.write_bytes(b"AutoCAD Binary DXF\r\n\x1a\x00\x01\x02\x03")
    assert convert._truncation_reason(target) is None


def test_trailing_whitespace_after_eof_is_still_complete(tmp_path: Path):
    target = tmp_path / "padded.dxf"
    target.write_text(COMPLETE_DXF + "\n\n  \n")
    assert convert._truncation_reason(target) is None


# ---------------------------------------------------------------------------
# ODA now says why it failed
# ---------------------------------------------------------------------------


def test_oda_reports_a_missing_install_as_a_reason_not_as_silence(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(convert.shutil, "which", lambda _binary: None)

    path, reason = convert._convert_with_oda(tmp_path / "x.dwg", tmp_path)

    assert path is None
    assert reason is not None and "not installed" in reason


def test_oda_err_file_fails_the_conversion_and_deletes_the_partial_output(
    tmp_path: Path, monkeypatch
):
    """ODA reports a bad read by writing `.err` beside the output.

    The process still exits 0, so the exit code cannot be trusted. This
    direction (DWG -> DXF) never read that channel; only DXF -> DWG did.
    """
    out = tmp_path / "out"
    out.mkdir()
    dwg = tmp_path / "drawing.dwg"
    dwg.write_bytes(b"stand-in")
    monkeypatch.setattr(convert.shutil, "which", _only(convert.ODA))

    def fake_run(cmd, **kwargs):
        (out / "drawing.dxf").write_text(TRUNCATED_DXF)
        (out / "drawing.dxf.err").write_text("Invalid Data Section Page Map")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(convert.subprocess, "run", fake_run)

    path, reason = convert._convert_with_oda(dwg, out)

    assert path is None
    assert reason is not None and "Invalid Data Section Page Map" in reason
    # The half-file must not survive to be mistaken for a good conversion.
    assert not (out / "drawing.dxf").exists()
    assert not (out / "drawing.dxf.err").exists()


def test_oda_output_that_is_truncated_is_refused_not_returned(
    tmp_path: Path, monkeypatch
):
    out = tmp_path / "out"
    out.mkdir()
    dwg = tmp_path / "drawing.dwg"
    dwg.write_bytes(b"stand-in")
    monkeypatch.setattr(convert.shutil, "which", _only(convert.ODA))

    def fake_run(cmd, **kwargs):
        (out / "drawing.dxf").write_text(TRUNCATED_DXF)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(convert.subprocess, "run", fake_run)

    path, reason = convert._convert_with_oda(dwg, out)

    assert path is None
    assert reason is not None and "truncated" in reason
    assert not (out / "drawing.dxf").exists()


def test_a_stale_err_file_cannot_fail_a_good_conversion(tmp_path: Path, monkeypatch):
    """A leftover `.err` would otherwise be read as this run's verdict."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "drawing.dxf.err").write_text("a failure from an earlier run")
    dwg = tmp_path / "drawing.dwg"
    dwg.write_bytes(b"stand-in")
    monkeypatch.setattr(convert.shutil, "which", _only(convert.ODA))

    def fake_run(cmd, **kwargs):
        (out / "drawing.dxf").write_text(COMPLETE_DXF)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(convert.subprocess, "run", fake_run)

    path, reason = convert._convert_with_oda(dwg, out)

    assert reason is None
    assert path == out / "drawing.dxf"


def test_oda_timeout_is_named_rather_than_swallowed(tmp_path: Path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    dwg = tmp_path / "drawing.dwg"
    dwg.write_bytes(b"stand-in")
    monkeypatch.setattr(convert.shutil, "which", _only(convert.ODA))

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, convert.CONVERT_TIMEOUT_S)

    monkeypatch.setattr(convert.subprocess, "run", fake_run)

    path, reason = convert._convert_with_oda(dwg, out)

    assert path is None
    assert reason is not None and str(convert.CONVERT_TIMEOUT_S) in reason


# ---------------------------------------------------------------------------
# The refusal contract
# ---------------------------------------------------------------------------


def _assert_refusal_contract(message: str, filename: str) -> None:
    """All four parts, or the message is not doing its job."""
    assert filename in message, "the refusal must name the file"
    assert "ODA File Converter" in message, "it must name the preferred converter"
    assert "LibreDWG" in message, "it must name the fallback converter"
    assert "Next step:" in message, "it must say what a human does now"


def test_refusal_names_both_converters_when_both_fail(tmp_path: Path, monkeypatch):
    """The defect this phase exists for.

    An ODA failure used to be invisible: the message named only `dwg2dxf`, so
    an operator would go and reinstall the converter that was never at fault.
    """
    out = tmp_path / "out"
    dwg = tmp_path / "drawing.dwg"
    dwg.write_bytes(b"stand-in")
    monkeypatch.setattr(convert.shutil, "which", _both)

    def fake_run(cmd, **kwargs):
        if convert.DWG2DXF in cmd:
            return subprocess.CompletedProcess(
                cmd, 1, "", "bit_read_RC buffer overflow"
            )
        return subprocess.CompletedProcess(cmd, 0, "", "")  # ODA writes nothing

    monkeypatch.setattr(convert.subprocess, "run", fake_run)

    with pytest.raises(convert.ConversionError) as caught:
        convert.convert(dwg, out)

    message = str(caught.value)
    _assert_refusal_contract(message, "drawing.dwg")
    # Both accounts are actually present, not merely both names.
    assert "wrote no output" in message
    assert "bit_read_RC buffer overflow" in message


def test_refusal_when_no_converter_is_installed_still_names_both(
    tmp_path: Path, monkeypatch
):
    """`is_available()` is true if EITHER converter exists, so the old
    message -- which named only dwg2dxf -- could be actively misleading."""
    monkeypatch.setattr(convert.shutil, "which", lambda _binary: None)
    dwg = tmp_path / "drawing.dwg"
    dwg.write_bytes(b"stand-in")

    with pytest.raises(convert.ConversionError) as caught:
        convert.convert(dwg, tmp_path / "out")

    _assert_refusal_contract(str(caught.value), "drawing.dwg")


def test_libredwg_truncated_output_is_refused_with_both_accounts(
    tmp_path: Path, monkeypatch
):
    """A truncated DXF never passes, whichever converter produced it."""
    out = tmp_path / "out"
    dwg = tmp_path / "drawing.dwg"
    dwg.write_bytes(b"stand-in")
    monkeypatch.setattr(convert.shutil, "which", _both)

    def fake_run(cmd, **kwargs):
        if convert.DWG2DXF in cmd:
            out.mkdir(parents=True, exist_ok=True)
            (out / "drawing.dxf").write_text(TRUNCATED_DXF)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(convert.subprocess, "run", fake_run)

    with pytest.raises(convert.ConversionError) as caught:
        convert.convert(dwg, out)

    message = str(caught.value)
    _assert_refusal_contract(message, "drawing.dwg")
    assert "truncated" in message
    assert not (out / "drawing.dxf").exists()


def test_a_good_libredwg_conversion_still_succeeds(tmp_path: Path, monkeypatch):
    """The hardening must not cost the fallback path that already worked."""
    out = tmp_path / "out"
    dwg = tmp_path / "drawing.dwg"
    dwg.write_bytes(b"stand-in")
    monkeypatch.setattr(convert.shutil, "which", _both)

    def fake_run(cmd, **kwargs):
        if convert.DWG2DXF in cmd:
            out.mkdir(parents=True, exist_ok=True)
            (out / "drawing.dxf").write_text(COMPLETE_DXF)
            # LibreDWG exits non-zero on almost every file and still writes a
            # good DXF; the file is the test, not the return code.
            return subprocess.CompletedProcess(cmd, 1, "", "warnings")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(convert.subprocess, "run", fake_run)

    result = convert.convert(dwg, out)

    assert result.converter == "libredwg"
    assert result.dxf_path == out / "drawing.dxf"


def test_a_good_oda_conversion_is_preferred_and_reported_as_oda(
    tmp_path: Path, monkeypatch
):
    out = tmp_path / "out"
    dwg = tmp_path / "drawing.dwg"
    dwg.write_bytes(b"stand-in")
    monkeypatch.setattr(convert.shutil, "which", _both)

    def fake_run(cmd, **kwargs):
        if convert.ODA in cmd:
            out.mkdir(parents=True, exist_ok=True)
            (out / "drawing.dxf").write_text(COMPLETE_DXF)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(convert.subprocess, "run", fake_run)

    result = convert.convert(dwg, out)

    assert result.converter == "oda"
