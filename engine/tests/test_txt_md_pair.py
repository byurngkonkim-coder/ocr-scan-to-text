"""저장 형식 txt+md — 한 번 처리로 같은 이름의 .txt 와 .md 짝 (2026-10-19)."""
from pathlib import Path

import ocr_gradio
from core.types import FileResult, OCREngine, OutputFormat, PageResult


def test_parse_fmt():
    assert ocr_gradio._parse_fmt("txt") == (OutputFormat.TXT, False)
    assert ocr_gradio._parse_fmt("md") == (OutputFormat.MD, False)
    assert ocr_gradio._parse_fmt("txt+md") == (OutputFormat.TXT, True)   # .txt 가 주 결과(처리 기록)


def _result(tmp_path) -> FileResult:
    res = FileResult(source_path=tmp_path / "보고서.pdf", output_path=None)
    res.pages.append(PageResult(page_num=1, raw_text="본문", corrected_text="본문 첫 쪽",
                                engine=OCREngine.RAPIDOCR, confidence=95.0))
    return res


def test_cleaned_pair_is_byte_identical(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr_gradio, "RESULTS_DIR", tmp_path / "02_결과")
    res = _result(tmp_path)
    out = ocr_gradio.writer.write_plain(res.source_path, "정제 본문\n\n| a | b |", tmp_path / "t", OutputFormat.TXT)
    md_src = ocr_gradio._md_pair_source(res.source_path, res, tmp_path / "t", "정제 본문\n\n| a | b |")
    final = ocr_gradio._finalize_results(None, None, out, "보고서", [], md_src)
    assert final.suffix == ".txt"
    assert final.with_suffix(".md").read_bytes() == final.read_bytes()
    assert ocr_gradio._result_files(final) == [str(final), str(final.with_suffix(".md"))]


def test_uncleaned_pair_keeps_md_layout(tmp_path):
    # 정제를 끈 쪽 단위 출력: .md 는 기존 MD 서식(## 페이지 N)
    res = _result(tmp_path)
    md = ocr_gradio._md_pair_source(res.source_path, res, tmp_path).read_text(encoding="utf-8")
    assert "## 페이지 1" in md and "본문 첫 쪽" in md


def test_single_format_has_no_pair(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr_gradio, "RESULTS_DIR", tmp_path / "02_결과")
    res = _result(tmp_path)
    out = ocr_gradio.writer.write_plain(res.source_path, "본문", tmp_path / "t", OutputFormat.TXT)
    final = ocr_gradio._finalize_results(None, None, out, "보고서", [])
    assert not Path(final).with_suffix(".md").exists()
    assert ocr_gradio._result_files(final) == str(final)
