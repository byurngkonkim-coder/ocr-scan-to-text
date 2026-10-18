"""디지털 PDF 경로에서 글자 층이 거의 없는 쪽(그림만 있는 쪽·장 제목 쪽)을 OCR 없이 버리던 것.

글자 층 있는 쪽이 절반만 넘으면 디지털 경로로 가서, 나머지 그림 쪽은 내용이 있어도 '텍스트 없는
페이지 제외'로 빠지고 '완료'가 됐다(감사 2026-10-18: 10쪽 중 그림 4쪽 누락, 같은 쪽 OCR 신뢰도 97%).
"""
from types import SimpleNamespace

import pytest

import ocr_gradio
from core import storage_manager
from core.types import OCREngine, PageResult


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr_gradio, "RESULTS_DIR", tmp_path / "results")
    base = tmp_path / "storage"
    monkeypatch.setattr(storage_manager, "_BASE", base)
    monkeypatch.setattr(storage_manager, "_UPLOADS", base / "uploads")
    monkeypatch.setattr(storage_manager, "_RESULTS", base / "results")
    monkeypatch.setattr(storage_manager, "_HISTORY", base / "history.json")


def _run(pages, read_page):
    last = None
    for last in ocr_gradio._run_digital_pdf(
            ocr_gradio.Path("책.pdf"), pages, ocr_gradio.OutputFormat.TXT, False, False, 0.0,
            read_page=read_page):
        pass
    return SimpleNamespace(summary=str(last[1]), text=ocr_gradio.Path(last[3]).read_text(encoding="utf-8"))


BODY = "경영은 사람을 통해 일을 이루는 것이다. 목표를 세우고 조직한다."


def _reader(texts):
    def read(page_num):
        t = texts.get(page_num, "")
        return PageResult(page_num=page_num, raw_text=t, corrected_text=t, engine=OCREngine.RAPIDOCR,
                          confidence=97.0, is_blank=not t), []
    return read


def test_image_only_page_is_read_by_ocr(sandbox):
    out = _run([BODY, "", BODY], _reader({2: "그림 속 표의 글자를 읽은 본문입니다."}))
    assert "그림 속 표의 글자" in out.text


def test_short_title_page_kept_when_ocr_finds_nothing(sandbox):
    out = _run([BODY, "제1장 경영의 본질", BODY], _reader({}))
    assert "제1장 경영의 본질" in out.text


def test_truly_blank_page_still_excluded(sandbox):
    out = _run([BODY, "", BODY], _reader({}))
    assert "텍스트 없는 페이지 1p 제외" in out.summary
