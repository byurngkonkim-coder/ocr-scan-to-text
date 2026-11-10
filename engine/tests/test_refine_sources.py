"""정제 함수 하나(_refine)를 스캔·디지털 PDF·txt 가 함께 쓴다 — 입력 종류에 따라 달라지는 것만 갈린다.

Intent: docs/intents/2026-11-10-정제-후처리-통합.md (정제 → 저장 흐름 3벌을 1벌로)
"""
import ocr_gradio
from core import korean_corrector, text_cleaner
from core.types import OCREngine, OutputFormat, PageResult


def _pages(engine=OCREngine.PDF_TEXT):
    texts = ["경영은 사람을 통해 성과를 내는 일이다. 그래서 경영자는 먼저 사람을 본다.",
             "성과는 조직 밖에 있다. 조직 안에는 비용만 있을 뿐이다."]
    return [PageResult(page_num=n, raw_text=t, corrected_text=t, engine=engine, confidence=100.0)
            for n, t in enumerate(texts, 1)]


def _drain(gen):
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        return stop.value


def test_digital_pages_get_korean_correction_once_and_raw_lines_for_verse_books(monkeypatch):
    """디지털 PDF 의 내장 글자는 읽을 때 한글 교정기를 거치지 않았다 → 정제 직전에 1번. 장절 책(성경)을 위해
    줄 잇기 전 원문 줄도 넘긴다."""
    corrected = []
    monkeypatch.setattr(korean_corrector, "refine_korean_text", lambda t: corrected.append(t) or t)
    seen = {}
    real = text_cleaner.iter_clean

    def spy(pages, **k):
        seen.update(raw_pages=k.get("raw_pages"), local=k.get("local"))
        return real(pages, **k)
    monkeypatch.setattr(text_cleaner, "iter_clean", spy)
    text, md, _crep, _note = _drain(ocr_gradio._refine(_pages(), False, OutputFormat.MD, [],
                                                        {"name": "책.pdf", "pages": 2}, source="digital"))
    assert len(corrected) == 2
    assert seen == {"raw_pages": [p.raw_text for p in _pages()], "local": True}
    assert "[p.2]" in md and "PDF 2쪽" in md


def test_scan_pages_are_not_corrected_again(monkeypatch):
    corrected = []
    monkeypatch.setattr(korean_corrector, "refine_korean_text", lambda t: corrected.append(t) or t)
    seen = {}
    real = text_cleaner.iter_clean

    def spy(pages, **k):
        seen.update(raw_pages=k.get("raw_pages"))
        return real(pages, **k)
    monkeypatch.setattr(text_cleaner, "iter_clean", spy)
    _drain(ocr_gradio._refine(_pages(OCREngine.RAPIDOCR), False, OutputFormat.MD, [], {"name": "책.pdf", "pages": 2}))
    assert corrected == [] and seen == {"raw_pages": None}


def test_txt_input_gets_no_page_marks():
    """txt 입력의 '쪽'은 문단 묶음일 뿐이라 쪽 표지를 넣지 않는다."""
    logs: list[str] = []
    text, md, _crep, _note = _drain(ocr_gradio._refine(_pages(), False, OutputFormat.MD, logs,
                                                        {"name": "책.txt", "pages": None}, source="txt"))
    assert "[p." not in md and "쪽 표지" not in md
    assert "경영은 사람을 통해" in text
    assert any("문장 정제 완료" in line for line in logs)
