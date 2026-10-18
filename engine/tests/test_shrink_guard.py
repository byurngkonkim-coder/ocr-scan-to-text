"""내장 규칙 정제가 본문을 크게 지웠을 때 정제본을 버리고 원문으로 저장하는지.

LLM 정제에는 분량 가드가 있지만 내장 규칙 정제에는 없어, 규칙 회귀로 한글 1,368자가 0자가
돼도 '완료'로 저장됐다(감사 2026-10-18, 합성 데이터).
"""
from types import SimpleNamespace

import pytest

import ocr_gradio
from core import storage_manager, text_cleaner
from core.types import OCREngine, PageResult


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr_gradio, "RESULTS_DIR", tmp_path / "results")
    base = tmp_path / "storage"
    monkeypatch.setattr(storage_manager, "_BASE", base)
    monkeypatch.setattr(storage_manager, "_UPLOADS", base / "uploads")
    monkeypatch.setattr(storage_manager, "_RESULTS", base / "results")
    monkeypatch.setattr(storage_manager, "_HISTORY", base / "history.json")
    return tmp_path / "results"


BODY = "경영은 사람을 통해 일을 이루는 것이다. 목표를 세우고 조직하고 동기를 부여한다. " * 8


def _run(src, monkeypatch, cleaner):
    monkeypatch.setattr(ocr_gradio, "_process_page", lambda image, n, *a, **k: (
        PageResult(page_num=n, raw_text=BODY, corrected_text=BODY, engine=OCREngine.RAPIDOCR,
                   confidence=95.0), []))
    monkeypatch.setattr(text_cleaner, "iter_clean", cleaner)
    last = None
    for last in ocr_gradio.process_full(
            SimpleNamespace(name=str(src)), "kor+eng", "txt", False, False, True, False, 0, "",
            "RapidOCR 우선", cleanup=True, front_matter=False, progress=lambda *a, **k: None):
        pass
    return last


def _cleaner(out):
    def gen(pages, **kw):
        yield 1, 1
        return out
    return gen


def test_big_shrink_saves_original(sandbox, monkeypatch, tmp_path):
    img = tmp_path / "a.png"
    from PIL import Image
    Image.new("RGB", (50, 50), "white").save(img)
    last = _run(img, monkeypatch, _cleaner("1. 2. 3."))
    text = ocr_gradio.Path(last[3]).read_text(encoding="utf-8")
    assert "경영은 사람을" in text
    assert "분량 급감" in str(last[1])


def test_normal_clean_kept(sandbox, monkeypatch, tmp_path):
    img = tmp_path / "a.png"
    from PIL import Image
    Image.new("RGB", (50, 50), "white").save(img)
    last = _run(img, monkeypatch, _cleaner(BODY[:-20]))
    assert "분량 급감" not in str(last[1])
