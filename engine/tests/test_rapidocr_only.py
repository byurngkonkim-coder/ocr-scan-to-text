"""로컬 OCR 은 RapidOCR 한글 v5 하나만 쓴다 — Tesseract 경로는 없다.

배경: RapidOCR 오류·미설치 때 Tesseract 로 조용히 넘어가 한글 품질이 크게 떨어진 결과를 '완료'로 냈고,
smart_run 은 한국어 쪽을 언어만 바꿔 두 번 읽으며 모델을 쪽마다 다시 불러왔다(2026-10-13 예측기계).
RapidOCR 이 실패해도 처리를 멈추지 않는다 — 신뢰도 0 결과로 돌려 Vision 폴백·검토 필요로 잇는다.
"""
from pathlib import Path

from PIL import Image

from core import blank_detector, rapidocr_engine
from core.types import OCREngine, OCRResult


def _ok(text="본문 글자입니다"):
    return OCRResult(text=text, confidence=95.0, engine=OCREngine.RAPIDOCR, has_jamo_split=False)


def test_rapidocr_error_becomes_zero_confidence_result(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("onnx 오류")
    monkeypatch.setattr(rapidocr_engine, "run", boom)
    res = rapidocr_engine.smart_run(Image.new("RGB", (50, 50), "white"))
    assert res.text == "" and res.confidence == 0.0 and res.engine == OCREngine.RAPIDOCR


def test_korean_page_read_once(monkeypatch):
    calls = []
    monkeypatch.setattr(rapidocr_engine, "run",
                        lambda img, lang="kor+eng", fast=False, refine=True: calls.append(lang) or _ok())
    rapidocr_engine.smart_run(Image.new("RGB", (50, 50), "white"))
    assert calls == ["kor+eng"]


def test_english_page_reread_with_english_model(monkeypatch):
    calls = []
    monkeypatch.setattr(rapidocr_engine, "run",
                        lambda img, lang="kor+eng", fast=False, refine=True:
                        calls.append(lang) or _ok("This page is almost entirely English text"))
    rapidocr_engine.smart_run(Image.new("RGB", (50, 50), "white"))
    assert calls == ["kor+eng", "eng"]


def test_blank_check_counts_with_rapidocr(monkeypatch):
    monkeypatch.setattr(rapidocr_engine, "read_boxes",
                        lambda img, lang="kor+eng": [("가나다 abc", 0.9, (0, 0, 10, 10)), ("·", 0.5, (0, 0, 1, 1))])
    assert blank_detector._quick_text_count(Image.new("RGB", (3000, 2000), "white")) == 6


def test_no_tesseract_left_in_engine_code():
    engine = Path(__file__).resolve().parents[1]
    hits = [str(p.relative_to(engine)) for p in engine.rglob("*.py")
            if "venv" not in p.parts and "tests" not in p.parts
            and "pytesseract" in p.read_text(encoding="utf-8", errors="ignore")]
    assert hits == []
    assert not (engine / "core" / "tesseract_engine.py").exists()
    assert not hasattr(OCREngine, "TESSERACT")
