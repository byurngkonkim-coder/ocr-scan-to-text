"""로컬 OCR 은 RapidOCR 한글 v5 하나만 쓴다 — Tesseract 로 조용히 넘어가지 않는다.

배경: RapidOCR 오류·미설치 때 Tesseract 로 넘어가 한글 품질이 크게 떨어진 결과를 '완료'로 냈고,
smart_run 은 한국어 쪽을 언어만 바꿔 두 번 읽으며 모델을 쪽마다 다시 불러왔다(2026-10-13 예측기계).
"""
import pytest
from PIL import Image

from core import blank_detector, rapidocr_engine, tesseract_engine
from core.types import OCREngine, OCRResult


def _ok(text="본문 글자입니다"):
    return OCRResult(text=text, confidence=95.0, engine=OCREngine.RAPIDOCR, has_jamo_split=False)


def test_rapidocr_error_is_not_hidden_by_tesseract(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("onnx 오류")
    monkeypatch.setattr(rapidocr_engine, "run", boom)
    with pytest.raises(RuntimeError, match="onnx 오류"):
        tesseract_engine.run(Image.new("RGB", (50, 50), "white"))


def test_missing_rapidocr_is_reported(monkeypatch):
    monkeypatch.setattr(rapidocr_engine, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="RapidOCR"):
        tesseract_engine.run(Image.new("RGB", (50, 50), "white"))


def test_korean_page_read_once(monkeypatch):
    calls = []
    monkeypatch.setattr(rapidocr_engine, "run", lambda img, lang="kor+eng", fast=False: calls.append(lang) or _ok())
    tesseract_engine.smart_run(Image.new("RGB", (50, 50), "white"))
    assert calls == ["kor+eng"]


def test_blank_check_counts_with_rapidocr(monkeypatch):
    monkeypatch.setattr(rapidocr_engine, "read_boxes",
                        lambda img, lang="kor+eng": [("가나다 abc", 0.9, (0, 0, 10, 10)), ("·", 0.5, (0, 0, 1, 1))])
    assert blank_detector._quick_text_count(Image.new("RGB", (3000, 2000), "white")) == 6
