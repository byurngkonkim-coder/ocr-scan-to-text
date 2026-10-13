# Regression: ISSUE-001 — Tesseract 미설치 PC 에서 스캔본 「전체 처리」가 쪽마다
#   'tesseract is not installed' 처리 오류로 빈 결과를 내던 문제
# Found by /qa on 2026-10-13
# Report: .gstack/qa-reports/run-20261013T070910Z/qa-report-ocr-dashboard-2026-10-13.md
"""Tesseract 가 없는 PC 에서 그림·표 감지가 쪽 전체를 실패시키지 않는지.

로컬 엔진은 RapidOCR 이라 Tesseract 는 선택 설치다. 그림 감지만 Tesseract 로 글자를
확인하므로, 없으면 감지를 건너뛰고(마스킹 없음 → 본문 보존) 원본 이미지를 돌려줘야 한다.
"""
import pytesseract
from PIL import Image, ImageDraw

from core import figure_detector


def _no_tesseract(*_a, **_k):
    raise pytesseract.TesseractNotFoundError()


def test_detect_without_tesseract_keeps_page(monkeypatch):
    monkeypatch.setattr(pytesseract, "get_tesseract_version", _no_tesseract)
    monkeypatch.setattr(pytesseract, "image_to_data", _no_tesseract)
    monkeypatch.setattr(pytesseract, "image_to_string", _no_tesseract)
    img = Image.new("RGB", (800, 400), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((100, 100, 400, 300), fill="steelblue")  # 그림 후보
    draw.text((450, 150), "본문 글자", fill="black")

    result = figure_detector.detect(img)

    assert result.regions == []
    assert result.masked_image is img
