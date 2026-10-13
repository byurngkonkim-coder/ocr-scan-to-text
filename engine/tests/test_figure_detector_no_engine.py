# Regression: ISSUE-001 — 글자 판별 엔진 없는 PC 에서 스캔본 「전체 처리」가 쪽마다
#   처리 오류로 빈 결과를 내던 문제
# Found by /qa on 2026-10-13
# Report: .gstack/qa-reports/run-20261013T070910Z/qa-report-ocr-dashboard-2026-10-13.md
"""RapidOCR 을 쓸 수 없는 PC 에서 그림·표 감지가 쪽 전체를 실패시키지 않는지.

그림 감지는 후보 안의 글자를 RapidOCR 로 확인하므로, 엔진이 없으면 감지를 건너뛰고
(마스킹 없음 → 본문 보존) 원본과 같은 이미지를 돌려줘야 한다.
"""
from PIL import Image, ImageChops, ImageDraw

from core import figure_detector, rapidocr_engine


def test_detect_without_engine_keeps_page(monkeypatch):
    monkeypatch.setattr(rapidocr_engine, "is_available", lambda: False)
    img = Image.new("RGB", (800, 400), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((100, 100, 400, 300), fill="steelblue")  # 그림 후보
    draw.text((450, 150), "본문 글자", fill="black")

    result = figure_detector.detect(img)

    assert result.regions == []
    assert ImageChops.difference(result.masked_image, img).getbbox() is None
