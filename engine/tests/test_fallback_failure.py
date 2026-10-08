"""Tesseract 우선 모드에서 Vision 폴백이 실패해도 작업이 중단되지 않아야 한다.

폴백 엔진(Gemini·Claude·Ollama)은 비정상 종료·빈 응답·연결 실패 시 예외를 던진다.
호출부가 받지 않으면 한 쪽 실패로 파일 전체 변환이 멈춘다 → Tesseract 결과 + 검토 필요로 이어간다.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import blank_detector, quality_diagnoser, spellcheck, tesseract_engine, vision
from core.types import OCREngine, OCRResult, QualityReport

IMG = Image.new("RGB", (800, 1000), "white")


def _boom(*a, **k):
    raise RuntimeError("Ollama 비정상 종료 done_reason=length")


def _vision_ok(*a, **k):
    return OCRResult(text="비전 본문", confidence=95.0, engine=OCREngine.OLLAMA)


def _setup(monkeypatch, blurry, conf, vision_run):
    monkeypatch.setattr(vision, "is_available", lambda: True)
    monkeypatch.setattr(vision, "run", vision_run)
    monkeypatch.setattr(spellcheck, "is_available", lambda: False)
    monkeypatch.setattr(blank_detector, "check", lambda *a, **k: SimpleNamespace(is_blank=False, reason=""))
    monkeypatch.setattr(tesseract_engine, "smart_run",
                        lambda *a, **k: OCRResult(text="테서랙트 본문", confidence=conf, engine=OCREngine.TESSERACT))
    monkeypatch.setattr(quality_diagnoser, "diagnose", lambda img: QualityReport(
        blur_score=50 if blurry else 500, avg_brightness=128, skew_angle=0,
        is_vertical_text=False, width_px=800, height_px=1000))


def _gui(monkeypatch, blurry, conf, vision_run):
    ocr_gradio = pytest.importorskip("ocr_gradio")
    _setup(monkeypatch, blurry, conf, vision_run)
    return ocr_gradio._process_page(IMG, 1, "kor+eng", no_preprocess=True, no_fallback=False,
                                    no_figure=True, llm_correct=False, llm_budget=0)[0]


def _cli(monkeypatch, blurry, conf, vision_run):
    ocr_tool = pytest.importorskip("ocr_tool")
    _setup(monkeypatch, blurry, conf, vision_run)
    return ocr_tool._process_page(IMG, 1, "kor+eng", no_preprocess=True, no_fallback=False,
                                  no_spellcheck=True, no_figure=True, save_figures=False, output_dir=".",
                                  stem="t", llm_correct=False, llm_budget=0, verbose=False)


# (품질불량?, Tesseract 신뢰도) — 저신뢰 폴백 / 품질불량 즉시폴백+저신뢰 / 품질불량 즉시폴백+고신뢰
CASES = [(False, 40.0), (True, 40.0), (True, 90.0)]


@pytest.mark.parametrize("run", [_gui, _cli])
@pytest.mark.parametrize("blurry,conf", CASES)
def test_vision_failure_keeps_tesseract(monkeypatch, run, blurry, conf):
    page = run(monkeypatch, blurry, conf, _boom)
    assert page.engine == OCREngine.TESSERACT and "테서랙트" in page.raw_text
    assert page.needs_review and "폴백 실패" in page.review_reason


@pytest.mark.parametrize("run", [_gui, _cli])
@pytest.mark.parametrize("blurry,conf", CASES[:2])
def test_vision_success_is_used(monkeypatch, run, blurry, conf):
    page = run(monkeypatch, blurry, conf, _vision_ok)
    assert page.engine == OCREngine.OLLAMA and "비전" in page.raw_text
    assert not page.needs_review
