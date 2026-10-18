"""RapidOCR 우선 모드에서 Vision 폴백이 실패해도 작업이 중단되지 않아야 한다.

폴백 엔진(Gemini·Claude·Ollama)은 비정상 종료·빈 응답·연결 실패 시 예외를 던진다.
호출부가 받지 않으면 한 쪽 실패로 파일 전체 변환이 멈춘다 → RapidOCR 결과 + 검토 필요로 이어간다.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import blank_detector, quality_diagnoser, spellcheck, rapidocr_engine, vision
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
    monkeypatch.setattr(rapidocr_engine, "smart_run",
                        lambda *a, **k: OCRResult(text="로컬 본문", confidence=conf, engine=OCREngine.RAPIDOCR))
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


# (품질불량?, RapidOCR 신뢰도) — 저신뢰 폴백 / 품질불량+저신뢰
# 품질불량+고신뢰는 이제 Vision 을 부르지 않는다(2026-10-15) → test_quality_flag_alone_does_not_call_vision
CASES = [(False, 40.0), (True, 40.0)]


@pytest.mark.parametrize("run", [_gui, _cli])
@pytest.mark.parametrize("blurry,conf", CASES)
def test_vision_failure_keeps_rapidocr(monkeypatch, run, blurry, conf):
    page = run(monkeypatch, blurry, conf, _boom)
    assert page.engine == OCREngine.RAPIDOCR and "로컬" in page.raw_text
    assert page.needs_review and "폴백 실패" in page.review_reason


@pytest.mark.parametrize("run", [_gui, _cli])
@pytest.mark.parametrize("blurry,conf", CASES)
def test_vision_success_is_used(monkeypatch, run, blurry, conf):
    page = run(monkeypatch, blurry, conf, _vision_ok)
    assert page.engine == OCREngine.OLLAMA and "비전" in page.raw_text
    assert not page.needs_review


def test_batch_low_confidence_page_flagged_for_review(monkeypatch):
    # 일괄(no_fallback=True)은 Vision 을 부르지 않는 대신 '검토 필요'로 남겨야 한다(825f19c 설계) —
    # 표시하는 블록이 no_fallback 조건 안에 있어 일괄에서는 늘 빠졌다
    ocr_gradio = pytest.importorskip("ocr_gradio")
    calls = []
    _setup(monkeypatch, False, 40.0, lambda *a, **k: calls.append(1))
    page = ocr_gradio._process_page(IMG, 1, "kor+eng", no_preprocess=True, no_fallback=True,
                                    no_figure=True, llm_correct=False, llm_budget=0)[0]
    assert page.needs_review and "신뢰도 낮음" in page.review_reason
    assert calls == []                                     # 유료 호출 없음


def test_vision_first_failure_flagged_for_review(monkeypatch):
    # Vision 우선 모드에서 Vision 이 실패해 로컬 결과로 넘어간 쪽은 표시 없이 저장됐다
    ocr_gradio = pytest.importorskip("ocr_gradio")
    _setup(monkeypatch, False, 95.0, _boom)
    page = ocr_gradio._process_page(IMG, 1, "kor+eng", no_preprocess=True, no_fallback=False,
                                    no_figure=True, llm_correct=False, llm_budget=0, vision_first=True)[0]
    assert page.engine == OCREngine.RAPIDOCR
    assert page.needs_review and "Vision 오류" in page.review_reason
