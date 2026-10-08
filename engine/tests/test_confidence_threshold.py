"""비용 우선(Tesseract 우선) 모드의 Vision 폴백 신뢰도 기준.

Regression: ISSUE-F — 깨진 쪽(짧은 캡션·그림 쪽)의 Tesseract 신뢰도가 63~84 로 기준 70 을
넘어 Gemini 로 넘어가지 않고 쓰레기 텍스트가 결과에 남았다(FocalPoint p39·92·116).
하루명화 38쪽 보정(정답=Gemini 텍스트): 깨진 쪽 4개는 신뢰도 63~84, 본문 30쪽은 전부 88+.
또 config.json 의 confidence_threshold 는 코드가 읽지 않았다(상수 70 하드코딩).
Found by /qa on 2026-10-03 · Report: .gstack/qa-reports/qa-report-ocr-2026-10-03.md
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import blank_detector, config, quality_diagnoser, spellcheck, tesseract_engine, vision
from core.types import OCREngine, OCRResult, QualityReport

IMG = Image.new("RGB", (800, 1000), "white")


def _setup(monkeypatch, conf, tmp_path, threshold=None):
    cfg = tmp_path / "config.json"
    if threshold is not None:
        cfg.write_text(json.dumps({"confidence_threshold": threshold}), encoding="utf-8")
    monkeypatch.setattr(config, "_CONFIG_FILE", cfg)
    monkeypatch.setattr(vision, "is_available", lambda: True)
    monkeypatch.setattr(vision, "run", lambda *a, **k: OCRResult(
        text="비전 본문", confidence=95.0, engine=OCREngine.GEMINI))
    monkeypatch.setattr(spellcheck, "is_available", lambda: False)
    monkeypatch.setattr(blank_detector, "check",
                        lambda *a, **k: SimpleNamespace(is_blank=False, reason=""))
    monkeypatch.setattr(tesseract_engine, "smart_run", lambda *a, **k: OCRResult(
        text="깨진 캡션 rr 00", confidence=conf, engine=OCREngine.TESSERACT))
    monkeypatch.setattr(quality_diagnoser, "diagnose", lambda img: QualityReport(
        blur_score=500, avg_brightness=128, skew_angle=0,
        is_vertical_text=False, width_px=800, height_px=1000))


def _gui():
    ocr_gradio = pytest.importorskip("ocr_gradio")
    return ocr_gradio._process_page(IMG, 1, "kor+eng", no_preprocess=True, no_fallback=False,
                                    no_figure=True, llm_correct=False, llm_budget=0)[0]


def _cli():
    ocr_tool = pytest.importorskip("ocr_tool")
    return ocr_tool._process_page(IMG, 1, "kor+eng", no_preprocess=True, no_fallback=False,
                                  no_spellcheck=True, no_figure=True, save_figures=False,
                                  output_dir=".", stem="t", llm_correct=False, llm_budget=0,
                                  verbose=False)


@pytest.mark.parametrize("run", [_gui, _cli])
def test_default_threshold_escalates_conf_80(monkeypatch, tmp_path, run):
    _setup(monkeypatch, 80.0, tmp_path)
    assert run().engine == OCREngine.GEMINI


@pytest.mark.parametrize("run", [_gui, _cli])
def test_threshold_read_from_config(monkeypatch, tmp_path, run):
    _setup(monkeypatch, 80.0, tmp_path, threshold=70.0)
    assert run().engine == OCREngine.TESSERACT


@pytest.mark.parametrize("run", [_gui, _cli])
def test_body_page_conf_90_stays_tesseract(monkeypatch, tmp_path, run):
    _setup(monkeypatch, 90.0, tmp_path)
    assert run().engine == OCREngine.TESSERACT
