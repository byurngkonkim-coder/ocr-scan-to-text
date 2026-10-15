"""비용 우선 모드 비용 보호 (2026-10-15 요금 폭탄 회귀 방지).

- 비용 우선 정제는 내장 규칙(기본) 또는 로컬 Ollama — Gemini 호출 0
- 문서당 Vision 폴백 상한 (GUI·CLI)
- Gemini 결제·인증 오류(402 등) 뒤에는 재시도·후속 호출 없음
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import config, gemini_vision, text_cleaner


def _boom(_p):
    raise AssertionError("비용 우선 정제가 LLM 을 불렀다")


def test_local_rules_cleanup_calls_no_llm(monkeypatch):
    monkeypatch.setattr(config, "get_local_cleanup_engine", lambda: "rules")
    monkeypatch.setattr(text_cleaner._gemini, "run_text", _boom)
    monkeypatch.setattr(text_cleaner._ollama, "run_text", _boom)
    out = text_cleaner.clean_pages(["가치가 있으며, 성", "찰의 노력 없이는 안 된다."],
                                   drop_front_matter=True, local=True)
    assert "성찰의" in out                     # C-03 쪽 경계 잘린 어절 결합


def test_local_ollama_cleanup_when_configured(monkeypatch):
    monkeypatch.setattr(config, "get_local_cleanup_engine", lambda: "ollama")
    monkeypatch.setattr(text_cleaner._gemini, "run_text", _boom)
    monkeypatch.setattr(text_cleaner._ollama, "run_text", lambda p: "로컬로 정제된 본문 문장입니다." * 20)
    out = text_cleaner.clean_pages(["로컬로 정제된 본문 문장입니다." * 20], local=True)
    assert "로컬로 정제된" in out


def test_vision_budget_caps_pages(monkeypatch):
    ocr_gradio = pytest.importorskip("ocr_gradio")
    monkeypatch.setattr(config, "get_vision_max_ratio", lambda: 0.1)
    b = ocr_gradio._VisionBudget(40)          # 40쪽 × 10% = 4쪽
    assert [b.take() for _ in range(6)] == [True] * 4 + [False] * 2
    assert ocr_gradio._VisionBudget(5).left == 3   # 짧은 문서도 최소 3쪽


def test_cli_vision_budget_caps_pages(monkeypatch):
    """CLI(ocr_tool)도 GUI 와 같은 문서당 Vision 상한을 지킨다."""
    ocr_tool = pytest.importorskip("ocr_tool")
    calls = []
    low = SimpleNamespace(confidence=10.0, has_jamo_split=False)
    monkeypatch.setattr(config, "get_vision_max_ratio", lambda: 0.1)
    monkeypatch.setattr(ocr_tool.detector, "detect", lambda _p: "image")
    monkeypatch.setattr(ocr_tool.pdf_splitter, "load_image", lambda _p: [object()] * 40)
    monkeypatch.setattr(ocr_tool.quality_diagnoser, "diagnose", lambda _i: None)
    monkeypatch.setattr(ocr_tool.rapidocr_engine, "smart_run", lambda *_a, **_k: low)
    monkeypatch.setattr(ocr_tool.claude_vision, "is_available", lambda: True)
    monkeypatch.setattr(ocr_tool.claude_vision, "run", lambda *_a, **_k: calls.append(1) or low)
    monkeypatch.setattr(ocr_tool, "_finalize_page", lambda *a, **k: k)
    monkeypatch.setattr(ocr_tool.writer, "write", lambda *_a: None)
    res = ocr_tool._process_file(Path("x.png"), "out", None, "kor", 300,
                                 True, False, True, True, False, False, 0, False)
    assert len(calls) == 4                      # 40쪽 × 10%
    assert sum("상한" in p.get("review_reason", "") for p in res.pages) == 36


class _Err(Exception):
    code = 402


def test_billing_error_blocks_further_calls(monkeypatch):
    calls = []

    def gen(**_k):
        calls.append(1)
        raise _Err("prepayment credits are depleted")

    monkeypatch.setattr(gemini_vision, "_blocked", None)
    monkeypatch.setattr(gemini_vision, "_client",
                        SimpleNamespace(models=SimpleNamespace(generate_content=gen)))
    monkeypatch.setattr(config, "get_google_api_key", lambda: "AQ.test")
    with pytest.raises(_Err):
        gemini_vision.run_text("x")
    assert len(calls) == 1                      # 402 는 재시도하지 않는다
    assert not gemini_vision.is_available()
    with pytest.raises(RuntimeError):
        gemini_vision.run_text("y")
    assert len(calls) == 1                      # 차단 뒤에는 호출 자체를 안 보낸다
