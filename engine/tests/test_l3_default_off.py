"""L3(Claude Haiku) 쪽별 교정 기본값.

Regression: ISSUE-E — Claude 키가 환경에 있으면 고급 설정 안의 L3 교정이 기본 ON 이라,
Gemini 를 골라도 쪽마다 Claude 유료 호출이 숨어서 나갔다(2026-09-27 실측 75회).
Found by /qa on 2026-10-03 · Report: .gstack/qa-reports/qa-report-ocr-2026-10-03.md
"""
import pytest

ocr_gradio = pytest.importorskip("ocr_gradio")


def test_l3_checkbox_off_even_with_claude_key(monkeypatch):
    import gradio as gr
    monkeypatch.setattr(ocr_gradio.llm_corrector, "is_available", lambda: True)
    demo = ocr_gradio.build_ui()
    boxes = [b for b in demo.blocks.values()
             if isinstance(b, gr.Checkbox) and "L3" in (b.label or "")]
    assert boxes and boxes[0].value is False
