"""① 탭 처리 중 화면에 '탭을 닫으면 멈춘다'는 안내가 나오는지(감사 17번 — 매뉴얼은 다시 열면 된다고만 안내)."""
import pytest

ocr_gradio = pytest.importorskip("ocr_gradio")


def test_status_warns_that_closing_tab_stops_processing():
    assert "닫거나 새로고침하지 마세요" in ocr_gradio._status_text(1, 10, 10, [])
