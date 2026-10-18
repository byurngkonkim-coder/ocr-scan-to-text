"""'처리 중단' 버튼이 처리 단계(제너레이터)를 취소하는지.

Gradio 의 cancels 는 가리킨 단계만 취소한다. 연쇄(.then)의 마지막 단계를 가리키던 탓에
버튼을 눌러도 처리는 끝까지 돌고 Vision 요금도 계속 나갔다(감사 2026-10-18).
"""
import pytest

ocr_gradio = pytest.importorskip("ocr_gradio")


def test_stop_cancels_the_processing_steps():
    demo = ocr_gradio.build_ui()
    cancel_fns = [f for f in demo.fns.values() if f.cancels]
    targets = {demo.fns[i].name for f in cancel_fns for i in f.cancels}
    assert {"process_preview", "process_full_or_queue", "process_continue"} <= targets
