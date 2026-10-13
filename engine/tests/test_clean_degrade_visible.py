"""문장 정제가 원문으로 강등되면 화면·대기열·처리 기록에 드러낸다.

Regression: 2026-10-13 — Gemini 키가 401 로 거절돼 오늘 결과물 4건 모두 정제가 원문 보존으로
강등됐는데, 화면 위쪽은 '키 연결됨', 대기열은 '완료', 처리 기록엔 아무것도 없어 알 수 없었다
(드러커 824줄 중 822줄이 원문 그대로).
"""
import json
import time

import pytest

import batch_runner as br
from core import config, storage_manager as sm

ocr_gradio = pytest.importorskip("ocr_gradio")


def test_note_names_api_failure():
    note = ocr_gradio._clean_degrade_note({"chunks": 40, "fallback_api": 40, "fallback_guard": 0})
    assert "문장 정제 미적용 40/40" in note and "API 오류" in note


def test_no_note_when_all_cleaned():
    assert ocr_gradio._clean_degrade_note({"chunks": 40, "fallback_api": 0, "fallback_guard": 0}) == ""
    assert ocr_gradio._clean_degrade_note(None) == ""


def test_history_shows_note(tmp_path, monkeypatch):
    h = tmp_path / "history.json"
    h.write_text(json.dumps([{"engine_mode": "rapidocr_first", "pages_total": 325, "vision_pages": 0,
                              "clean_note": "문장 정제 미적용 40/40 구간"}], ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(sm, "_HISTORY", h)
    assert "정제 미적용 40/40" in sm.get_history_table()[0][4]


def test_batch_queue_shows_note(monkeypatch, tmp_path):
    br.stop()
    br._jobs.clear()
    br._stop.clear()
    monkeypatch.setattr(config, "get_batch_workers", lambda: 1)

    def process_full(file_obj, *args, progress=None, **kwargs):
        yield ("", "  ⚠️ 문장 정제 미적용 3/3 구간(API 오류 3 — 키·연결 확인)\n  ✅ 저장", "", "a.txt", None, None)
    monkeypatch.setattr(ocr_gradio, "process_full", process_full)
    br.submit([tmp_path / "a.pdf"], (), {})
    end = time.time() + 5
    while br.busy() and time.time() < end:
        time.sleep(0.05)
    assert br._jobs[0].state == "완료"
    assert "정제 미적용 3/3" in br.status_html()
    br._jobs.clear()
