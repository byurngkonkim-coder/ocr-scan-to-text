"""처리 기록 표에 Vision 처리 쪽 수 표시.

Regression: ISSUE-L — FocalPoint(233쪽)는 10쪽만 Gemini, 223쪽이 Tesseract 로 처리돼
품질이 떨어졌지만 처리 기록 표에는 'tesseract_first' 만 보여 알 수 없었다.
Found by /qa on 2026-10-03 · Report: .gstack/qa-reports/qa-report-ocr-2026-10-03.md
"""
import json

from core import storage_manager as sm


def _table(tmp_path, monkeypatch, entry):
    h = tmp_path / "history.json"
    h.write_text(json.dumps([entry], ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(sm, "_HISTORY", h)
    return sm.get_history_table()


def test_engine_cell_shows_vision_share(tmp_path, monkeypatch):
    row = _table(tmp_path, monkeypatch, {"engine_mode": "rapidocr_first",
                                         "pages_total": 233, "vision_pages": 10})[0]
    assert len(row) == 9 and "Vision 10/233" in row[4]


def test_text_layer_mode_unchanged(tmp_path, monkeypatch):
    row = _table(tmp_path, monkeypatch, {"engine_mode": "pdf_text_layer",
                                         "pages_total": 8, "vision_pages": 0})[0]
    assert row[4] == "pdf_text_layer"
