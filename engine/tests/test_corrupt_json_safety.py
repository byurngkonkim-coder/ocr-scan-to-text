"""이력·설정 JSON 손상 시 소실 방지.

Regression: ISSUE-C — history.json / config.json 을 읽다 실패하면 빈 값으로 넘어가고
다음 저장이 그 빈 값으로 파일을 덮어써 이력(최대 500건)·설정이 통째로 사라졌다.
쓰기도 원자적이지 않아 쓰는 도중 종료되면 파일이 깨졌다.
Found by /qa on 2026-10-03 · Report: .gstack/qa-reports/qa-report-ocr-2026-10-03.md
"""
import json

from core import config as cfg
from core import storage_manager as sm


def test_corrupt_history_is_backed_up_not_overwritten(tmp_path, monkeypatch):
    h = tmp_path / "history.json"
    h.write_text('[{"id": "old"', encoding="utf-8")          # 깨진 JSON
    monkeypatch.setattr(sm, "_HISTORY", h)
    assert sm.load_history() == []
    sm._save_history([{"id": "new"}])
    baks = list(tmp_path.glob("history.json.bak*"))
    assert baks and baks[0].read_text(encoding="utf-8") == '[{"id": "old"'
    assert json.loads(h.read_text(encoding="utf-8")) == [{"id": "new"}]


def test_history_write_leaves_no_temp(tmp_path, monkeypatch):
    h = tmp_path / "history.json"
    monkeypatch.setattr(sm, "_HISTORY", h)
    sm._save_history([{"id": "a"}])
    assert [p.name for p in tmp_path.iterdir()] == ["history.json"]


def test_corrupt_config_is_backed_up_not_overwritten(tmp_path, monkeypatch):
    c = tmp_path / "config.json"
    c.write_text('{"gemini_model": "x",', encoding="utf-8")
    monkeypatch.setattr(cfg, "_CONFIG_FILE", c)
    cfg.save({"default_dpi": 200})
    baks = list(tmp_path.glob("config.json.bak*"))
    assert baks and baks[0].read_text(encoding="utf-8") == '{"gemini_model": "x",'
    assert json.loads(c.read_text(encoding="utf-8"))["default_dpi"] == 200
