"""CLI 가 API 키를 config.json(git 추적·USB 동기화 대상)에 평문으로 쓰지 않는다.

Regression: ISSUE-D — `ocr_tool.py settings --set-key` / `setup` 이 키를 config.json 에
저장해, 일반 커밋 한 번으로 키가 저장소·동기화본에 들어갈 수 있었다.
웹 화면처럼 환경변수 전용으로 통일(2026-09-26 결정과 일치).
Found by /qa on 2026-10-03 · Report: .gstack/qa-reports/qa-report-ocr-2026-10-03.md
"""
from click.testing import CliRunner

import ocr_tool
from core import config


def test_set_key_does_not_write_config(tmp_path, monkeypatch):
    cfg = tmp_path / "config.json"
    monkeypatch.setattr(config, "_CONFIG_FILE", cfg)
    res = CliRunner().invoke(ocr_tool.cli, ["settings", "--set-key", "AIzaSECRET-test-123"])
    assert res.exit_code != 0
    assert not cfg.exists() or "AIzaSECRET" not in cfg.read_text(encoding="utf-8")
    assert "GEMINI_API_KEY" in res.output or "ANTHROPIC_API_KEY" in res.output
