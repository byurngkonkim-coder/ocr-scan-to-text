"""02_결과 확정 실패 시 경고.

Regression: ISSUE-J — 02_결과 저장이 잠김·권한 문제로 실패해도 예외를 삼키고(None),
화면에는 임시 폴더 경로가 '저장 위치'로 떠 사용자가 결과가 저장된 줄 알았다.
Found by /qa on 2026-10-03 · Report: .gstack/qa-reports/qa-report-ocr-2026-10-03.md
"""
import pytest

ocr_gradio = pytest.importorskip("ocr_gradio")


def test_publish_failure_is_logged(tmp_path, monkeypatch):
    blocker = tmp_path / "02_결과"
    blocker.write_text("폴더 자리에 파일이 있어 저장 불가", encoding="utf-8")
    monkeypatch.setattr(ocr_gradio, "RESULTS_DIR", blocker)
    out = tmp_path / "out.txt"
    out.write_text("본문", encoding="utf-8")
    logs: list[str] = []
    assert ocr_gradio._finalize_results(None, None, out, "책", logs) is None
    assert logs and "02_결과" in logs[-1]
