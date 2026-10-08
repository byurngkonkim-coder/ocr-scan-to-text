"""정제 체크포인트(복구본) 재실행 시 보존.

Regression: ISSUE-G — `{stem}_정제중` 파일을 open("w") 로 열어, 앞 실행이 중간에 끊겨
남긴 복구본이 재실행 시작 즉시 0바이트로 덮어써졌다.
Found by /qa on 2026-10-03 · Report: .gstack/qa-reports/qa-report-ocr-2026-10-03.md
"""
import pytest

ocr_gradio = pytest.importorskip("ocr_gradio")
from core.types import OutputFormat


def test_previous_checkpoint_survives_rerun(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr_gradio, "RESULTS_DIR", tmp_path)
    first = ocr_gradio._CleanCheckpoint("책", OutputFormat.TXT)
    first.append("앞 실행에서 정제된 150구간")
    first.close()                                   # 중간에 끊김
    second = ocr_gradio._CleanCheckpoint("책", OutputFormat.TXT)
    second.close()
    kept = [p for p in tmp_path.iterdir() if "앞 실행" in p.read_text(encoding="utf-8")]
    assert len(kept) == 1 and kept[0] != second.path
