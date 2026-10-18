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


def test_same_stem_books_running_together_get_separate_checkpoints(tmp_path, monkeypatch):
    # 다른 폴더의 같은 이름 책 두 권이 동시에 돌면 열린 복구본 이름 바꾸기가 WinError 32 로 실패하거나
    # 두 책 쪽이 한 파일에 섞였다(감사 18번)
    monkeypatch.setattr(ocr_gradio, "RESULTS_DIR", tmp_path)
    a = ocr_gradio._CleanCheckpoint("001", OutputFormat.TXT, suffix="OCR중")
    a.append("A 책 1쪽")
    b = ocr_gradio._CleanCheckpoint("001", OutputFormat.TXT, suffix="OCR중")   # A 가 열고 있는 중
    b.append("B 책 1쪽")
    c = ocr_gradio._CleanCheckpoint("001", OutputFormat.TXT, suffix="OCR중")
    for ck in (a, b, c):
        ck.close()
    assert len({a.path, b.path, c.path}) == 3
    assert "A 책" in a.path.read_text(encoding="utf-8") and "A 책" not in b.path.read_text(encoding="utf-8")
    import batch_runner
    assert all(batch_runner._PARTIAL_RE.search(ck.path.stem) for ck in (a, b, c))
