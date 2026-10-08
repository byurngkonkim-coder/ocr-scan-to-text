"""미리보기 완료 안내 문구가 실제 화면과 맞는지.

Regression: ISSUE-K — 미리보기 후 '아래 버튼으로 전체 처리를 계속하거나…' 라고 안내했지만
그 버튼 줄(continue_row)은 images_state 가 늘 None 이라 절대 나타나지 않았다.
버튼을 되살리면 4쪽부터만 처리돼 1~3쪽이 결과에서 빠지므로, 안내를 「전체 처리」로 바꾼다.
Found by /qa on 2026-10-03 · Report: .gstack/qa-reports/qa-report-ocr-2026-10-03.md
"""
from types import SimpleNamespace

import pytest

ocr_gradio = pytest.importorskip("ocr_gradio")
from core.types import OCREngine, PageResult


def test_preview_points_to_full_run_button(tmp_path, monkeypatch):
    fitz = pytest.importorskip("fitz")
    pdf = tmp_path / "scan.pdf"
    doc = fitz.open()
    for _ in range(6):
        doc.new_page()
    doc.save(str(pdf))
    doc.close()
    monkeypatch.setattr(ocr_gradio, "_process_page", lambda img, n, *a, **k: (PageResult(
        page_num=n, raw_text="본문.", corrected_text="본문.",
        engine=OCREngine.TESSERACT, confidence=95.0), []))
    last = None
    for last in ocr_gradio.process_preview(
            SimpleNamespace(name=str(pdf)), "kor+eng", "txt", False, False, True, False, 0, "",
            "Tesseract 우선", progress=lambda *a, **k: None):
        pass
    html = str(last[4])
    assert "전체 처리" in html and "아래 버튼" not in html
