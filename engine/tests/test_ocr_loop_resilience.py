"""OCR 페이지 루프 중단·예외 시 결과 보존.

Regression: ISSUE-B — 페이지 루프에 예외 격리·중간 저장이 없어, 한 쪽의 예외나
'처리 중단'이면 그때까지 OCR 한 쪽(유료 Gemini 포함)이 전부 사라졌다.
Found by /qa on 2026-10-03 · Report: .gstack/qa-reports/qa-report-ocr-2026-10-03.md
"""
from types import SimpleNamespace

import pytest

import ocr_gradio
from core import storage_manager
from core.types import OCREngine, PageResult


@pytest.fixture
def three_page_scan(tmp_path):
    fitz = pytest.importorskip("fitz")
    p = tmp_path / "scan.pdf"
    doc = fitz.open()
    for _ in range(3):
        doc.new_page()                       # 텍스트 레이어 없음 → OCR 경로
    doc.save(str(p))
    doc.close()
    return p


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr_gradio, "RESULTS_DIR", tmp_path / "results")
    base = tmp_path / "storage"
    monkeypatch.setattr(storage_manager, "_BASE", base)
    monkeypatch.setattr(storage_manager, "_UPLOADS", base / "uploads")
    monkeypatch.setattr(storage_manager, "_RESULTS", base / "results")
    monkeypatch.setattr(storage_manager, "_HISTORY", base / "history.json")
    return tmp_path / "results"


def _fake_page(fail_on=None):
    def fake(image, page_num, *a, **k):
        if page_num == fail_on:
            raise RuntimeError("OCR 폭발")
        t = f"{page_num}쪽 본문입니다."
        return PageResult(page_num=page_num, raw_text=t, corrected_text=t,
                          engine=OCREngine.RAPIDOCR, confidence=95.0), []
    return fake


def _run(src):
    return ocr_gradio.process_full(
        SimpleNamespace(name=str(src)), "kor+eng", "txt",
        False, False, True, False, 0, "", "RapidOCR 우선",
        cleanup=False, front_matter=False, progress=lambda *a, **k: None)


def test_page_exception_does_not_kill_run(three_page_scan, sandbox, monkeypatch):
    monkeypatch.setattr(ocr_gradio, "_process_page", _fake_page(fail_on=2))
    last = None
    for last in _run(three_page_scan):
        pass
    out = [p for p in sandbox.iterdir() if "OCR중" not in p.name]
    text = out[0].read_text(encoding="utf-8")
    assert "1쪽 본문" in text and "3쪽 본문" in text
    assert "p.2" in str(last[1])                     # 실패 쪽이 화면에 드러남


def test_cancel_keeps_ocr_checkpoint(three_page_scan, sandbox, monkeypatch):
    monkeypatch.setattr(ocr_gradio, "_process_page", _fake_page())
    gen = _run(three_page_scan)
    for out in gen:
        if "2/3페이지 완료" in str(out[0]):
            break
    gen.close()                                      # '처리 중단'
    ck = [p for p in sandbox.iterdir() if "OCR중" in p.name]
    assert ck and "2쪽 본문" in ck[0].read_text(encoding="utf-8")


def test_checkpoint_removed_on_success(three_page_scan, sandbox, monkeypatch):
    monkeypatch.setattr(ocr_gradio, "_process_page", _fake_page())
    for _ in _run(three_page_scan):
        pass
    assert not [p for p in sandbox.iterdir() if "OCR중" in p.name]
