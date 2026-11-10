"""쪽 교정을 글자 읽기에서 떼어 후처리로 — 1차는 판독 원문만 남기고, 쪽 교정은 다 읽은 뒤 한 번에 한다.

미룬 교정의 결과는 지금(읽으면서 바로 교정)과 글자 단위로 같아야 한다.
Intent: docs/intents/2026-11-10-정제-후처리-통합.md
"""
from types import SimpleNamespace

import pytest

import ocr_gradio
from core import figure_detector, storage_manager
from core.types import OCREngine, OCRResult, PageResult

_RAW = ("제1장 전략\n  경영은 사람을 통해 성과를\n내는 일이다. 그래서 사람이\n먼저다.\n"
        + figure_detector.table_token(1, 1) + "\n  다음 문단은 표 뒤에\n이어진다.")
_FIGS = ["[표 1]\n| 항목 | 값 |\n|---|---|\n| 가 | 1 |"]


@pytest.mark.parametrize("engine", [OCREngine.RAPIDOCR, OCREngine.GEMINI])
def test_deferred_correction_equals_inline_correction(engine):
    ocr = OCRResult(text=_RAW, confidence=96.0, engine=engine)
    now = ocr_gradio._finalize(ocr, 1, _FIGS, False, 0)
    later = ocr_gradio._finalize(ocr, 1, _FIGS, False, 0, correct=False)
    assert later.raw_text == _RAW and later.corrected_text == ""      # 1차는 판독 원문만
    assert now.corrected_text and now.corrected_text != _RAW          # 지금 흐름은 줄을 잇는 등 글자를 고친다
    ocr_gradio._correct_pages([later])
    assert later == now


def test_pages_that_already_have_corrected_text_are_left_alone():
    """유료 교정본이 저장된 쪽·옛 복구본에서 되살린 쪽 — 두 번 교정하면 띄어쓰기를 다시 붙이기도 한다."""
    done = PageResult(page_num=1, raw_text="판독 원문", corrected_text="이미 교정한 글", engine=OCREngine.RESTORED,
                      confidence=0.0)
    blank = PageResult(page_num=2, raw_text="", corrected_text="", engine=OCREngine.RAPIDOCR, confidence=0.0,
                       is_blank=True)
    ocr_gradio._correct_pages([done, blank])
    assert done.corrected_text == "이미 교정한 글" and blank.corrected_text == ""


def test_correction_error_keeps_the_page(monkeypatch):
    """교정 오류 하나로 읽은 쪽을 잃지 않는다 — 판독 원문을 그대로 쓰고 알린다."""
    def boom(*a, **k):
        raise RuntimeError("사전 고장")
    monkeypatch.setattr(ocr_gradio, "_correct_text", boom)
    page = PageResult(page_num=7, raw_text="읽은 글", corrected_text="", engine=OCREngine.RAPIDOCR, confidence=90.0)
    logs: list[str] = []
    ocr_gradio._correct_pages([page], logs=logs)
    assert page.corrected_text == "읽은 글"
    assert any("p.7" in line and "사전 고장" in line for line in logs)


def test_paid_correction_option_is_passed_through(monkeypatch):
    seen = []
    monkeypatch.setattr(ocr_gradio, "_correct_text", lambda t, eng, llm, budget: seen.append((eng, llm, budget)) or t)
    page = PageResult(page_num=1, raw_text="읽은 글", corrected_text="", engine=OCREngine.GEMINI, confidence=90.0)
    ocr_gradio._correct_pages([page], True, 5)
    assert seen == [("gemini", True, 5)]


@pytest.fixture
def scan(tmp_path, monkeypatch):
    fitz = pytest.importorskip("fitz")
    p = tmp_path / "scan.pdf"
    doc = fitz.open()
    for _ in range(2):
        doc.new_page()                       # 텍스트 레이어 없음 → OCR 경로
    doc.save(str(p))
    doc.close()
    results = tmp_path / "results"
    results.mkdir()
    monkeypatch.setattr(ocr_gradio, "RESULTS_DIR", results)
    base = tmp_path / "storage"
    monkeypatch.setattr(storage_manager, "_BASE", base)
    monkeypatch.setattr(storage_manager, "_UPLOADS", base / "uploads")
    monkeypatch.setattr(storage_manager, "_RESULTS", base / "results")
    monkeypatch.setattr(storage_manager, "_HISTORY", base / "history.json")
    return p, results


def test_full_run_reads_without_correcting_then_corrects_once(scan, monkeypatch):
    src, results = scan
    asked = []

    def read(image, page_num, *a, **k):
        asked.append(k.get("correct"))
        return ocr_gradio._finalize(OCRResult(text=f"{page_num}쪽 판독", confidence=95.0, engine=OCREngine.RAPIDOCR),
                                    page_num, [], False, 0, correct=k.get("correct", True)), []

    corrected = []
    monkeypatch.setattr(ocr_gradio, "_process_page", read)
    monkeypatch.setattr(ocr_gradio, "_correct_text",
                        lambda t, *a: corrected.append(t) or f"교정본({t})")
    for _ in ocr_gradio.process_full(SimpleNamespace(name=str(src)), "kor+eng", "txt", False, False, True, False, 0,
                                     "", "RapidOCR 우선", cleanup=False, front_matter=False,
                                     progress=lambda *a, **k: None):
        pass
    assert asked == [False, False]                                  # 읽을 때는 교정을 미룬다
    assert corrected == ["1쪽 판독", "2쪽 판독"]                    # 다 읽은 뒤 쪽마다 한 번
    out = [p for p in results.iterdir() if "OCR중" not in p.name]
    text = out[0].read_text(encoding="utf-8")
    assert "교정본(1쪽 판독)" in text and "교정본(2쪽 판독)" in text
