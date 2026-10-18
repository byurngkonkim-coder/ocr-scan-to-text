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


def test_smart_run_retries_once_after_failure(monkeypatch):
    # 일괄 처리 중 일시 오류로 비던 쪽(대한민국CEO명언집 53쪽)을 한 번 더 읽는다 — 2026-10-17
    from core import rapidocr_engine as re_
    from core.types import OCREngine, OCRResult
    calls = []

    def flaky(image, lang="kor+eng", fast=False, refine=True):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("onnxruntime 일시 오류")
        return OCRResult(text="정의 속에서만 사회 질서가 중심이 된다.", confidence=93.0, engine=OCREngine.RAPIDOCR)

    monkeypatch.setattr(re_, "run", flaky)
    res = re_.smart_run(None)
    assert res.confidence == 93.0 and len(calls) == 2


def _flaky_render(fail_at):
    from PIL import Image

    def render(pdf_path, dpi=200, poppler_path=None, batch_size=8, first_page=1, last_page=None):
        for n in range(first_page, (last_page or 3) + 1):
            if n == fail_at:
                raise ValueError("PDF 페이지 수를 읽을 수 없습니다: I/O Error")
            yield Image.new("RGB", (400, 600), "white")
    return render


def test_render_failure_mid_book_is_partial_not_done(three_page_scan, sandbox, monkeypatch):
    # 책 중간에 쪽 그리기가 실패하면(외장 드라이브 끊김 등) 앞부분만 저장하고 '완료'로 확정하던 것
    from core import pdf_splitter
    monkeypatch.setattr(ocr_gradio, "_process_page", _fake_page())
    monkeypatch.setattr(ocr_gradio.margin_detector, "measure", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(pdf_splitter, "iter_pdf_pages", _flaky_render(3))
    last = None
    for last in _run(three_page_scan):
        pass
    assert str(last[1]).startswith("⚠ 부분 저장")
    names = [p.name for p in sandbox.iterdir()]
    assert any("OCR중" in n for n in names)                       # 복구본을 지우지 않는다
    assert any(n.endswith("_미완.txt") for n in names)            # 결과 이름에 미완 표시
    assert storage_manager.load_history() == []                   # 처리 기록에 '완료'로 넣지 않는다


def test_render_retries_transient_failure(tmp_path, monkeypatch):
    from core import pdf_splitter
    from PIL import Image
    calls = []

    def flaky_convert(path, dpi, first_page, last_page, **kw):
        calls.append(first_page)
        if len(calls) == 1:
            raise OSError("I/O Error")
        return [Image.new("RGB", (10, 10)) for _ in range(first_page, last_page + 1)]

    monkeypatch.setattr(pdf_splitter, "_require_pdf2image", lambda p: None)
    monkeypatch.setattr(pdf_splitter, "count_pages", lambda *a, **k: 3)
    monkeypatch.setattr(pdf_splitter, "convert_from_path", flaky_convert)
    monkeypatch.setattr(pdf_splitter, "_RETRY_WAIT", 0)
    assert len(list(pdf_splitter.iter_pdf_pages(tmp_path / "x.pdf"))) == 3


def test_render_short_batch_raises_instead_of_shifting_pages(tmp_path, monkeypatch):
    # poppler 가 묶음 중 일부 쪽만 내보내면 이후 쪽 번호가 하나씩 밀렸다
    from core import pdf_splitter
    from PIL import Image
    monkeypatch.setattr(pdf_splitter, "_require_pdf2image", lambda p: None)
    monkeypatch.setattr(pdf_splitter, "count_pages", lambda *a, **k: 3)
    monkeypatch.setattr(pdf_splitter, "convert_from_path",
                        lambda *a, **k: [Image.new("RGB", (10, 10))])
    monkeypatch.setattr(pdf_splitter, "_RETRY_WAIT", 0)
    with pytest.raises(ValueError):
        list(pdf_splitter.iter_pdf_pages(tmp_path / "x.pdf"))


def test_mostly_english_page_with_hangul_keeps_korean_read(monkeypatch):
    # 영문이 대부분인 참고문헌 쪽을 영문 모델로 다시 읽어 한글 항목이 통째로 사라지던 것(74자 → 0자)
    from core import rapidocr_engine as re_
    from core.types import OCRResult
    eng_line = "Drucker, P. F. (1954). The Practice of Management. New York: Harper & Row.\n" * 12
    calls = []

    def fake_run(image, lang="kor+eng", fast=False, refine=True):
        calls.append(lang)
        text = eng_line + "피터 드러커 (2006). 경영의 실제." if lang != "eng" else eng_line
        return OCRResult(text=text, confidence=95.0, engine=OCREngine.RAPIDOCR)

    monkeypatch.setattr(re_, "run", fake_run)
    res = re_.smart_run(None)
    assert "경영의 실제" in res.text and calls == ["kor+eng"]


def test_pure_english_page_still_reread_with_english_model(monkeypatch):
    from core import rapidocr_engine as re_
    from core.types import OCRResult
    calls = []
    monkeypatch.setattr(re_, "run", lambda image, lang="kor+eng", fast=False, refine=True: (
        calls.append(lang), OCRResult(text="The Practice of Management " * 5, confidence=95.0,
                                      engine=OCREngine.RAPIDOCR))[1])
    re_.smart_run(None)
    assert calls == ["kor+eng", "eng"]
