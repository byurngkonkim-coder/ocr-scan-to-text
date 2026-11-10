"""전체 처리(1차)는 읽은 쪽을 쪽 원문 파일로 남긴다 — 후처리(쪽 교정·정제·재처리)를 OCR 없이 다시 돌릴 재료.

Intent: docs/intents/2026-11-10-정제-후처리-통합.md
"""
from types import SimpleNamespace

import pytest

import ocr_gradio
from core import page_store, storage_manager
from core.types import OCREngine, OCRResult, PageResult


@pytest.fixture
def scan(tmp_path, monkeypatch):
    fitz = pytest.importorskip("fitz")
    p = tmp_path / "책.pdf"
    doc = fitz.open()
    for _ in range(3):
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

    def read(image, page_num, *a, **k):      # 1차가 하는 대로 — 판독 원문만(줄 그대로), 교정은 미룸
        ocr = OCRResult(text=f"{page_num}쪽 첫 줄\n  {page_num}쪽 둘째 줄", confidence=95.0, engine=OCREngine.RAPIDOCR)
        return ocr_gradio._finalize(ocr, page_num, [], False, 0, correct=k.get("correct", True)), []
    monkeypatch.setattr(ocr_gradio, "_process_page", read)
    return p, results, base / "uploads"


def _run(src, cleanup=False, front_matter=False, llm_correct=False):
    last = None
    for last in ocr_gradio.process_full(
            SimpleNamespace(name=str(src)), "kor+eng", "txt", False, False, True, llm_correct, 3, "",
            "RapidOCR 우선", cleanup=cleanup, front_matter=front_matter, progress=lambda *a, **k: None):
        pass
    return last


def _page_files(uploads):
    return sorted(uploads.glob("*_쪽원문.json"))


def test_full_run_leaves_page_file_named_after_the_result(scan):
    src, results, uploads = scan
    _run(src)
    result = [p for p in results.iterdir() if p.suffix == ".txt"][0]
    files = _page_files(uploads)
    assert [f.name for f in files] == [f"{result.stem}_쪽원문.json"]       # 결과 이름과 짝 — 임시 이름이 남지 않는다
    assert files[0] == storage_manager.page_file_path(result)
    pages, meta = page_store.load(files[0])
    assert [p.raw_text for p in pages] == [f"{n}쪽 첫 줄\n  {n}쪽 둘째 줄" for n in (1, 2, 3)]   # 판독 원문, 줄 그대로
    assert all(p.corrected_text == "" for p in pages)                     # 교정본은 후처리가 다시 만든다
    assert meta["source"] == "책.pdf" and meta["total_pages"] == 3
    assert meta["options"]["cleanup"] is False and meta["options"]["front_matter"] is False


def test_page_file_records_the_choices_of_the_run(scan):
    src, _, uploads = scan
    _run(src, cleanup=True, front_matter=True)
    meta = page_store.load(_page_files(uploads)[0])[1]
    assert meta["options"]["cleanup"] is True and meta["options"]["front_matter"] is True


def test_paid_correction_is_kept_in_the_page_file(scan, monkeypatch):
    """유료 교정(L3)을 켠 실행 — 그 교정본은 다시 만들 수 없으므로 쪽 원문 파일에 함께 남긴다."""
    src, _, uploads = scan
    monkeypatch.setattr(ocr_gradio, "_correct_text", lambda t, eng, llm, budget: f"유료교정({llm}):{t.split()[0]}")
    _run(src, llm_correct=True)
    pages, meta = page_store.load(_page_files(uploads)[0])
    assert [p.corrected_text for p in pages] == [f"유료교정(True):{n}쪽" for n in (1, 2, 3)]
    assert meta["options"]["llm_correct"] is True


def test_run_still_finishes_when_page_file_cannot_be_written(scan, monkeypatch):
    src, results, uploads = scan

    def boom(*a, **k):
        raise OSError("디스크 가득")
    monkeypatch.setattr(page_store, "save", boom)
    last = _run(src)
    assert [p for p in results.iterdir() if p.suffix == ".txt"]           # 결과는 저장된다
    assert not _page_files(uploads)
    assert "쪽 원문 파일" in str(last[1]) and "디스크 가득" in str(last[1])   # 숨기지 않고 알린다


def test_refine_returns_cleaned_text_and_md_and_reports_progress():
    """정제 블록을 묶은 함수 — 전체 처리와 재처리 도구가 같은 순서를 쓴다."""
    pages = [
        PageResult(page_num=1, raw_text="첫 쪽 본문은 여기서", corrected_text="첫 쪽 본문은 여기서",
                   engine=OCREngine.RAPIDOCR, confidence=95.0),
        PageResult(page_num=2, raw_text="", corrected_text="", engine=OCREngine.RAPIDOCR, confidence=0.0,
                   is_blank=True),
        PageResult(page_num=3, raw_text="끝난다. 다음 문단이다.", corrected_text="끝난다. 다음 문단이다.",
                   engine=OCREngine.RAPIDOCR, confidence=95.0),
    ]
    logs: list[str] = []
    chunks: list[str] = []
    gen = ocr_gradio._refine(pages, False, ocr_gradio.OutputFormat.MD, logs, {"name": "책.pdf", "pages": 3},
                             on_chunk=chunks.append)
    steps = []
    try:
        while True:
            steps.append(next(gen))
    except StopIteration as stop:
        text, md, crep, note = stop.value
    assert steps and steps[-1][0] == steps[-1][1]                       # (끝난 구간, 전체 구간)
    assert "첫 쪽 본문은 여기서 끝난다." in text                         # 쪽 경계에서 잘린 문장을 이었다
    assert md.startswith("> **문서 정보**") and "책.pdf" in md
    assert chunks and crep["chunks"] >= 1 and note == ""
