"""끊긴 전체 처리를 OCR 복구본(_OCR중.txt)에서 이어서 처리 · 쪽당 대기 시간 제한.

Regression: 「분석의힘」 277쪽이 261쪽에서 멈춰 재시작하면 처음부터 다시 읽어야 했다(2026-11-02).
Intent: docs/intents/2026-11-02-OCR-이어서처리.md
"""
import threading
from types import SimpleNamespace

import pytest

import ocr_gradio
from core import config, storage_manager
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
    results = tmp_path / "results"
    results.mkdir()
    monkeypatch.setattr(ocr_gradio, "RESULTS_DIR", results)
    base = tmp_path / "storage"
    monkeypatch.setattr(storage_manager, "_BASE", base)
    monkeypatch.setattr(storage_manager, "_UPLOADS", base / "uploads")
    monkeypatch.setattr(storage_manager, "_RESULTS", base / "results")
    monkeypatch.setattr(storage_manager, "_HISTORY", base / "history.json")
    return results


def _recording_page(calls):
    def fake(image, page_num, *a, **k):
        calls.append(page_num)
        t = f"{page_num}쪽 새로 읽음."
        return PageResult(page_num=page_num, raw_text=t, corrected_text=t,
                          engine=OCREngine.RAPIDOCR, confidence=95.0), []
    return fake


def _run(src):
    return ocr_gradio.process_full(
        SimpleNamespace(name=str(src)), "kor+eng", "txt",
        False, False, True, False, 0, "", "RapidOCR 우선",
        cleanup=False, front_matter=False, progress=lambda *a, **k: None)


def _drain(gen):
    last = None
    for last in gen:
        pass
    return last


def _final(results):
    out = [p for p in results.iterdir() if "OCR중" not in p.name]
    assert len(out) == 1
    return out[0].read_text(encoding="utf-8")


def test_resume_reads_only_remaining_pages(three_page_scan, sandbox, monkeypatch):
    (sandbox / "scan_OCR중.txt").write_text(
        "[페이지 1]\n1쪽 복원 본문.\n\n[페이지 2]\n2쪽 끊긴 본문\n\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(ocr_gradio, "_process_page", _recording_page(calls))
    last = _drain(_run(three_page_scan))
    assert calls == [2, 3]                       # 마지막 복원 쪽(2)은 다시 읽는다
    text = _final(sandbox)
    assert "1쪽 복원 본문." in text and "2쪽 새로 읽음." in text and "3쪽 새로 읽음." in text
    assert "2쪽 끊긴 본문" not in text
    assert "1쪽 복원" in str(last[1])            # 진행 기록에 복원 표시
    assert not [p for p in sandbox.iterdir() if "OCR중" in p.name]   # 완료 후 복구본 정리


def test_resume_keeps_restored_pages_if_interrupted_again(three_page_scan, sandbox, monkeypatch):
    (sandbox / "scan_OCR중.txt").write_text(
        "[페이지 1]\n1쪽 복원 본문.\n\n[페이지 2]\n2쪽 본문\n\n", encoding="utf-8")
    monkeypatch.setattr(ocr_gradio, "_process_page", _recording_page([]))
    gen = _run(three_page_scan)
    for out in gen:
        if "2/3페이지 완료" in str(out[0]):
            break
    gen.close()
    new = [p for p in sandbox.iterdir() if "OCR중_" in p.name]   # 이번 실행의 복구본
    assert new and "1쪽 복원 본문." in new[0].read_text(encoding="utf-8")


def test_resume_skipped_when_checkpoint_has_more_pages_than_source(three_page_scan, sandbox, monkeypatch):
    # 이름만 같은 다른 책의 복구본 — 쪽 수가 원본보다 많으면 쓰지 않는다
    (sandbox / "scan_OCR중.txt").write_text(
        "".join(f"[페이지 {n}]\n다른 책 {n}쪽\n\n" for n in range(1, 6)), encoding="utf-8")
    calls = []
    monkeypatch.setattr(ocr_gradio, "_process_page", _recording_page(calls))
    _drain(_run(three_page_scan))
    assert calls == [1, 2, 3]
    assert "다른 책" not in _final(sandbox)


def test_resume_restores_blank_page_as_blank(three_page_scan, sandbox, monkeypatch):
    (sandbox / "scan_OCR중.txt").write_text(
        "[페이지 1]\n\n\n[페이지 2]\n2쪽\n\n[페이지 3]\n3쪽\n\n", encoding="utf-8")
    monkeypatch.setattr(ocr_gradio, "_process_page", _recording_page([]))
    pages = ocr_gradio._resume_pages("scan")[0]
    assert [p.page_num for p in pages] == [1, 2]
    assert pages[0].is_blank and not pages[1].is_blank
    assert pages[1].engine == OCREngine.RESTORED


def test_resume_skips_checkpoint_still_being_written(sandbox):
    # 같은 책을 다른 처리가 지금 읽는 중이면 그 복구본은 이어받지 않는다 — 실물 확인 중 겹쳐 돌아
    # 진행 중 복구본을 가져가고 끝에 지우려다 WinError 32 로 죽었다(2026-11-02)
    done = sandbox / "scan_OCR중.txt"
    done.write_text("[페이지 1]\n옛 1쪽\n\n[페이지 2]\n옛 2쪽\n\n", encoding="utf-8")
    live = ocr_gradio._CleanCheckpoint("scan", ocr_gradio.OutputFormat.TXT, suffix="OCR중")
    try:
        for n in range(1, 4):
            live.append(f"[페이지 {n}]\n지금 {n}쪽")
        pages, src = ocr_gradio._resume_pages("scan")
        assert src == done and [p.corrected_text for p in pages] == ["옛 1쪽"]
    finally:
        live.close()
    assert ocr_gradio._resume_pages("scan")[1] == live.path     # 닫힌 뒤에는 이어받을 수 있다


def _table_page(calls):
    # 표 있는 쪽 — 본문엔 표 자리 표지, 표 내용은 figures 에(실제 _finalize 와 같은 꼴)
    def fake(image, page_num, *a, **k):
        calls.append(page_num)
        t = f"{page_num}쪽 본문.\n\n⟨TABLE:{page_num}-1⟩"
        return PageResult(page_num=page_num, raw_text=t, corrected_text=t,
                          engine=OCREngine.RAPIDOCR, confidence=95.0,
                          figures=[f"[표 1]\n| 항목 | 값 |\n|---|---|\n| 표{page_num}칸 | 1 |"]), []
    return fake


def test_resume_keeps_tables_of_restored_pages(three_page_scan, sandbox, monkeypatch):
    # 복구본엔 표 자리 표지만 남고 표 내용이 없어, 이어 붙인 쪽의 표가 통째로 사라졌다
    # (분석의힘 표 8개 소실, 2026-11-02)
    calls = []
    monkeypatch.setattr(ocr_gradio, "_process_page", _table_page(calls))
    gen = _run(three_page_scan)
    for out in gen:
        if "3/3페이지 완료" in str(out[0]):
            break
    gen.close()                                      # 다 읽고 저장 전에 끊김
    calls.clear()
    _drain(_run(three_page_scan))
    assert calls == [3]                              # 1·2쪽은 표까지 복원
    text = _final(sandbox)
    assert "표1칸" in text and "표2칸" in text and "표3칸" in text


def test_old_checkpoint_without_table_data_stops_before_table_page(sandbox):
    # 표 내용이 없는 옛 복구본 — 표 쪽 앞까지만 복원하고 그 쪽부터 다시 읽는다
    (sandbox / "scan_OCR중.txt").write_text(
        "[페이지 1]\n1쪽\n\n[페이지 2]\n2쪽\n\n⟨TABLE:2-1⟩\n\n[페이지 3]\n3쪽\n\n[페이지 4]\n4쪽\n\n",
        encoding="utf-8")
    pages, _ = ocr_gradio._resume_pages("scan")
    assert [p.page_num for p in pages] == [1]


def test_page_timeout_marks_review_and_continues(three_page_scan, sandbox, monkeypatch):
    release = threading.Event()

    def fake(image, page_num, *a, **k):
        if page_num == 2:
            release.wait(5)                      # 응답 없는 쪽
        t = f"{page_num}쪽 본문."
        return PageResult(page_num=page_num, raw_text=t, corrected_text=t,
                          engine=OCREngine.RAPIDOCR, confidence=95.0), []

    monkeypatch.setattr(ocr_gradio, "_process_page", fake)
    monkeypatch.setattr(config, "get_page_timeout_s", lambda: 0.3)
    try:
        last = _drain(_run(three_page_scan))
    finally:
        release.set()
    text = _final(sandbox)
    assert "1쪽 본문." in text and "3쪽 본문." in text
    assert "시간 초과" in str(last[1])


# ── 판독 원문 복구본(intent 2026-11-10 정제-후처리-통합) — 1차가 쪽 교정을 미루므로 복구본에도 판독 원문을 적는다 ──

def _raw_page(n, **kw):
    return PageResult(page_num=n, raw_text=f"  {n}쪽 들여 쓴 첫 줄\n둘째 줄", corrected_text="",
                      engine=OCREngine.RAPIDOCR, confidence=91.0, **kw)


def test_checkpoint_restores_raw_pages_exactly(sandbox):
    pages = [_raw_page(1, needs_review=True, review_reason="신뢰도 낮음 (61.0%)"), _raw_page(2), _raw_page(3)]
    (sandbox / "scan_OCR중.txt").write_text(
        "".join(ocr_gradio._ckpt_chunk(p) + "\n\n" for p in pages), encoding="utf-8")
    restored, _ = ocr_gradio._resume_pages("scan")
    assert restored == pages[:2]            # 줄·들여쓰기·엔진·신뢰도·검토 표시까지 같다(마지막 쪽은 다시 읽는다)


def test_resumed_raw_pages_are_corrected_and_old_pages_are_not(three_page_scan, sandbox, monkeypatch):
    # 새 복구본 쪽(판독 원문)은 이어서 처리할 때 교정하고, 옛 복구본 쪽(이미 교정된 글)은 다시 교정하지 않는다
    (sandbox / "scan_OCR중.txt").write_text(
        "[페이지 1]\n옛 복구본 1쪽.\n\n" + ocr_gradio._ckpt_chunk(_raw_page(2)) + "\n\n"
        + ocr_gradio._ckpt_chunk(_raw_page(3)) + "\n\n", encoding="utf-8")
    monkeypatch.setattr(ocr_gradio, "_process_page", _recording_page([]))
    monkeypatch.setattr(ocr_gradio, "_correct_text", lambda t, *a: f"교정본({t.split()[0]})")
    _drain(_run(three_page_scan))
    text = _final(sandbox)
    assert "옛 복구본 1쪽." in text and "교정본(옛" not in text
    assert "교정본(2쪽)" in text
    assert "3쪽 새로 읽음." in text
