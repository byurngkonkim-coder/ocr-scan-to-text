"""앞부속 제거 청크의 과다 축약 경고.

Regression: ISSUE-I — 앞부속 제거 청크는 '전부 삭제→빈 출력'이 정당해 무결성 가드를
면제했는데, 그 탓에 LLM이 첫 청크 본문을 요약해 돌려줘도 아무 표시가 없었다.
거부하면 정당한 삭제까지 버리므로, 분량이 크게 줄면 경고만 남긴다(원문 사본은 보존됨).
Found by /qa on 2026-10-03 · Report: .gstack/qa-reports/qa-report-ocr-2026-10-03.md
"""
from core import text_cleaner


def _run(monkeypatch, reply):
    monkeypatch.setattr(text_cleaner._gemini, "run_text", lambda p: reply)
    text_cleaner.clean_pages(["본문 문장이 아주 길게 이어집니다. " * 40], drop_front_matter=True)
    return text_cleaner.get_last_report()["warnings"]


def test_heavy_shrink_in_front_chunk_is_warned(monkeypatch):
    assert any("앞부속" in w for w in _run(monkeypatch, "요약된 한 줄." * 10))


def test_empty_front_chunk_is_not_warned(monkeypatch):
    assert not _run(monkeypatch, "")          # 전부 부속물이라 비운 경우는 정당


def test_warning_reaches_screen(tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace
    import pytest
    ocr_gradio = pytest.importorskip("ocr_gradio")
    from core import storage_manager
    from core.types import OCREngine, PageResult
    monkeypatch.setattr(ocr_gradio, "RESULTS_DIR", tmp_path / "results")
    for name, sub in (("_BASE", ""), ("_UPLOADS", "uploads"), ("_RESULTS", "results"),
                      ("_HISTORY", "history.json")):
        monkeypatch.setattr(storage_manager, name, tmp_path / "storage" / sub)
    body = "본문 문장이 아주 길게 이어집니다. " * 40
    monkeypatch.setattr(ocr_gradio, "_process_page", lambda img, n, *a, **k: (PageResult(
        page_num=n, raw_text=body, corrected_text=body,
        engine=OCREngine.RAPIDOCR, confidence=95.0), []))
    monkeypatch.setattr(text_cleaner, "is_available", lambda local=False: True)
    monkeypatch.setattr(text_cleaner._gemini, "run_text", lambda p: "요약된 한 줄." * 10)
    monkeypatch.setattr(text_cleaner._ollama, "run_text", lambda p: "요약된 한 줄." * 10)  # 비용 우선 = 로컬 정제
    from core import config
    monkeypatch.setattr(config, "get_local_cleanup_engine", lambda: "ollama")  # LLM 축약 경고 경로
    png = Path(__file__).parent / "sample_docs" / "sample_kor_eng.png"
    last = None
    for last in ocr_gradio.process_full(
            SimpleNamespace(name=str(png)), "kor+eng", "txt", False, False, True, False, 0, "",
            "RapidOCR 우선", cleanup=True, front_matter=True, progress=lambda *a, **k: None):
        pass
    assert "앞부속" in str(last[1])
