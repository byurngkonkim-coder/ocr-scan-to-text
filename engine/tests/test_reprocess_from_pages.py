"""재처리 도구의 새 경로 — 쪽 원문 파일이 있는 책은 판독 원문에서 쪽 교정·정제부터 다시 하고 이어서 재처리한다.

같은 쪽에서 출발하면, '1차 직후 결과(02_결과 의 정제본)를 읽어 재처리'한 것과 글자 단위로 같아야 한다.
Intent: docs/intents/2026-11-10-정제-후처리-통합.md
"""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import ocr_gradio  # noqa: E402
import reprocess_book as rb  # noqa: E402
from core import ocr_typos, page_store, storage_manager, writer  # noqa: E402
from core.types import FileResult, OCREngine, OutputFormat, PageResult  # noqa: E402

STEM = "20261110_120000_경영의기본_홍길동"


def _raw_pages():
    def page(n, text, **kw):
        return PageResult(page_num=n, raw_text=text, corrected_text="", engine=OCREngine.RAPIDOCR,
                          confidence=95.0, **kw)
    return [
        page(1, "제1장 경영의 기본\n경영은 사람을 통해 성과롤 내는\n일이다. 그래서 경영자는 먼저\n사람을 본다. 사람이 일하게 하는 것이"),
        page(2, "경영의 첫째 과제다. 둘째 과제는\n자원을 성과로 바꾸는 일이다.\n자원은 저절로 성과가 되지 않는다."),
        page(3, "제2장 성과의 조건\n성과는 조직 밖에 있다. 조직 안에는\n비용만 있을 뿐이다. 고객이 값을\n치를 때 비로소 성과가 생긴다."),
        PageResult(page_num=4, raw_text="", corrected_text="", engine=OCREngine.RAPIDOCR, confidence=0.0,
                   is_blank=True, blank_reason="흰 쪽"),
        page(5, "따라서 경영자는 밖을 보아야 한다.\n시장과 고객을 아는 것이 출발점이다."),
    ]


@pytest.fixture
def env(tmp_path, monkeypatch):
    results = tmp_path / "02_결과"
    (results / "_재처리" / "_책정보").mkdir(parents=True)
    monkeypatch.setattr(rb, "RESULTS", results)
    monkeypatch.setattr(rb, "OUT_DIR", results / "_재처리")
    monkeypatch.setattr(rb, "INFO_DIR", results / "_재처리" / "_책정보")
    base = tmp_path / "storage"
    monkeypatch.setattr(storage_manager, "_BASE", base)
    monkeypatch.setattr(storage_manager, "_UPLOADS", base / "uploads")
    monkeypatch.setattr(storage_manager, "_RESULTS", base / "results")
    monkeypatch.setattr(storage_manager, "_HISTORY", base / "history.json")
    return results


def _first_pass(results, tmp_path):
    """1차 직후의 일을 그대로 — 쪽 교정 → 정제 → (일괄 처리의) 조사 자동 교정 → 02_결과 MD · 원문 사본."""
    pages = copy.deepcopy(_raw_pages())
    ocr_gradio._correct_pages(pages)
    gen = ocr_gradio._refine(pages, False, OutputFormat.MD, [], {"name": "경영의기본_홍길동.pdf", "pages": len(pages)})
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        md = stop.value[1]
    fixed, n = ocr_typos.auto_fix(md)
    md = fixed if n and len(fixed) == len(md) else md
    result = results / f"{STEM}.md"
    result.write_text(md, encoding="utf-8")
    raw = writer.write(FileResult(source_path=Path("경영의기본_홍길동.pdf"), output_path=None, pages=pages),
                       tmp_path / "raw", OutputFormat.MD)
    storage_manager.save_raw_copy(raw, result)
    return result


def _save_page_file(result, **options):
    return page_store.save(storage_manager.page_file_path(result), _raw_pages(),
                           {"source": "경영의기본_홍길동.pdf", "total_pages": 5, "options": options})


def test_page_file_book_matches_the_stored_result_path(env, tmp_path):
    result = _first_pass(env, tmp_path)
    out, rep = rb.reprocess(STEM)                       # 쪽 원문 파일 없음 — 지금 경로(02_결과 의 정제본)
    assert rep["from_pages"] is False
    old = out.read_text(encoding="utf-8")
    assert "경영의 첫째 과제다" in old

    _save_page_file(result)
    result.write_text("엉뚱한 글 — 쪽 원문 파일이 있으면 이 파일은 읽지 않는다", encoding="utf-8")
    out, rep = rb.reprocess(STEM)                       # 쪽 원문 파일에서 정제부터 다시
    assert rep["from_pages"] is True
    assert out.read_text(encoding="utf-8") == old       # 글자 단위로 같다


def test_particle_auto_fix_runs_as_in_batch(env, monkeypatch):
    _save_page_file(env / f"{STEM}.md")
    monkeypatch.setattr(ocr_typos, "auto_fix", lambda t: (t.replace("성과롤", "성과를"), 1))
    out, _ = rb.reprocess(STEM)
    assert "성과를 내는" in out.read_text(encoding="utf-8")
    # 글자 수가 달라지는 교정은 쓰지 않는다(일괄 처리와 같은 안전장치)
    monkeypatch.setattr(ocr_typos, "auto_fix", lambda t: (t.replace("성과롤", "성과를을"), 1))
    out, _ = rb.reprocess(STEM)
    assert "성과롤 내는" in out.read_text(encoding="utf-8")


def test_front_matter_choice_comes_from_the_page_file(env, monkeypatch):
    _save_page_file(env / f"{STEM}.md", front_matter=True)
    seen = []
    real = ocr_gradio._refine

    def spy(pages, front_matter, *a, **k):
        seen.append(front_matter)
        return real(pages, front_matter, *a, **k)
    monkeypatch.setattr(ocr_gradio, "_refine", spy)
    rb.reprocess(STEM)
    assert seen == [True]


def test_book_read_without_cleanup_gets_page_marks_from_the_page_file(env):
    """정제를 끄고 읽은 책은 원문 사본이 없다 — 쪽 표지는 쪽 원문 파일의 쪽으로 넣는다."""
    _save_page_file(env / f"{STEM}.md", cleanup=False)
    out, rep = rb.reprocess(STEM)
    text = out.read_text(encoding="utf-8")
    assert rep["from_pages"] is True
    assert "PDF 5쪽" in text and "[p.3]" in text


def test_reports_when_no_refined_text_comes_out(env, monkeypatch):
    _save_page_file(env / f"{STEM}.md")

    def nothing(pages, front_matter, out_fmt, logs, info, on_chunk=None):
        logs.append("  ⚠️ 문장 정제 미적용 — 정제 후 한글이 40% 줄어(분량 급감) 정제 전 원문으로 저장")
        return None, None, {}, ""
        yield                                        # 제너레이터로 만든다
    monkeypatch.setattr(ocr_gradio, "_refine", nothing)
    with pytest.raises(ValueError, match="분량 급감"):
        rb.reprocess(STEM)
