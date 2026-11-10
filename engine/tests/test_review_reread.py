"""'검토 필요 쪽만 다시 읽기' — 그 쪽만 Vision 으로 바꾸고 나머지 쪽은 건드리지 않는다 (2026-10-15)."""
from pathlib import Path
from types import SimpleNamespace

import pytest

ocr_gradio = pytest.importorskip("ocr_gradio")
from core import storage_manager
from core.types import FileResult, OCREngine, OCRResult, OutputFormat, PageResult

PNG = Path(__file__).parent / "sample_docs" / "sample_kor_eng.png"


def _page(n, text, review):
    return PageResult(page_num=n, raw_text=text, corrected_text=text, engine=OCREngine.RAPIDOCR,
                      confidence=50.0 if review else 95.0, needs_review=review,
                      review_reason="신뢰도 낮음" if review else "")


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr_gradio, "RESULTS_DIR", tmp_path / "results")
    for name, sub in (("_BASE", ""), ("_UPLOADS", "uploads"), ("_RESULTS", "results"),
                      ("_HISTORY", "history.json")):
        monkeypatch.setattr(storage_manager, name, tmp_path / "storage" / sub)
    calls = []

    def fake_run(img, lang_hint="kor+eng"):
        calls.append(1)
        return OCRResult(text="비전으로 다시 읽은 본문입니다.", confidence=95.0,
                         engine=OCREngine.GEMINI, has_jamo_split=False)

    monkeypatch.setattr(ocr_gradio.claude_vision, "run", fake_run)
    monkeypatch.setattr(ocr_gradio.claude_vision, "is_available", lambda: True)
    monkeypatch.setattr(ocr_gradio, "_render_page", lambda f, n: None)
    result = FileResult(source_path=PNG, output_path=None)
    result.pages = [_page(1, "로컬로 잘 읽은 첫 쪽입니다.", False), _page(2, "깨 진 글 ㅈ", True)]
    ocr_gradio._LAST_REVIEW.clear()
    ocr_gradio._LAST_REVIEW.update(file=str(PNG), result=result, cleanup=False, front_matter=False,
                                   out_fmt=OutputFormat.TXT, lang="kor+eng", total=2)
    return calls


def test_rereads_only_review_pages(env):
    status, _preview, path, _btn = ocr_gradio.reread_review_pages(SimpleNamespace(name=str(PNG)))
    assert len(env) == 1                                   # 검토 필요 1쪽만 호출
    out = Path(path).read_text(encoding="utf-8")
    assert "비전으로 다시 읽은" in out and "로컬로 잘 읽은 첫 쪽" in out
    assert "1쪽 교체" in status and "남은 검토 필요 0쪽" in status
    assert not ocr_gradio._LAST_REVIEW                     # 다 고쳤으면 버튼 상태도 비움


def test_button_shows_count_and_cost(env):
    upd = ocr_gradio._review_btn_update()
    assert upd["visible"] and "1쪽" in upd["value"] and "$" in upd["value"]


def test_other_file_is_refused(env):
    status, *_ = ocr_gradio.reread_review_pages(SimpleNamespace(name="다른파일.pdf"))
    assert "먼저 전체 처리" in status and not env


def test_state_cleared_mid_reread_does_not_lose_paid_result(env, monkeypatch):
    # 다시 읽는 도중 ② 일괄 작업이 책을 끝내며 _LAST_REVIEW 를 비우면 KeyError 로 유료 결과를 잃었다
    def run_and_clear(img, lang_hint="kor+eng"):
        ocr_gradio._LAST_REVIEW.clear()
        return OCRResult(text="비전으로 다시 읽은 본문입니다.", confidence=95.0, engine=OCREngine.GEMINI)
    monkeypatch.setattr(ocr_gradio.claude_vision, "run", run_and_clear)
    status, _p, path, _b = ocr_gradio.reread_review_pages(SimpleNamespace(name=str(PNG)))
    assert "1쪽 교체" in status and "비전으로 다시 읽은" in Path(path).read_text(encoding="utf-8")


def test_reread_with_cleanup_is_refined_like_a_full_run(env, monkeypatch):
    """다시 읽은 뒤의 정제도 전체 처리와 같은 함수(_refine)로 — 예전엔 이 경로만 차례 처리·문서 정보가 빠졌다
    (intent 2026-11-10 정제-후처리-통합)."""
    ocr_gradio._LAST_REVIEW.update(cleanup=True, out_fmt=OutputFormat.MD)
    seen = []
    real = ocr_gradio._refine

    def spy(pages, *a, **k):
        seen.append([p.page_num for p in pages])
        return real(pages, *a, **k)
    monkeypatch.setattr(ocr_gradio, "_refine", spy)
    _status, _preview, path, _btn = ocr_gradio.reread_review_pages(SimpleNamespace(name=str(PNG)))
    out = Path(path).read_text(encoding="utf-8")
    assert seen == [[1, 2]]
    assert out.startswith("> **문서 정보**") and "비전으로 다시 읽은" in out and "로컬로 잘 읽은 첫 쪽" in out


def test_reread_writes_page_file_for_the_new_result(env):
    """다시 읽은 쪽(유료)이 쪽 원문 파일에도 들어가야, 나중에 후처리를 다시 돌려도 그 글자가 남는다."""
    from core import page_store
    ocr_gradio._LAST_REVIEW.update(page_meta={"source": PNG.name, "total_pages": 2,
                                              "options": {"cleanup": False, "llm_correct": False}})
    _status, _preview, path, _btn = ocr_gradio.reread_review_pages(SimpleNamespace(name=str(PNG)))
    pages, meta = page_store.load(storage_manager.page_file_path(path))
    assert [p.raw_text for p in pages] == ["로컬로 잘 읽은 첫 쪽입니다.", "비전으로 다시 읽은 본문입니다."]
    assert pages[1].engine == OCREngine.GEMINI and not pages[1].needs_review
    assert all(p.corrected_text == "" for p in pages)            # 교정본은 후처리가 판독 원문에서 다시 만든다
    assert meta["source"] == PNG.name


def test_batch_job_does_not_touch_tab1_review_state(env):
    import threading
    before = dict(ocr_gradio._LAST_REVIEW)
    t = threading.Thread(target=lambda: list(ocr_gradio.process_full(
        SimpleNamespace(name=str(PNG)), "kor+eng", "txt", True, True, True, False, 0, "", "RapidOCR 우선",
        cleanup=False, progress=lambda *a, **k: None)), name="batch-runner-9")
    t.start()
    t.join(120)
    assert ocr_gradio._LAST_REVIEW == before
