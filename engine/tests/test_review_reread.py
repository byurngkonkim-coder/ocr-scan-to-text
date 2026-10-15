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
