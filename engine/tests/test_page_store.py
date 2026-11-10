"""쪽 원문 파일 — 1차(글자 읽기)가 남기는 쪽 단위 결과를 잃는 것 없이 저장하고 다시 읽는다.

Intent: docs/intents/2026-11-10-정제-후처리-통합.md
"""
import pytest

from core import page_store, storage_manager
from core.types import OCREngine, PageResult


def _pages():
    return [
        PageResult(page_num=1, raw_text="제1장\n  첫 줄은 들여 썼다\n둘째 줄 ⟨TABLE:1-1⟩", corrected_text="",
                   engine=OCREngine.RAPIDOCR, confidence=97.5, figures=["[표 1]\n| 가 | 나 |\n|---|---|\n| 1 | 2 |"]),
        PageResult(page_num=2, raw_text="", corrected_text="", engine=OCREngine.RAPIDOCR, confidence=0.0,
                   is_blank=True, blank_reason="흰 쪽"),
        PageResult(page_num=3, raw_text="흐린 쪽", corrected_text="", engine=OCREngine.GEMINI, confidence=61.0,
                   needs_review=True, review_reason="신뢰도 낮음 (61.0%)"),
    ]


def test_round_trip_keeps_every_field(tmp_path):
    path = page_store.save(tmp_path / "책_쪽원문.json", _pages(),
                           {"source": "책.pdf", "total_pages": 3, "options": {"front_matter": True}})
    pages, meta = page_store.load(path)
    assert pages == _pages()                      # 줄바꿈·들여쓰기·표 자리 표지·그림표·검토 사유까지 같다
    assert meta["source"] == "책.pdf" and meta["total_pages"] == 3
    assert meta["options"] == {"front_matter": True}


def test_corrected_text_kept_only_where_present(tmp_path):
    """교정본이 이미 있는 쪽(유료 교정·옛 복구본에서 되살린 쪽)은 그 글자를 남긴다 — 다시 만들 수 없다.
    교정본이 비어 있는 쪽은 후처리가 판독 원문에서 다시 교정하므로 적지 않는다."""
    pages = _pages()
    pages[2].corrected_text = "유료 교정으로 고친 쪽"
    path = page_store.save(tmp_path / "책_쪽원문.json", pages)
    text = path.read_text(encoding="utf-8")
    assert text.count('"corrected_text"') == 1
    assert page_store.load(path)[0][2].corrected_text == "유료 교정으로 고친 쪽"


def test_broken_file_says_which_file(tmp_path):
    path = tmp_path / "깨진책_쪽원문.json"
    path.write_text('{"version": 1, "pages": [{"page_num": 1', encoding="utf-8")
    with pytest.raises(ValueError, match="깨진책_쪽원문.json"):
        page_store.load(path)


def test_save_leaves_no_partial_file(tmp_path):
    path = page_store.save(tmp_path / "책_쪽원문.json", _pages())
    assert [p.name for p in tmp_path.iterdir()] == [path.name]      # 임시 파일이 남지 않는다


def test_page_file_sits_next_to_raw_copy_and_goes_with_the_record(tmp_path, monkeypatch):
    base = tmp_path / "storage"
    monkeypatch.setattr(storage_manager, "_BASE", base)
    monkeypatch.setattr(storage_manager, "_UPLOADS", base / "uploads")
    monkeypatch.setattr(storage_manager, "_RESULTS", base / "results")
    monkeypatch.setattr(storage_manager, "_HISTORY", base / "history.json")
    result = tmp_path / "20261110_120000_책.md"
    result.write_text("본문", encoding="utf-8")
    path = storage_manager.page_file_path(result)
    assert path == base / "uploads" / "20261110_120000_책_쪽원문.json"
    page_store.save(path, _pages())
    storage_manager._delete_entry_files({"result_path": str(result), "upload_path": "", "output_path": ""})
    assert not path.exists()                      # 처리 기록을 지우면 쪽 원문 파일도 함께 지운다
