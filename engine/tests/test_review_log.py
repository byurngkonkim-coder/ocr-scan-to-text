"""최종 검수 기록 — 책마다 OCR 원문 → 쪽 교정 → 최종 정제본 글자 수와 정제 보고를 남긴다
(intent 2026-10-18-정제-순서-재배치 5단계)."""
from core import review_log


def test_counts_character_classes():
    c = review_log.count("공자(孔子)는 BC 551년에 태어났다.")
    assert c == {"한글": 9, "한자": 2, "숫자": 3, "영문": 2, "어절": 4}


def test_save_writes_stage_table_and_warnings(tmp_path):
    result = tmp_path / "20261018_120000_장원.txt"
    result.write_text("최종 본문\n", encoding="utf-8")
    path = review_log.save(result, raw_pages=["최 종 본문 23"], page_pages=["최종 본문"],
                           final_text="최종 본문", warnings=["붙어 쓴 낱말 띄어쓰기 3곳 교정"])
    assert path == tmp_path / "_검수기록" / "20261018_120000_장원.txt"
    body = path.read_text(encoding="utf-8")
    assert "판독 원문" in body and "쪽 교정·잡음 제거 후" in body and "최종 정제본" in body
    assert "| 판독 원문 | 4 | 0 | 2 | 0 | 4 |" in body
    assert "| 원문→최종 변화 | 0 | 0 | -2 | 0 | -2 |" in body
    assert "- 붙어 쓴 낱말 띄어쓰기 3곳 교정" in body
    assert "쪽 수: 1" in body


def test_ui_saves_review_and_reports_line(tmp_path):
    import ocr_gradio
    from core.types import OCREngine, PageResult
    result = tmp_path / "20261018_책.txt"
    pages = [PageResult(page_num=1, raw_text="원 문", corrected_text="원문", engine=OCREngine.RAPIDOCR,
                        confidence=99.0)]
    note = ocr_gradio._save_review(result, pages, "원문", {"warnings": ["띄어쓰기 1곳 교정"]})
    assert note == "   🧾 검수기록: _검수기록/20261018_책.txt"
    assert "- 띄어쓰기 1곳 교정" in (tmp_path / "_검수기록" / "20261018_책.txt").read_text(encoding="utf-8")
    assert ocr_gradio._save_review(None, pages, "원문", {}) == ""     # 02_결과 저장 실패
    assert ocr_gradio._save_review(result, pages, "", {}) == ""       # 정제 안 함


def test_save_without_warnings_says_none(tmp_path):
    result = tmp_path / "책.txt"
    path = review_log.save(result, raw_pages=["가"], page_pages=["가"], final_text="가", warnings=[])
    assert "- 없음" in path.read_text(encoding="utf-8")
