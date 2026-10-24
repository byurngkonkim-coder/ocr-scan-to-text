"""NotebookLM 소스 한도를 넘는 큰 책 나누기(intent 2026-10-24 NotebookLM용 MD 구조개선, 작업 5)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import notebooklm

_HEAD = "> **문서 정보** — 스캔본 OCR 텍스트 전문\n> - 책: 시험\n\n# 차례\n\n- 제1장 가\n- 제2장 나\n- 제3장 다\n\n"
_PARA = " ".join(["낱말"] * 40)          # 40단어 문단


def _chapter(n: int) -> str:
    return f"# 제{n}장 장 제목\n\n[p.{n}]\n" + "\n\n".join([_PARA] * 3) + "\n\n"


def _bodies(parts: list[str], head: str) -> str:
    """조각에서 '나눔' 줄·머리말을 떼고 이은 본문 — 무손실 확인용."""
    out = ""
    for p in parts:
        first, rest = p.split("\n\n", 1)
        assert first.startswith("> 나눔:")
        assert rest.startswith(head)
        out += rest[len(head):]
    return out


def test_small_book_is_not_split():
    md = _HEAD + _chapter(1)
    assert notebooklm.split_parts(md, max_words=10_000) == []


def test_split_at_chapter_headings_with_header_and_toc_in_each_part():
    body = _chapter(1) + _chapter(2) + _chapter(3)
    parts = notebooklm.split_parts(_HEAD + body, max_words=300)
    assert len(parts) == 2
    assert parts[0].startswith("> 나눔: 2개 중 1번째 파일")
    assert all(len(p.split()) <= 300 for p in parts)
    assert _bodies(parts, _HEAD) == body
    assert parts[1].split(_HEAD, 1)[1].startswith("# 제3장")     # 장 경계에서 끊는다


def test_save_parts_next_to_result(tmp_path, monkeypatch):
    """큰 책은 결과 옆 '…_NotebookLM분할' 폴더에 조각을 쓰고, 결과 파일(처리 기록 기준)은 그대로 둔다."""
    import ocr_gradio
    monkeypatch.setattr(notebooklm, "MAX_WORDS", 300)
    final = tmp_path / "20261024_120000_큰책_저자.txt"
    final.write_text("본문", encoding="utf-8")
    logs: list[str] = []
    md = _HEAD + _chapter(1) + _chapter(2) + _chapter(3)
    ocr_gradio._save_notebooklm_parts(final, md, logs)
    d = tmp_path / "20261024_120000_큰책_저자_NotebookLM분할"
    assert sorted(p.name for p in d.iterdir()) == ["20261024_120000_큰책_저자_1of2.md", "20261024_120000_큰책_저자_2of2.md"]
    assert final.read_text(encoding="utf-8") == "본문" and "2개로 나눔" in logs[0]
    ocr_gradio._save_notebooklm_parts(final, _HEAD + _chapter(1), [])      # 한도 안 — 아무것도 안 만든다
    ocr_gradio._save_notebooklm_parts(None, md, [])                         # 저장 실패 경로


def test_book_without_headings_splits_at_paragraphs():
    """제목 줄이 없는 책(장절 책 성경 등)은 빈 줄(문단) 경계에서."""
    head = "> **문서 정보** — 시험\n\n"
    body = "\n\n".join([_PARA] * 10) + "\n"
    parts = notebooklm.split_parts(head + body, max_words=150)
    assert len(parts) >= 3 and all(len(p.split()) <= 150 for p in parts)
    assert _bodies(parts, head) == body
