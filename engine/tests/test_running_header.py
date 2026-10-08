"""러닝헤더 결정론 제거(strip_running_headers) 테스트.

Regression: QA 2026-10-03 — 정제 LLM은 청크(약 4000자)만 보므로 책 전체에서
반복되는 머리글/꼬리말('Chapter 1 새벽, …', '하루 명화 하루 명언')을 알아보지
못해 본문 문단 사이에 그대로 남았다(하루명화 약 38건·드러커 14건·FocalPoint 23건).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import text_cleaner

H = "Chapter 1 새벽, 잠 못 이루는 당신에게"


def _pages(n, footer=H):
    return [f"본문 {i}쪽 문장입니다.\n{footer}" for i in range(n)]


def test_repeated_footer_removed_except_first():
    out = text_cleaner.strip_running_headers(_pages(6))
    assert out[0].endswith(H)                       # 첫 등장(장 시작·표제)은 보존
    assert all(H not in p for p in out[1:])
    assert all("문장입니다." in p for p in out)      # 본문은 그대로


def test_top_header_removed():
    pages = [f"하루 명화 하루 명언\n본문 {i}." for i in range(6)]
    out = text_cleaner.strip_running_headers(pages)
    assert out[1] == "본문 1."


def test_below_threshold_untouched():
    pages = _pages(4)                                # 5쪽 미만 반복 → 장 제목일 수 있어 보존
    assert text_cleaner.strip_running_headers(pages) == pages


def test_page_number_on_header_line():
    pages = [f"본문 {i}.\n{20 + i} 한 가지로 승부하라" for i in range(6)]
    out = text_cleaner.strip_running_headers(pages)
    assert all("승부하라" not in p for p in out[1:])


def test_single_word_with_number_not_collapsed():
    # 'Chapter 1'·'누가복음 5' 처럼 숫자를 떼면 한 단어만 남는 줄은 묶지 않는다
    # (차례·장 번호·성경 장 머리글 오삭제 방지 — 성경 머리글은 LLM 규칙이 처리)
    pages = [f"Chapter {i}\n본문 {i}." for i in range(1, 8)]
    assert text_cleaner.strip_running_headers(pages) == pages


def test_middle_line_untouched():
    pages = [f"위 {i}.\n{H}\n아래 {i}." for i in range(6)]
    assert text_cleaner.strip_running_headers(pages) == pages


def test_tool_placeholder_untouched():
    pages = [f"본문 {i}.\n[그림 1: 이미지 감지됨]" for i in range(6)]
    assert text_cleaner.strip_running_headers(pages) == pages


def test_iter_clean_applies_it(monkeypatch):
    seen = []
    monkeypatch.setattr(text_cleaner._gemini, "run_text",
                        lambda p: seen.append(p) or p.split("---")[1])
    text_cleaner.clean_pages(_pages(6))
    assert seen and seen[0].count(H) == 1
