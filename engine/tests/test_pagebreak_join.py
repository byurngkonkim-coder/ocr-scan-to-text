"""마진 필터 ON 의 페이지 경계 어절 결합(post_process_pagebreaks) 오결합.

Regression: ISSUE-H — 뒤 어절이 조사 '글자'로 시작하기만 해도(가족을·도시) 붙였고,
1음절 의존명사(수·것)도 붙였으며, \\s* 가 줄바꿈까지 먹어 쪽 경계 문단 구분이 사라졌다.
Found by /qa on 2026-10-03 · Report: .gstack/qa-reports/qa-report-ocr-2026-10-03.md
"""
import pytest

from core import text_cleaner

PB = text_cleaner.PAGEBREAK


@pytest.mark.parametrize("src, want", [
    (f"할 수{PB}있다", "할 수 있다"),
    (f"그는{PB}가족을", "그는 가족을"),
    (f"로마{PB}도시", "로마 도시"),
    (f"문단이다.\n{PB}\n둘째", "문단이다.\n\n둘째"),
    # 기존 동작 유지
    (f"성 {PB} 찰의", "성찰의"),
    (f"결정 {PB} 을 뜻합니다", "결정을 뜻합니다"),
    (f"사람{PB}에게", "사람에게"),
])
def test_join_rules(src, want):
    assert text_cleaner.post_process_pagebreaks(src) == want
