"""붙어 쓴 한글 띄어쓰기 교정(R-17) 회귀 점검 — 2026-10-17 대한민국CEO명언집·보관 원문 24권 실측."""
import pytest

from core import spacing

pytestmark = pytest.mark.skipif(not spacing.is_available(), reason="kiwipiepy 미설치")


def test_glued_words_are_spaced():
    text = "인류에게하나밖에없는 진정한 원칙은 정의다.\n정의가 없는 용기는나약하기짝이 없다"
    out, n = spacing.fix_spacing(text)
    assert out == "인류에게 하나밖에 없는 진정한 원칙은 정의다.\n정의가 없는 용기는 나약하기 짝이 없다"
    assert n == 4


def test_never_removes_spaces_or_changes_letters():
    # Kiwi 단독은 '배워야 한다'를 '배워야한다'로 붙였다 — 넣기만 한다
    text = "빨리 배워야 한다. 옮긴이 최은경은 서울에서 책상 위에"
    assert spacing.fix_spacing(text) == (text, 0)


def test_hangul_glued_to_latin_name_is_left_alone():
    # OCR 이 잘못 읽은 저자 이름('엔드루카네기Andrew')을 쪼개지 않는다
    text = "엔드루카네기Andrew Caregle, 1835-1919."
    assert spacing.fix_spacing(text) == (text, 0)
