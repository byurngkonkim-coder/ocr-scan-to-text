"""korean_corrector 회귀 테스트.

Regression (2026-10-10): 66/99 따옴표 교정이 숫자까지 바꾸고('1999 년'→'19" 년'),
'인해/대해'·한 글자 관형사 분리가 멀쩡한 단어를 쪼갰다('확인해'→'확 인해').
기존 검증은 성공 사례만 봐서 놓쳤다 — 망가뜨리면 안 되는 입력을 함께 고정한다.
MyOCR/engine/korean_corrector.py 와 같은 파일이며 테스트도 같다.
"""
import pytest

from core.korean_corrector import refine_korean_text

KEEP = [
    "확인해 주세요", "결재를 승인해 주십시오", "총 66 명이 참석", "Route 66 Highway",
    "1999 년 설립", "가격 1,099.", "수량99", "저장되었습니다", "이용자관리화면",
    "무등록 업체", "무등산", "넙적다리", "거대해서", "변수없이", "말이 통한다고",
    "공부하는구나",
]

FIX = {
    '66안녕하세요.99': '"안녕하세요."',
    "할수있다": "할 수 있다",
    "이를위해": "이를 위해",
    "그로인해": "그로 인해",
    "먹을것이다": "먹을 것이다",
    "않되는": "안 되는",
    "훌륨한": "훌륭한",
    "위로하는그림하루를다독이는명언": "위로하는 그림 하루를 다독이는 명언",
}


@pytest.mark.parametrize("text", KEEP)
def test_preserves(text):
    assert refine_korean_text(text) == text


@pytest.mark.parametrize("src, want", FIX.items())
def test_fixes(src, want):
    assert refine_korean_text(src) == want
