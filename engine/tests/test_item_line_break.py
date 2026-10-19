"""번호·구분자로 시작하는 항목 줄은 앞 줄에 잇지 않는다 — 스캔 OCR·디지털 PDF·txt 정제 공통."""
import pytest

from core import postprocessor as pp
from core import text_cleaner

_FILL = "이 문장은 오른쪽 끝까지 찬 본문 줄로 다음 줄과 이어질 수 있는 길이입니다"

ITEMS = ["1. 첫째 항목", "2) 둘째 항목", "(3) 셋째 항목", "① 첫째 항목",
         "- 기호 항목", "• 기호 항목", "▶ 기호 항목", "※ 참고 사항", "■ 기호 항목",
         "제1조 (목적)", "제2장 총칙"]


@pytest.mark.parametrize("item", ITEMS)
def test_item_line_not_joined(item):
    out = pp.join_broken_lines(f"{_FILL}\n{item}\n{_FILL}").splitlines()
    assert item in out


@pytest.mark.parametrize("a,b", [("가.", "나."), ("가)", "나)"), ("(가)", "(나)")])
def test_korean_items_not_joined(a, b):
    out = pp.join_broken_lines(f"{_FILL}\n{a} {_FILL}\n{b} {_FILL}").splitlines()
    assert len(out) == 3 and out[2].startswith(b)


def test_lone_korean_syllable_wrap_still_joined():
    # '…운용되었|다. 그러나' — 목록('나.' 등)이 없으면 한글 한 글자+마침표는 잘린 낱말 끝이다
    assert len(pp.join_broken_lines(f"{_FILL} 운용되었\n다. 그러나 일반 도로에서는").splitlines()) == 1


@pytest.mark.parametrize("cont", ["1.5배 늘었다고 한다", "가나다 순으로 정렬했다", "제3조에 따라 처리한다"])
def test_plain_continuation_still_joined(cont):
    # 숫자·'가'·'제N조' 로 시작해도 항목 표시가 아니면 예전처럼 잇는다
    assert len(pp.join_broken_lines(f"{_FILL}\n{cont}").splitlines()) == 1


_PB = text_cleaner.PAGEBREAK


@pytest.mark.parametrize("nxt", ["4\n지난 일은 지나가게 두라\n  사람들은 자주 가벼운 실수를 한다.",
                                 "지난 일은 지나가게 두라\n  사람들은 자주 가벼운 실수를 한다.",
                                 "  사람들은 자주 가벼운 실수를 한다."])
def test_heading_or_indent_after_pagebreak_not_glued(nxt):
    # 인생잠언 21쪽: 장 끝 요약 문구가 마침표 없이 끝나 다음 쪽 장 번호·소제목이 그 끝에 붙었다
    out = text_cleaner.post_process_pagebreaks(f"같은 실수를 되풀이하지 않는 방법을 터득하는 것이다\n{_PB}\n{nxt}")
    assert out.splitlines()[0] == "같은 실수를 되풀이하지 않는 방법을 터득하는 것이다"


@pytest.mark.parametrize("nxt", ["고 하는 데에는 다른 이유가 있다.\n  이런 삶의 지혜들이",
                                 "져나갈 궁리를 한다\n  룰플레이어가 되는 것도",
                                 "말하면서 고민을 늘어놓는 순간 상대는 어떻게든 빨리 그 상황에서 빠\n져나갈 궁리를 한다."])
def test_continuation_after_pagebreak_still_glued(nxt):
    out = text_cleaner.post_process_pagebreaks(f"그들은 잘 지낸다는 대답만을 듣고 싶어한다 당신이 힘들다\n{_PB}\n{nxt}")
    assert out.splitlines()[0].startswith("그들은") and nxt.split()[0] in out.splitlines()[0]


@pytest.mark.parametrize("item", ["2. 둘째 항목", "② 둘째 항목", "제2조 (정의)"])
def test_item_after_pagebreak_not_glued(item):
    out = text_cleaner.post_process_pagebreaks(f"첫째 항목 설명이 이어지는 줄\n{text_cleaner.PAGEBREAK}\n{item}")
    assert item in out.splitlines()
