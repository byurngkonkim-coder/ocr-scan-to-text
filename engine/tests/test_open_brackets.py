"""짝 없는 '[' — 낫표 「」『』 오인식 복원 (2천년의강의_사마천의생각경영법 실측, 2026-11-07).
여는 낫표가 '[' 로, 닫는 낫표가 ','·띄어쓰기로 읽히거나 사라졌다. 「」 로 통일해 짝을 맞춘다."""
import pytest

from core import text_cleaner as tc


@pytest.mark.parametrize("src,want", [
    ("[이사열전", "「이사열전」"),
    ("[사기, 에 바쳐진 우리 시대의 허사", "「사기」에 바쳐진 우리 시대의 허사"),
    ("사마천의[사기는 삼황오제부터 다룬다.", "사마천의「사기」는 삼황오제부터 다룬다."),
    ("말한다.[사기열전 에서 이것은 변한다.", "말한다.「사기열전」에서 이것은 변한다."),
    ("쓴[장의 열전 첫머리는 인상적이다.", "쓴「장의 열전」 첫머리는 인상적이다."),
    ("우리에게[폭력과 성스러움의 저자로 알려진 그", "우리에게「폭력과 성스러움」의 저자로 알려진 그"),
    ("통해[적과 혹[죄와 벌 등의 소설을 분석했다.", "통해「적과 혹」「죄와 벌」 등의 소설을 분석했다."),
    ("모든 문학과[시경, 과[서경, 제자백가의 책", "모든 문학과「시경」과「서경」, 제자백가의 책"),
    ("그가 남긴[회사라는 시에는 굴원의 원망과 분노, 스스로의 삶", "그가 남긴「회사」라는 시에는 굴원의 원망과 분노, 스스로의 삶"),
    ("뼈 위에 난 균열[을 보고 말한다.", "뼈 위에 난 균열을 보고 말한다."),
])
def test_unmatched_open_bracket_repaired(src, want):
    assert tc.repair_open_brackets(src)[0] == want


@pytest.mark.parametrize("src", ["[그림 1] 매출 추이", "[p.12]", "| [a | b |", "참고: [1] 원문"])
def test_matched_or_marker_brackets_kept(src):
    assert tc.repair_open_brackets(src)[0] == src


# ── 조사로 시작하는 문단·조사 뒤 숫자·둥근 따옴표 (2천년의강의, 2026-11-07) ─────────────────────

def test_paragraph_starting_with_josa_joins_previous():
    src = "당신은 진정한 ‘큰생각'\n\n을 만나게 될 것이다.\n\n이 책은 다르다."
    out, n = tc.join_josa_paragraphs(src)
    assert n == 1 and out == "당신은 진정한 ‘큰생각'을 만나게 될 것이다.\n\n이 책은 다르다."


def test_sentence_end_or_heading_not_joined():
    src = "# 머리말\n\n을 이라는 글자.\n\n앞 문장이 끝났다.\n\n의 뜻."
    assert tc.join_josa_paragraphs(src) == (src, 0)


@pytest.mark.parametrize("src,want", [("당신은2000년 간", "당신은 2000년 간"), ("진정한‘큰생각'", "진정한 ‘큰생각'"),
                                      ("제2장 시작", "제2장 시작"), ("약3배", "약3배")])
def test_glued_digit_and_curly_quote_spaced(src, want):
    assert tc.space_glued_marks(src)[0] == want


@pytest.mark.parametrize("src,want", [
    ("영원한 주제인 ‘권력'과'조직'에 관한 통찰", "영원한 주제인 ‘권력'과 '조직'에 관한 통찰"),
    ("그는'삼각형의 이론'을 통해", "그는 '삼각형의 이론'을 통해"),
    ('그는"좋다"고 말했다', '그는 "좋다"고 말했다'),
    ("홀수 '따옴표만'있는 줄은'그대로", "홀수 '따옴표만'있는 줄은'그대로"),
    ("don't 와 it's", "don't 와 it's"),
])
def test_space_before_opening_quote(src, want):
    assert tc.space_glued_marks(src)[0] == want


def test_doubled_open_quote_becomes_inner_single_quote():
    # 바깥 큰따옴표 + 『 를 '"' 로 읽은 겹따옴표 — 안쪽 인용은 작은따옴표(사용자 요청 2026-11-07)
    src = '구본형 변화 경영 연구소 소장은""사기열전, 에 등장하는 사람들은 모두 사마천의 분신이며"'
    out = tc.space_glued_marks(tc.repair_open_brackets(src)[0])[0]
    assert out == '구본형 변화 경영 연구소 소장은 "\'사기열전\'에 등장하는 사람들은 모두 사마천의 분신이며"'


@pytest.mark.parametrize("src,want", [("고백하기도했다", "고백하기도 했다"), ("쓰이기도 한다", "쓰이기도 한다"),
                                      ("하나님께 기도했다", "하나님께 기도했다"), ("나타나기도한다", "나타나기도한다")])
def test_auxiliary_gido_spaced(src, want):
    # '나타나기도한다' 는 줄기 목록 밖이라 그대로 — 넓히려면 목록에 더한다
    assert tc.space_glued_marks(src)[0] == want


@pytest.mark.parametrize("src,want", [
    ("유일한 조직은[변화 주도자(change leader)뿐이다.", "유일한 조직은「변화 주도자」(change leader)뿐이다."),
    ("신용 카드는[법화(法貨)라고 불리던", "신용 카드는「법화」(法貨)라고 불리던"),
])
def test_term_before_gloss_parenthesis(src, want):
    assert tc.repair_open_brackets(src)[0] == want
