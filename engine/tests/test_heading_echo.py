"""장 표지의 큰 글자를 OCR 이 따로 읽어 제목 바로 아래에 제목 끝말이 한 번 더 남은 줄을 지운다
(4차산업혁명시대 2026-11-07 — '## 1장 혁신 전쟁의 서막' 아래 '서막', '## 3장 미래예측모델 만들기' 아래 '만들기',
'## 4장 …새로운 산업 트렌드' 아래 '산업 트렌드 오픈 이노베이션(…)' 처럼 용어 상자 제목에 붙은 것)."""
from core import text_cleaner as tc


def test_echo_line_dropped():
    out, n = tc.drop_heading_echo("## 1장 혁신 전쟁의 서막\n\n서막\n\nICBM\n\n본문.")
    assert n == 1 and out == "## 1장 혁신 전쟁의 서막\n\nICBM\n\n본문."


def test_echo_prefix_stripped_from_short_line():
    text = "## 4장 4차 산업혁명의 새로운 산업 트렌드\n\n산업 트렌드 오픈 이노베이션(개방형 혁신,Open Innovation)\n\n뜻."
    out, n = tc.drop_heading_echo(text)
    assert n == 1 and "\n\n오픈 이노베이션(개방형 혁신,Open Innovation)\n\n" in out


def test_body_sentence_starting_with_title_word_kept():
    # 본문 문장이 제목 끝말로 시작해도 그대로 — 문장 끝(다.)이 있거나 길면 본문이다
    for body in ["서막은 이미 올랐다.", "만들기 위해서는 먼저 데이터를 모아야 한다. 그 다음 단계는 이렇다."]:
        text = f"## 3장 미래예측모델 만들기\n\n{body}"
        assert tc.drop_heading_echo(text) == (text, 0)


def test_label_line_above_same_numbered_heading_dropped():
    # 부·장 표지 번호만 따로 남은 줄(Alianment정렬 '1부' / '# 1부 정렬의 중요성 이해', 'CHAPTER9' / '## 9장 …')
    text = "앞 글.\n\n1부\n\n# 1부 정렬의 중요성 이해\n\n본문.\n\nCHAPTER9\n\n## 9장 정렬 프로세스의 관리\n\n본문."
    out, n = tc.drop_heading_echo(text)
    assert n == 2 and out == "앞 글.\n\n# 1부 정렬의 중요성 이해\n\n본문.\n\n## 9장 정렬 프로세스의 관리\n\n본문."


def test_label_line_with_other_number_kept():
    text = "2부\n\n# 1부 정렬의 중요성 이해"
    assert tc.drop_heading_echo(text) == (text, 0)


def test_heading_text_only_line_not_counted_twice():
    # 제목 전체가 다시 나온 것은 다른 규칙(차례 조각) 몫 — 끝말 한두 낱말만 본다
    text = "## 2장 9가지 미래전략산업\n\n인공지능 2020\n\n본문."
    assert tc.drop_heading_echo(text) == (text, 0)
