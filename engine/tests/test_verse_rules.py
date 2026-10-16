"""장절 책 정제 규칙(R-15) 회귀 점검 — 2026-10-16 king_james_bible.pdf 실측 사례."""
from core import text_cleaner, verse_rules as vr

# 디지털 PDF 원문 쪽 모양: 두 단 줄바꿈, 줄 끝 공백 = 낱말 경계, 쪽 머리글('창세기 1')·쪽번호
PAGE1 = "\n".join([
    "기원이라 하는 모세의 첫째 책",
    "창 세 기",
    "기록자: 모세, 기록 시기: 주전 1450년경, 성격: 역사 ",    # 책 해설은 한 단 넓은 줄
    "제 1 장",
    "하나님의 6일 특별 창조",
    "(창2:4­9; 욥38:4­11) ",
    "1 처음에 하나님께서 하늘과 땅을 ",
    "창조하시니라.",
    "2 땅은 형태가 없고 비어 있으며 ",
    "어둠은 깊음의 표면 위에 있고 하나님",
    "의 영은 물들의 표면 위에서 움직이시",
    "니라.",
    "3 하나님께서 이르시되, 빛이 있으",
    "라, 하시매 빛이 있었고",
])
PAGE2 = "\n".join([
    "창세기 1",
    "2",
    "4 하나님께서 빛을 보시니 좋았더",
    "라.",
    "5",                                    # 절 번호만 한 줄
    "하나님께서 빛을 낮이라 부르시니라. ",
    "안식일 제정",                            # 절 뒤 소제목
    "6 땅과 하늘들이 창조된 때 곧 주 ",
    "하나님께서 그것들을 만들고",
    "7 들의 모든 초목이 땅에 있기 전에 ",
    "제 2 장",
    "1 이같이 하늘들과 땅이 완성되니라.",
    "제 3 장",
    "1 이제 뱀은 간교하더라.",
])


def test_verse_book_structure():
    pages = [PAGE1, PAGE2]
    assert vr.is_verse_book(pages)
    text, st = vr.format_book(pages)
    lines = text.splitlines()
    assert "제 1 장" in lines and "제 2 장" in lines            # 장 머리 보존
    assert "(창2:4-9; 욥38:4-11)" in lines                       # 관주 보존, 범위 하이픈 복원
    assert "2 땅은 형태가 없고 비어 있으며 어둠은 깊음의 표면 위에 있고 하나님의 영은 물들의 표면 위에서 움직이시니라." in lines
    assert "4 하나님께서 빛을 보시니 좋았더라." in lines          # 쪽 머리글·쪽번호 제거, 한 음절 줄 보존
    assert "5 하나님께서 빛을 낮이라 부르시니라." in lines
    assert "안식일 제정" in lines
    assert "6 땅과 하늘들이 창조된 때 곧 주 하나님께서 그것들을 만들고" in lines   # 다음 절이 붙지 않음
    assert "창세기 1" not in text and st["headers"] == 2
    assert st["chapters"] == 3 and st["verses"] == 9


def test_numbered_quote_book_is_not_verse_book():
    # 명언집: '제1장' 뒤에도 항목 번호가 이어진다 → 장절 책 아님
    pages = ["제1장\n52 우리의 성공\n53 지금은", "제2장\n54 시간을\n55 실패는", "제3장\n56 성공은\n57 오늘"]
    assert not vr.is_verse_book(pages)


def test_iter_clean_uses_raw_pages_for_verse_book(monkeypatch):
    monkeypatch.setattr(text_cleaner.config, "get_local_cleanup_engine", lambda: "rules")
    gen = text_cleaner.iter_clean(["줄 이은 본문"], local=True, raw_pages=[PAGE1, PAGE2])
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        out = stop.value
    assert "제 1 장\n하나님의 6일 특별 창조\n(창2:4-9; 욥38:4-11)\n1 처음에" in out
    assert "창세기\n" in out                                       # '창 세 기' 띄어 쓴 제목 결합
