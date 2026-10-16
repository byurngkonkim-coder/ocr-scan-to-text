"""비용 우선(내장 정제규칙) OCR 잡음·단락 규칙 회귀 점검 — 2026-10-16 명언집 5권 실측 사례."""
from core import noise_rules as nr


def test_garbage_lines_removed_but_numbers_and_words_kept():
    page = "\n".join([
        "V•니 G ' ;，",                 # 장식 그림을 읽은 잡음
        "■次'系懸",
        "HMAi.乂 또,'",
        "• 272",                        # 항목 번호 — 남긴다
        "소망",                          # 짧은 소제목 — 남긴다
        "I believe in innovation.",
    ])
    out = nr.clean_page(page).splitlines()
    assert out == ["272", "소망", "I believe in innovation."]


def test_noise_tokens_stripped_inside_body_line():
    assert nr.clean_page("艾^ 하루에 한개 팔고 ᄀ 하지만 ■■■nr내게 ______ 끝이다.") \
        == "하루에 한개 팔고 하지만 nr내게 끝이다."
    # 한글 어절 앞에 붙은 한자 오인식은 떼고 어절은 남긴다
    assert nr.clean_page("一오스틴 펠프스(1820-1890), 목사") == "오스틴 펠프스(1820-1890), 목사"
    # 한자 병기는 본문
    assert nr.clean_page("옛 것을 익히는 온고지신 (溫故知新) 의 자세") == "옛 것을 익히는 온고지신 (溫故知新) 의 자세"


def test_body_line_with_symbol_junk_keeps_hangul():
    assert nr.clean_page("/1 r* xr \\ x - 나항상 선택의 갈림길이 있다") == "나항상 선택의 갈림길이 있다"


def test_fullwidth_and_broken_entry_numbers():
    assert nr.clean_page("OO52 우리의 성공（파트너십）：시작") == "0052 우리의 성공(파트너십):시작"
    assert nr.clean_page("0 0 5 1 지금은 환상적인 시기입니다") == "0051 지금은 환상적인 시기입니다"


def test_one_word_english_lines_joined():
    page = "0052 우리의 성공은 파트너십이다 Our\nsuccess\nhas\nreally\nbeen\nbased."
    assert nr.clean_page(page) == "0052 우리의 성공은 파트너십이다 Our success has really been based."


def test_garbled_running_headers_removed_without_body_loss():
    # 쪽마다 조금씩 다르게 깨져 읽힌 머리글(실측 꼴) + 쪽마다 다른 본문
    variants = ["실리콘밸리 천재들의 생각 아포리즘", "실리콘밸리 천재들의 생각 아포리증",
                "실라콘밸리 천재들의 생각 아포리즘", "실리콘밸리천재들의생각아포리즘"]
    bodies = ["기회는 준비된 사람에게 온다", "실패는 배움의 다른 이름이다", "고객이 모든 것을 결정한다",
              "혁신은 빠르게 진행되고 있다", "작게 시작해서 크게 키워라", "질문이 답을 바꾼다"]
    pages = [f"{100 + i} {variants[i % 4]}\n0{100 + i} {bodies[i % 6]} ({i})" for i in range(12)]
    pages.append("그렇지 않다면 당 74 실리콘밸리 천재들의 생각 아포리즘")
    out, n = nr.strip_fuzzy_headers(pages)
    assert n >= 12
    assert all("아포리" not in p for p in out)
    assert out[0].splitlines() == ["0100 기회는 준비된 사람에게 온다 (0)"]
    assert out[-1] == "그렇지 않다면 당"


def test_part_header_with_page_number_removed_mid_line():
    pages = ["Give me a turbulent world and I'll take the PART 3 실라콘뱅리 혁신가들의 통학 331 turbulent one."]
    out, _ = nr.strip_fuzzy_headers(pages)
    assert out[0] == "Give me a turbulent world and I'll take the turbulent one."


def test_entry_paragraphs_restored_and_page_numbers_dropped():
    body = []
    for k in range(1, 41):
        body.append(f"{k:04d} 명언 {k}번 문장이다")
        if k == 20:
            body.append("157 0021 같은 줄에 붙은 쪽번호와 다음 항목")
    text = " ".join(body).replace("0021 명언 21번", "0022 명언 22번", 0)
    out, n = nr.paragraph_entries(text)
    paras = out.split("\n\n")
    assert n >= 30
    assert paras[0] == "0001 명언 1번 문장이다"
    assert "157" not in out.split("0021")[0].split("\n\n")[-1]


def test_book_matter_rules_drop_front_matter_but_keep_prologue_and_body():
    toc = "\n".join(f"장 제목 {k}\n{k * 10}" for k in range(1, 10))
    body_entry_page = "\n".join(f"감사\n{k}\n감사하는 삶이 더 좋은 것이다." for k in range(5))
    pages = (
        ["실리콘밸리 천재들의 생각 아포리즘\nOf Silicon Valley",                      # 표지
         "저자 손호성\n아르고나인 미디어그룹 대표이사\n국내 최초로 스도쿠를 개발했다.",   # 저자 소개
         "성경, 왜 믿어야하는가 케빈 드영 지음 | 175면 | 11.000원",                    # 다른 책 광고
         "Korean Edition Copyright © 2020\n저작권법에 의하여 보호를 받는 저작물이다",    # 저작권
         "PROLOGUE\n이 책은 짧은 문장이다. 그래서 오래 남는다. 천천히 읽기 바란다.",       # 머리말 — 본문
         toc, toc]
        + [f"{k:04d} 본문 명언 {k}번째 문장은 그대로 남는다." for k in range(1, 30)]
        + [body_entry_page])                                                           # 본문 속 '번호/짧은 줄' 쪽
    notes: list[str] = []
    out = nr.drop_book_matter_rules(pages, notes)
    assert [bool(p.strip()) for p in out[:7]] == [False, False, False, False, True, False, False]
    assert all(p.strip() for p in out[7:])
    assert any("저자 소개" in n for n in notes) and any("차례" in n for n in notes)


def test_short_number_runs_are_not_paragraphed():
    text = "1984년에 3. 가지 원칙과 2 3 4 를 말했다."
    assert nr.paragraph_entries(text) == (text, 0)
