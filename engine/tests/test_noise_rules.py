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


def test_stacked_margin_numbers_moved_to_their_quotes():
    # 시네마명언 23쪽 실측: 쪽번호 '19'가 붙은 번호 몰림, 013 은 OCR 이 놓침, 뒤에 소제목
    text = "\n".join([
        "012 우리의 인생도 선택에 따라 달라질 수 있어.",
        "Also does our life select how to follow.",
        "PART 1 꿈과 자유를 찾아주는 명대사",
        "19 014",
        "015",
        "과거는 뒤에 둬야 앞으로 나갈 수 있어.",
        "You got to put the past behind you.",
        "제 운명은 뭐예요? / 그건 너가 알아서 해야 해.",
        "Whats my destiny? / It's up to you.",
        "저마다 운명이 있는지 아니면 그냥 떠도는 건지 모르겠어 I don't know if we each have a destiny.",
        "희망은",
        "사라지지 않는다",
    ])
    out, moved = nr.redistribute_number_stacks(text)
    lines = out.split("\n")
    assert moved == 2
    assert "014 제 운명은 뭐예요? / 그건 너가 알아서 해야 해." in lines
    assert any(l.startswith("015 저마다 운명이") for l in lines)
    assert "과거는 뒤에 둬야 앞으로 나갈 수 있어." in lines      # 놓친 013 몫 — 번호를 지어내지 않는다
    assert "19 014" not in out and "사라지지 않는다" in lines


def test_stacked_numbers_skip_previous_entry_tail_and_split_glued_numbers():
    # 실리콘밸리 25·27쪽 실측: 앞 문장 끝에 붙은 번호들, 앞 항목 꼬리(조사로 시작·영어 번역 줄)
    text = "\n".join([
        "0026 내가 17살 때, 이런 인용구를 읽은 적이 있다 만약 당신이 하루하루를 마지막 날인 0027",
        "0028",
        "니.” 일 때마다 나는 변화해야 한다고 느꼈다",
        "When I was 17, I read a quote that went something like this.",
        "여러분의 일은 여러분의 인생의 큰 부분이 될 것이고 진정으로 만족하는 방법은 하나다",
        "Your work is going to fill a large part of your life.",
        "나는 하루 이틀이 아닌 수년을 최소 단위로 생각하는 것에 익숙해졌다",
    ])
    out, moved = nr.redistribute_number_stacks(text)
    lines = out.split("\n")
    assert moved == 2
    assert lines[1] == "니.” 일 때마다 나는 변화해야 한다고 느꼈다"
    assert lines[3].startswith("0027 여러분의 일은")
    assert lines[5].startswith("0028 나는 하루 이틀이")


def test_number_line_with_trailing_letter_o_is_read_as_number():
    # 실리콘밸리 38쪽 실측: 'OO7O' 가 숫자로 안 읽혀 0068~0071 몰림이 끊겼다
    assert nr.clean_page("OO68\nOO69\nOO7O\nOO71").splitlines() == ["0068", "0069", "0070", "0071"]


def test_stack_with_too_few_quotes_does_not_skip_first_quote():
    # 실리콘밸리 80쪽 실측: 앞 문장이 끊긴 정황만으로 첫 명언을 꼬리로 보내 0201·0202 가 비었다
    text = "\n".join([
        "0197 많은 사람과 함께 일할 방법을 생각해내야",
        "0198", "0199", "0200",
        "저는 여러분이 세상을 바꾸는 일에 관심이 있습니다",
        "I'm interested in things that change the world.",
        "우리에겐 의식의 빛을 비출 의무가 있다 어떤 일이 지속할 수 있는지",
        "I think we have a duty to maintain the light.",
    ])
    out, moved = nr.redistribute_number_stacks(text)
    lines = out.split("\n")
    assert moved == 2
    assert lines[1].startswith("0198 저는 여러분이") and lines[3].startswith("0199 우리에겐")
    assert lines[-1] == "0200"          # 붙일 곳이 없는 번호는 본문 뒤에(순서 유지)


def test_lone_page_number_at_paragraph_end_removed():
    entries = [f"{k:04d} 명언 {k}번 문장이다." for k in range(200, 240)]
    entries[38] += "\n두 종류의 회사가 있다.\n91\n"         # 0238 뒤 쪽번호 + 쪽 경계 빈 줄
    out, _ = nr.paragraph_entries("\n".join(entries))
    assert "\n91" not in out and "두 종류의 회사가 있다." in out


def test_short_number_runs_are_not_paragraphed():
    text = "1984년에 3. 가지 원칙과 2 3 4 를 말했다."
    assert nr.paragraph_entries(text) == (text, 0)
