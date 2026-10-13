"""MyOCR text_refiner 의 2026-10-10 정제 수정 이식 회귀 테스트.

MyOCR 커밋 8b1a0ed(깨진 반복 머리글·어미로 시작하는 줄 잇기)·f956b2b(OCR 잡음 쪽)·
514eef3(줄 잇기 75% 기준·구두점·'즉,' 줄)과 같은 동작을 01_OCR 의
text_cleaner·postprocessor 에서 확인한다. 테스트 내용은 MyOCR engine/tests/test_text_refiner.py 와 같다.
"""
from core import postprocessor as pp
from core import text_cleaner as tc

SENT = "이것은 충분히 긴 본문 문장으로 마침표로 끝난다."


# ── 반복 머리글 (8b1a0ed) ──────────────────────────────────────────────────────

def test_garbled_running_header_variants_removed():
    # 실측(1분의사결정): 같은 머리글을 쪽마다 다르게 읽음 — 띄어쓰기·깨진 쪽번호·장 번호 공백
    variants = ["n. 머리를 써라 39", "n. 머리를써라 43", "n. 머리를 써라 6?", "n. 머리를써라6i",
                "n. 머리를 써라 si", "in. 머리를 써라 47", "I . 머리를 써라 17"]
    pages = [f"{h}\n본문 {i}쪽 문장입니다." for i, h in enumerate(variants)]
    out = tc.strip_running_headers(pages)
    assert out[0] == pages[0]                              # 첫 등장(장 머리글)은 보존
    assert all("써라" not in p for p in out[1:]), out


def test_chapter_heads_without_section_mark_kept():
    # 장 번호 표시가 없는 '누가복음 5' 꼴은 묶지 않는다 (장 머리글 오삭제 방지)
    pages = [f"누가복음 {i}\n본문 {i}." for i in range(1, 8)]
    assert tc.strip_running_headers(pages) == pages


# ── 어미·조사로 시작하는 줄 잇기 (8b1a0ed) ─────────────────────────────────────

def test_short_line_joined_when_next_starts_with_ending():
    text, n = tc.rejoin_split_sentences("자신의 잘못을 발견하\n\n며,아무도 원하지 않는다.")
    assert n == 1 and text == "자신의 잘못을 발견하며,아무도 원하지 않는다."
    text, _ = tc.rejoin_split_sentences("성공을 놓치는 일\n을 하는 것 같다.")
    assert text == "성공을 놓치는 일을 하는 것 같다."


def test_short_line_not_joined_to_standalone_word():
    # '이 책'·'서 있다'·'만 원' 처럼 독립 단어로 시작하는 줄은 제목 뒤에 붙이지 않는다
    for nxt in ["이 책은 경영을 다룬다.", "서 있는 사람이 많았다.", "만 원을 냈다."]:
        src = f"머리말\n\n{nxt}"
        assert tc.rejoin_split_sentences(src)[0] == src, nxt


# ── OCR 잡음 쪽 (f956b2b) ──────────────────────────────────────────────────────

def test_noise_page_inside_sentence_dropped():
    # 실측(1분의사결정 91쪽): '…해낼 만큼' ⟨'인걸'⟩ '의 인격을…'
    pages = [SENT, "이 인물은 일을 해낼 만큼", "인걸", "의 인격을 갖고 있는가?", SENT]
    notes = []
    out = tc.drop_noise_pages(pages, notes)
    assert out[2] == "" and notes and "3번째 쪽('인걸')" in notes[0]


def test_short_title_pages_kept():
    # 장 제목 쪽 — 앞 쪽이 서명처럼 문장 부호 없이 끝나도, 다음 쪽이 본문 문장으로 시작하면 보존
    for title in ["서문", "PART 1", "I. 자신의 길을 찾는다"]:
        pages = [SENT, "2026년 봄\n스펜서 존슨", title, SENT]
        notes = []
        assert tc.drop_noise_pages(pages, notes) == pages and notes == [], title


def test_toc_page_number_not_shifted_by_front_ad():
    # 앞쪽 도서 광고를 지운 뒤 빈 쪽을 바로 걸러 내 차례 보고 번호가 당겨지던 것(3쪽 → '2번째')
    ad = "고흐 그림여행\n\n최상운 지음 | 296쪽 | 14,000원"
    pages = [ad, SENT, "차례\n\n1 경영의 등장 043\n2 경영 붐 058", SENT] + [f"{k}장 {SENT}" for k in range(60)]
    out, notes = tc.strip_book_matter(pages)
    assert any("차례 3~3번째 쪽" in n for n in notes), notes
    assert "14,000원" not in "\n".join(out) and "경영 붐" not in "\n".join(out)


def test_noise_page_report_keeps_numbering_with_book_matter(monkeypatch):
    # 잡음 쪽(4번째)을 지워도 뒤쪽 판권 보고 번호가 당겨지지 않는다
    monkeypatch.setattr(tc._gemini, "run_text", lambda p: "정제된 충분히 긴 본문 " * 10)
    body = [f"{k}장 {SENT}" for k in range(60)]
    colophon = "KI신서 467\n\n펴낸곳 북21 펴낸이 김영곤\n\nISBN 89-509-0533-7"
    pages = [SENT, SENT, "이 인물은 일을 해낼 만큼", "인걸", "의 인격을 갖고 있는가?"] + body + [colophon]
    tc.clean_pages(pages, drop_front_matter=True)
    warns = tc.get_last_report()["warnings"]
    assert any("OCR 잡음 쪽" in w and "4번째 쪽('인걸')" in w for w in warns), warns
    assert any(f"판권 {len(pages)}번째 쪽" in w for w in warns), warns


# ── 줄 잇기·구두점·잡음 줄 (514eef3)───────────────────────────────────────────

def test_heading_and_list_not_joined_to_body():
    # 정제본 실측: 짧은 제목·목록·표 행을 본문에 붙였다 — 오른쪽 끝까지 찬 줄만 잇는다
    body = "".join(f"본문 {k}번째 줄은 쪽의 오른쪽 끝까지 꽉 차 있다.\n" for k in range(6))
    page = (body +
            "경영자는 조직의 성과를 책임지는 사람이며 그 성과는\n"
            "조직 바깥에서 고객에 의해 판정된다고 보아야 한다.\n"
            "목표는 완전한 이해다\n"
            "리서치를 하는 목적은 완벽하게 주제를 파악하는 것이다.\n"
            "- 태양열 시장을 개발할 전략\n"
            "국내 극장 4,928,000\n"
            "국외 극장 1,228,000")
    out = pp.join_broken_lines(page).splitlines()
    assert out[6] == "경영자는 조직의 성과를 책임지는 사람이며 그 성과는 조직 바깥에서 고객에 의해 판정된다고 보아야 한다."
    assert "목표는 완전한 이해다" in out and "- 태양열 시장을 개발할 전략" in out
    assert "국내 극장 4,928,000" in out and "국외 극장 1,228,000" in out


def test_line_wrap_inside_word_joined_without_space():
    # 예측기계 30쪽 실측: 줄 끝에서 잘린 낱말을 띄워 이었다('운용되었 다', '일 반적인')
    body = "".join(f"본문 {k}번째 줄은 쪽의 오른쪽 끝까지 꽉 차 있다.\n" for k in range(6))
    page = (body +
            "그런 차량은 공장이나 창고 등 세부 평면도가 확보된 장소에서만 운용되었\n"
            "다. 그러나 일반 도로에서는 이런 차량을 사용할 수 없었고 모든 데이터를\n"
            "꼼꼼히 살피는 동시에 운전자의 행동을 유심히 관찰했다고 한다.")
    out = pp.join_broken_lines(page)
    assert "운용되었다." in out and "운용되었 다" not in out
    assert "데이터를 꼼꼼히" in out          # 낱말 경계는 그대로 띄운다


def test_page_number_lines_dropped_at_page_edges():
    # 예측기계 30쪽: 쪽 번호 줄이 본문 끝에 붙었다('어떤 환경데 30')
    text = "31\n본문 첫 줄은 이렇게 시작한다.\n2017년에 11까지 돌려 보라\n30"
    out = pp.process_full(text, engine="rapidocr")
    assert out.splitlines()[0] == "본문 첫 줄은 이렇게 시작한다."
    assert not out.rstrip().endswith("30") and "11까지" in out


def test_punctuation_after_initials_kept():
    assert pp._fix_spacing("C.의 주장과 U.S.의 정책") == "C.의 주장과 U.S.의 정책"
    assert pp._fix_spacing("했다.그리고") == "했다. 그리고"


def test_punctuation_spacing_keeps_urls_and_newlines():
    # 구두점 뒤 영문을 띄우던 것('www. naver. com')과 구두점 앞 줄바꿈을 먹던 것
    assert pp._fix_spacing("www.naver.com 과 OCR.pdf") == "www.naver.com 과 OCR.pdf"
    assert pp._fix_spacing("첫 줄\n.둘째") == "첫 줄\n. 둘째"


def test_single_char_word_line_kept():
    assert pp.remove_noise_lines("즉,\n본문 줄\nl") == "즉,\n본문 줄"


def test_page_footer_not_joined_into_last_body_line():
    # 예측기계 54쪽 실측: 쪽 꼬리('장 제목+쪽번호')가 앞 줄 끝에 붙어('…된다.회 4장왜지능이라고 하는가?55')
    # 문서 단위 머리글 제거가 놓쳤고, 각주 줄에 붙은 것은 정제 LLM 이 각주째 지웠다
    body = "".join(f"본문 {k}번째 줄은 쪽의 오른쪽 끝까지 꽉 차 있다.\n" for k in range(6))
    # 짝수 쪽 꼬리('쪽번호+부 제목')는 OCR 이 숫자를 붙여 읽는다 — '1582부의사결정'(158쪽 2부, 예측기계 실측 4건)
    for footer in ("4장왜지능이라고 하는가?55", "12장워크플로해체179", "2장 값이 싸지면 모든 것이 바뀐다 19",
                   "1582부의사결정", "172 3부툴"):
        page = body + "평균적으로 정확하다는것은사실 번번이 틀린다는뜻도 된다.회\n" + footer
        assert pp.join_broken_lines(page).splitlines()[-1] == footer, footer
    # 쪽 끝 본문 줄이 숫자로 끝나는 보통 문장이면 그대로 잇는다
    page = body + "그 회사의 매출은 꾸준히 늘어 2019년에는 역대 최고인\n1,250억 원에 이르렀다 2019"
    assert pp.join_broken_lines(page).splitlines()[-1].startswith("그 회사의 매출은")


def test_paragraph_start_mark_survives_without_period():
    # 예측기계 158쪽 실측: OCR 이 마침표를 빠뜨려('…따라달라진다') 다음 문단이 앞 문장 끝에 붙었다.
    # rapidocr_engine 이 들여 쓴 줄 앞에 붙인 공백 2칸을 잡음 제거가 지워 줄 잇기가 몰랐다
    body = "".join(f"본문 {k}번째 줄은 쪽의 오른쪽 끝까지 꽉 차 있는 줄이다\n" for k in range(4))
    page = body + "된기능도할 때그상대적 이득이어느정도인지에 따라달라진다\n  인간과 기계는데이터의유형에 따라 입력이나 훈련이나\n위한데이터를축적한다."
    out = pp.process_full(page).splitlines()
    assert out[-1] == "인간과 기계는데이터의유형에 따라 입력이나 훈련이나 위한데이터를축적한다.", out
    assert out[-2].endswith("따라달라진다"), out
