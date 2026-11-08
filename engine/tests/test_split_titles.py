"""장 번호 표지('03')를 사이에 두고 여러 문단으로 갈라진 장 제목을 차례 제목 줄로 세운다
(삶의지혜_자아편 실측, 2026-11-07 — 장 제목이 '…사람'과 / …사람'은 / 03 / 목표가 다르다' 로 흩어져
한 줄만 보는 rematch_headings 로는 21장 중 하나도 찾지 못했다)."""
from core import text_cleaner as tc

E = [{"level": 2, "title": "03. '살기 위해 먹는 사람'과 '먹기 위해 사는 사람'은 목표가 다르다", "page": 31},
     {"level": 2, "title": "04. '사병'도 장군의 제복을 입으면 '장군'처럼 변하듯, 능력이란 중책에서 나온다", "page": 40},
     {"level": 2, "title": "08. 자신의 '열정'을 제어하고, 아무 것도 하지 않는다고 생각해 보라", "page": 90}]


def test_title_scattered_around_number_becomes_heading():
    text = "\n\n".join(["앞 장의 마지막 문단이다.", "'살기 위해 먹는 사람'과", "'먹기 위해 사는 사람'은", "03",
                       "목표가 다르다", "위대한 사람에게는 반드시 목표가 있다 태양을 조준하라"])
    out, n = tc.place_split_titles(text, E[:1])
    assert n == 1
    assert out.split("\n\n") == ["앞 장의 마지막 문단이다.", "## " + E[0]["title"],
                                 "위대한 사람에게는 반드시 목표가 있다 태양을 조준하라"]


def test_number_inside_title_line_and_title_glued_to_body():
    text = "\n\n".join(["앞 문단이다.", "'사병'도 장군의 제복을 입으면 '장군'처럼 변하듯 04", "능력이란 중책에서 나온다",
                       "압력을 이용하라",
                       "자신의 '열정'을 제어하고, 아무 것도 하지 않는다고 08 생각해 보라 나태함은 무료함을 부른다 "
                       "천성적으로 게으른 사람은 없다."])
    out, n = tc.place_split_titles(text, E[1:])
    assert n == 2
    assert out.split("\n\n") == ["앞 문단이다.", "## " + E[1]["title"], "압력을 이용하라",
                                 "## " + E[2]["title"], "나태함은 무료함을 부른다 천성적으로 게으른 사람은 없다."]


def test_number_in_unrelated_body_left_alone():
    text = "\n\n".join(["13년 뒤 그는 백만장자가 되었다.", "03", "전혀 다른 이야기가 이어진다."])
    assert tc.place_split_titles(text, E[:1]) == (text, 0)


def test_title_without_number_found_between_neighbours():
    # OCR 이 장 번호를 못 읽은 장(01) — 앞 쪽 인용문 끝에 제목이 붙어 있다
    e = [{"level": 1, "title": "01. 진정 가치 있는 것은 '황금알' 자체가 아니라 '황금알'을 낳을 수 있는 닭이다", "page": 12},
         {"level": 1, "title": "02. 다음 장", "page": 24}]
    text = "\n\n".join(["좋은 기회인 것이다 발타자르그라 시안 진정 가치 있는 것은 '황금알'자체가 아니라.",
                       "'황금알'을 낳을 수 있는 닭이다 자신을 바꿔라", "인간은 세상에서 가장 복잡한 동물이다.",
                       "## 02. 다음 장"])
    out, n = tc.place_split_titles(text, e)
    assert n == 1
    assert out.split("\n\n")[:3] == ["좋은 기회인 것이다 발타자르그라 시안", "# " + e[0]["title"], "자신을 바꿔라"]


def test_toc_remnant_in_front_dropped():
    e = [{"level": 1, "title": t, "page": None} for t in
         ("서문", "11. 성숙함은 사람의 외모에서도 나타나지만, 그 사람의 마음에서 더욱 확연히 드러난다",
          "12. 정신 소양의 최고 경지는 평범한 충동들에 굴복하지 않는 것이다", "13. 자신부터 평온하게 살아라",
          "14. 마음을 바꾸면 세상도 따라서 변화한다", "역자후기")]
    body = ["서문", "서문 본문이다. 이 책은 그라시안의 잠언을 해설한다.", "제1편 자 아 편",
            "11. 성숙함은 사람의 외모에서도 나타나지만, 그사람의 마음에서 더욱 확연히 드러난다 111",
            "12. 정신 소양의 최고 경지는 평범한 충동들에 굴복하지 않는 것이다",
            "13. 자신부터 평온하게 살아라 126 14. 마음을 바꾸면 세상도 따라서 변화한다", "210", "역자후기", "216",
            "부록1. 삶의지혜 실천편 차례"] + [f"본문 문단 {i}이다." for i in range(60)]
    out, n = tc.drop_toc_remnant("\n\n".join(body), e)
    assert n == 8
    assert out.split("\n\n")[:3] == ["서문", "서문 본문이다. 이 책은 그라시안의 잠언을 해설한다.", "본문 문단 0이다."]


def test_part_and_chapter_titles_right_after_toc_remnant_kept():
    # 차례 조각('… 터보코칭 과정·247 / 옮긴이의 말·250') 바로 뒤의 부 표지·장 여는 쪽 제목 — 차례 순서가 맨 앞으로
    # 되돌아간 쪽번호 없는 제목은 본문이 시작된 것(TurboStrategy: 1부·1장 제목이 함께 지워졌다)
    e = [{"level": 1, "title": t, "page": None} for t in
         ("한국어판 서문", "원점에서 재검토하라", "1. 지금 있는 곳에서 시작하라", "21. 결과에 집중하라",
          "터보전략 과정", "터보코칭 과정", "옮긴이의 말")]
    body = ["한국어판 서문", "서문 본문이다.", "21. 결과에 집중하라·223", "터보전략 과정·243", "터보코칭 과정·247",
            "옮긴이의 말·250", "원점에서 재검토하라", "지금 있는 곳에서 시작하라",
            "Start Where You Are"] + [f"본문 문단 {i}이다." for i in range(60)]
    out, n = tc.drop_toc_remnant("\n\n".join(body), e)
    assert n == 4
    assert out.split("\n\n")[:5] == ["한국어판 서문", "서문 본문이다.", "원점에서 재검토하라", "지금 있는 곳에서 시작하라",
                                     "Start Where You Are"]


def test_inline_chapter_running_header_dropped():
    # 쪽 머리글 '제N장 장이름:부제 쪽번호' 가 쪽 경계에서 문장 한가운데 끼었다(Victory 46곳) — 차례 장 제목과 맞는 것만
    e = [{"level": 1, "title": t, "page": None} for t in ("2장 공세 – 대담하게 행동하라", "5장 정보 – 확실하게 선별하라")]
    text = "\n\n".join(["긴 활의 사용 제2장공세:대담하게 행동하라67 을 금지하는 칙령까지 나왔다.",
                       "제2장 공세:대담하게 행동하라81",
                       "뢰벤브로이가 진출했을 때의 제5장정보:확실하게 선별하라141 일이다.",
                       "# 2장 공세 – 대담하게 행동하라",
                       "- 차례\n  - 들여쓴  항목",
                       "제3장 실습:직접 해 보라12 같은 말은 차례에 없는 장이라 그대로 둔다."])
    out, n = tc.drop_inline_chapter_headers(text, e)
    assert n == 3
    assert out.split("\n\n") == ["긴 활의 사용 을 금지하는 칙령까지 나왔다.",
                                 "뢰벤브로이가 진출했을 때의 일이다.",
                                 "# 2장 공세 – 대담하게 행동하라",
                                 "- 차례\n  - 들여쓴  항목",
                                 "제3장 실습:직접 해 보라12 같은 말은 차례에 없는 장이라 그대로 둔다."]


def test_chapter_opener_with_repeated_title_not_toc_remnant():
    # 장 여는 쪽: 장 제목이 두 번(앞 쪽 끝·장 표지) + 첫 절 제목 — 제목 문단 3개지만 서로 다른 제목은 2개(Helping 2장)
    e = [{"level": 1, "title": t, "page": None} for t in ("1장 헬핑", "2장 도움의 작동원리", "기본 원칙 두 가지")]
    body = ["앞 장 끝 문단이다.", "도움의 작동 원리", "도움의 작동원리", "기본 원칙 두 가지"] \
        + [f"본문 문단 {i}이다." for i in range(60)]
    text = "\n\n".join(body)
    assert tc.drop_toc_remnant(text, e) == (text, 0)


# ── 1001가지기독교명언 (2026-11-07) ─────────────────────────────────────────────

def _topics(*titles):
    return [{"level": 1, "title": t, "page": None} for t in titles]


def test_short_topic_glued_to_end_of_previous_item_becomes_heading():
    e = _topics("부정직함", "부족함", "열매 맺음", "유혹")
    text = "\n\n".join(["# 부정직함", "328", "거짓말은 사탄의 무기다.", "-J.C.라일", "진리가 그 속에 없다 부족함",
                       "329", "하나님의 백성들은 자신을 혐오했다", "고 가르치셨다 열매맺음", "553 나무는 열매로 안다.",
                       "듯하게 하라 유혹", "655 어떤 경우에도"])
    out, n = tc.place_split_titles(text, e, tail=True)
    assert n == 3
    paras = out.split("\n\n")
    assert paras[4:7] == ["진리가 그 속에 없다", "# 부족함", "329"]
    assert "# 열매 맺음" in paras and "고 가르치셨다" in paras and "# 유혹" in paras


def test_tail_title_needs_item_number_after():
    e = _topics("부정직함", "부족함")
    text = "\n\n".join(["# 부정직함", "자신의 부족함", "다음 문단도 본문이다."])
    assert tc.place_split_titles(text, e, tail=True) == (text, 0)


def test_item_numbers_attach_to_following_item():
    paras = ["# 거룩함", "첫 명언(번호를 못 읽음)"]
    for k in range(10, 40):
        paras += [str(k), f"명언 {k} 본문", "-저자"]
    out, n = tc.attach_item_numbers("\n\n".join(paras))
    assert n == 30
    assert "10 명언 10 본문" in out.split("\n\n") and "39 명언 39 본문" in out.split("\n\n")


def test_scattered_numbers_are_not_item_chain():
    text = "\n\n".join(["2019", "본문", "15", "본문", "300", "본문"])
    assert tc.attach_item_numbers(text) == (text, 0)


# ── 12년차기획맨의기획실무노트 (2026-11-07) ─────────────────────────────────────

_E3 = [{"level": 1, "title": "PART III. 기획의 실행, 어떻게 할 것인가", "page": 235},
       {"level": 2, "title": "기획서, 제안에서 성공까지", "page": 237},
       {"level": 3, "title": "1. 기획서의 제안과 승인", "page": 239},
       {"level": 3, "title": "2. 기획서의 실천", "page": 241},
       {"level": 2, "title": "현실적으로는 기민한 대응 전술이 필요하다", "page": 243}]


def test_part_opener_toc_table_replaced_by_part_heading():
    text = "\n\n".join([
        "앞 부의 마지막 문단이다.",
        "| 기획의 실행 | PAR T III |  |\n| --- | --- | --- |\n|  | 어떻게할 것인가 | Chapter 7. 기획서,제안에서 성공까지"
        " 1.기획서의 제안과 승인 2.기획서의 실천 Chapter 8. 현실적으로는기민한대응전숨이 필요하다 |",
        "본문이 시작된다."])
    out, n = tc.drop_toc_tables(text, _E3)
    assert n == 1
    assert out.split("\n\n") == ["앞 부의 마지막 문단이다.", "# " + _E3[0]["title"], "본문이 시작된다."]


def test_ordinary_table_kept():
    text = "본문\n\n| 구분 | 매출 |\n| --- | --- |\n| 2019 | 100 |"
    assert tc.drop_toc_tables(text, _E3) == (text, 0)


def test_concept_map_table_with_some_titles_kept():
    # 책 내용 개념도 표 — 차례 제목이 3개 넘게 들었지만 표 글의 대부분은 차례에 없는 글(IDEAHACKS49 15쪽, 제목 몫 0.10)
    text = ("| | (1) 환경조성 | (2) 정보정리 |\n| --- | --- | --- |\n"
            "| (A) Hack의 기술 | 기획서의 제안과 승인 다음 단계로 고기를 먹지 않는다 걷는 속도는 1초에 두 걸음 |"
            " 기획서의 실천 아이디어 메모는 한시 바삐 버려라 노트의 시간축 관리 블로그로 아이디어를 발표한다 |\n"
            "| (B) Hack의 도구 | 현실적으로는 기민한 대응 전술이 필요하다 모닝 샤워는 아이디어 샤워 비밀의 화원을 찾아라"
            " 사무실에서 다과회를 열자 만년필로 선을 긋는다 나만의 테마송을 틀어둔다 | 수첩에 나의 사전을 만든다"
            " 명함 클립으로 인맥 정리 책장을 지식의 데이터베이스로 바꾼다 하드디스크는 보물로 가득 차 있다 |")
    assert tc.drop_toc_tables(text, _E3) == (text, 0)


def test_category_box_listing_all_chapters_kept():
    # 꼭지마다 있는 '기능별 분류' 표 — 장 이름 7개가 모두 들었다. 작은 차례는 맨 윗단계 항목이 하나뿐이다
    # (The28BusinessBible 2026-11-08: 표 6개를 지우고 '제1장 제너럴 매니지먼트' 제목 6개로 바꿈)
    chapters = ["제너럴 매니지먼트", "논리적 사고", "기술경영·기업가 정신", "사람 (HR/조직행동)", "물건 (마케팅)",
                "돈 (회계·재무)", "전략"]
    entries = [{"level": 1, "title": f"제{i}장 {t}", "page": None} for i, t in enumerate(chapters, 1)]
    text = ("본문\n\n| 기능별 분류 | 제너럴매니지먼트 | O |\n| --- | --- | --- |\n"
            "|  | 논리적 사고 기술경영·기업가 정신 사람(HR/조직행동) 물건(마케팅) 돈(회계·재무) 전략 |  |")
    assert tc.drop_toc_tables(text, entries) == (text, 0)


def test_heading_in_chapter_opener_list_moves_to_real_section():
    e = [{"level": 2, "title": "환경 분석, 어떻게 할 것인가", "page": 15},
         {"level": 3, "title": "1. 외부 환경 분석", "page": 16},
         {"level": 3, "title": "2. 내부 역량 분석", "page": 34}]
    text = "\n\n".join(["## 환경 분석, 어떻게 할 것인가", "### 1. 외부 환경 분석", "1)거시 환경 분석", "2)산업분석",
                       "2. 내부 여량 분석 1)성과분석 2)재무분석", "1.외부 환경 분석", "외부 환경은 기업에 영향을 미친다.",
                       "### 2. 내부 역량 분석"])
    out, n = tc.relocate_list_headings(text, e)
    assert n == 1
    paras = out.split("\n\n")
    assert paras[1] == "1. 외부 환경 분석" and paras[5] == "### 1. 외부 환경 분석"


def test_same_title_twice_in_toc_placed_in_both_parts():
    # PART I·II 에 같은 절 제목 — 글자로만 찾으면 두 항목이 한 줄을 가리켰다
    e = [{"level": 1, "title": "PART I. 이론", "page": 1}, {"level": 2, "title": "1. 외부 환경 분석", "page": 2},
         {"level": 1, "title": "PART II. 사례", "page": 3}, {"level": 2, "title": "1. 외부 환경 분석", "page": 4}]
    text = "\n\n".join(["# PART I. 이론", "## 1. 외부 환경 분석", "이론 본문.", "# PART II. 사례",
                       "1. 외부 환경 분석 외부 환경 분석을 담당하게 된 김갑돌은 틀부터 작성한다."])
    assert tc._entry_positions(text.split("\n\n"), e) == [0, 1, 3, None]
    out, n = tc.place_split_titles(text, e)
    assert n == 1 and out.split("\n\n")[4] == "## 1. 외부 환경 분석"


def test_cover_ad_before_first_heading_dropped():
    # 1분의사결정 — 머리말 앞에 뒤표지 광고(ISBN·값·시리즈 목록·그림 크레딧)가 남았다
    text = "\n\n".join(["인생은 선택의 연속이다", "-분 시리즈 … ISBN 89-352-0180-4 값5000원", "청림1분시리즈",
                       "1.1분 의사결정 스펜서 존슨 지음·서기원 옮김", "2. 1분 매니저 케네스 블랜차드 지음",
                       "15. 1분창조력개발 와타나베 지음 본문그림:손문상", "# 머리말", "본문이다. 값 5000원이라고 썼다."])
    out, n = tc.drop_cover_ad(text)
    assert n == 5
    assert out.split("\n\n") == ["인생은 선택의 연속이다", "# 머리말", "본문이다. 값 5000원이라고 썼다."]


def test_no_cover_ad_unchanged():
    text = "# 머리말\n\n본문 ISBN 이야기."
    assert tc.drop_cover_ad(text) == (text, 0)


# ── 2x2Matrix (2026-11-07) ─────────────────────────────────────────────────────

def _m(*titles):
    return [{"level": 3, "title": t, "page": None} for t in titles]


def test_chapter_opener_name_list_dropped():
    e = _m("고객 주도를 넘어서", "불연속성과 수명주기 전략", "가치 매니저로서의 고객", "고객 가치 분석", "구매 가능성")
    text = "\n\n".join(["# Part 3", "고객 니즈의 프레임 워크 고객 주도를 넘어서 불연속성과 수명 주기 전략 가치 매니저로서의 고객 고객가치 분석",
                       "구매 가능성", "고객 주도를 넘어서 불연속성과 수명주기 전략 가치 매니저로서의 고객",
                       "본문이다. 고객 가치 분석을 해 보자."])
    out, n = tc.drop_toc_dense(text, e)
    assert n == 3
    assert out.split("\n\n") == ["# Part 3", "본문이다. 고객 가치 분석을 해 보자."]


def test_single_title_line_is_not_dense():
    e = [{"level": 2, "title": t, "page": None} for t in
         ("제1장에 들어가면서: 가정이 중요한 이유", "제5장에 들어가면서", "제6장에 들어가면서")]
    text = "제1장에 들어가면서:가정이 중요한 이유"
    assert tc.drop_toc_dense(text, e) == (text, 0)


def test_title_followed_by_english_original_title():
    e = _m("포트폴리오 분석", "문제와 해결책", "T-그룹 리더십", "변증법적 SWOT 분석")
    text = "\n\n".join(["### 포트폴리오 분석", "본문.",
                       "문제와 해결책Problems and Solutions Watts Wacker 미래가 나타나길 기다리는 사람에게는 미래가 없다.",
                       "본문 2.", "T-그룹리더십T-Group Leadership Richard Nelson Jones 리더의 업무를…", "### 변증법적 SWOT 분석"])
    out, n = tc.place_split_titles(text, e)
    assert n == 2
    paras = out.split("\n\n")
    assert paras[2:4] == ["### 문제와 해결책", "Problems and Solutions Watts Wacker 미래가 나타나길 기다리는 사람에게는 미래가 없다."]
    assert "### T-그룹 리더십" in paras


def test_exact_pass_skips_similar_index_line():
    e = _m("본원적 네트워크 전략", "매스 커스터마이제이션 _ 4가지 접근법")
    lines = ["### 본원적 네트워크 전략", "본문", "매스커스터마이제이션_4가지 접근법Mass Customization 고객마다 다르게", "본문",
             "매스 커스터마이제이션 238"]
    out = tc.rematch_headings("\n".join(lines), e, {}, respace=False, keep_titles=True, exact=True)
    assert "### 매스 커스터마이제이션 _ 4가지 접근법" not in out.split("\n")   # 색인 줄은 같은 글자가 아니다


def test_early_entry_found_only_in_end_notes_does_not_drag_others():
    # Alianment정렬 실측(2026-11-07) — 본문엔 '서문' 줄이 없고 맨 뒤 주석에만 있어, 차례 순서대로 앞에서부터 짝지으면
    # '서문' 이 주석 줄을 잡고 그 뒤 장 제목 10개가 모두 주석 속 장 이름 줄에 섰다. 뒷부분 항목('주석') 앞 항목은
    # 그 줄 너머에서 찾지 않는다.
    e = _m("서문", "감사의 말", "재무와 고객 전략의 정렬", "지원 기능의 정렬", "주석")
    paras = ["감사의 말", "고마운 분들.", "재무와 고객 전략의 정렬", "본문 1.", "지원 기능의 정렬", "본문 2.",
             "주석", "서문", "1 캐플란 교수와 노튼 박사 저.", "재무와 고객 전략의 정렬", "2 각주."]
    out = tc.rematch_headings("\n\n".join(paras), e, {}, respace=False, keep_titles=True)
    got = out.split("\n\n")
    assert got[:5] == ["### 감사의 말", "고마운 분들.", "### 재무와 고객 전략의 정렬", "본문 1.", "### 지원 기능의 정렬"]
    assert got[6:10] == ["### 주석", "서문", "1 캐플란 교수와 노튼 박사 저.", "재무와 고객 전략의 정렬"]


def test_title_with_latin_tail_counted_found():
    # 영어 원제를 제목 아래로 내린 줄도 찾은 제목으로 센다(Helping 7장 '…팀워크Teamwork' — 문서 정보가 '못 찾음' 이라 적었다)
    e = _m("팀워크")
    stats = {}
    out = tc.rematch_headings("앞 글.\n\n팀워크Teamwork\n\n본문.", e, stats, respace=False, keep_titles=True)
    assert "### 팀워크\n\nTeamwork" in out and stats["missing"] == []


def test_small_title_after_underscore_matches_body_line():
    # 장 표지 '기업 전략과 조직 구조 / 역사적 고찰' — 본문 OCR 에는 작은 제목 줄만 남았다(Alianment정렬)
    e = _m("1장 들어가며", "2장 기업 전략과 조직 구조 _ 역사적 고찰", "3장 재무와 고객 전략의 정렬")
    paras = ["### 1장 들어가며", "본문.", "역사적 고찰", "본문 2.", "### 3장 재무와 고객 전략의 정렬"]
    out = tc.rematch_headings("\n\n".join(paras), e, {}, respace=False, keep_titles=True)
    assert out.split("\n\n")[2] == "### 2장 기업 전략과 조직 구조 _ 역사적 고찰"
    # 앞(엄격) 패스에서는 작은 제목으로 찾지 않는다 — 큰 제목 줄을 뒤 패스가 찾을 몫(2천년의강의 회귀)
    out = tc.rematch_headings("\n\n".join(paras), e, {}, respace=False, keep_titles=True, exact=True)
    assert out.split("\n\n")[2] == "역사적 고찰"


def test_chapter_title_wins_over_section_names_in_opener_list():
    # 2x2Matrix 회귀(2026-11-07) — 장 여는 쪽의 절 이름 목록이 장 제목 줄보다 앞에 있을 때, '가장 많이 세우는 짝'
    # 방식이 절 이름을 세우고 장 제목을 버렸다(64→61). 차례 순서대로 앞 항목부터 찾으면 장 제목이 선다.
    e = [{"level": 2, "title": "7장 조직 혁신의 프레임워크", "page": None},
         {"level": 3, "title": "구성원 동기부여", "page": None}]
    paras = ["앞 장 본문.", "구성원 동기부여", "7장 조직 혁신의 프레임 워크", "본문."]
    out = tc.rematch_headings("\n\n".join(paras), e, {}, respace=False, keep_titles=True)
    assert "## 7장 조직 혁신의 프레임워크" in out.split("\n\n")


def test_glued_list_numbers_in_run_spaced():
    # CreatYourOwnFuture 실측(2026-11-07) — 실천 목록 '1오늘부터 …' / '2자신의 …' 번호가 글에 붙었다(10권 합 393곳)
    text = "\n\n".join(["실천 과제", "1오늘부터 생각하자.", "2자신의 가능성을 믿자.", "3항상 최선을 생각하자.",
                       "5자신의 목표를 공언하자.", "본문 문단."])
    out, n = tc.space_list_numbers(text)
    assert n == 4 and out.split("\n\n")[1:5] == ["1 오늘부터 생각하자.", "2 자신의 가능성을 믿자.",
                                                 "3 항상 최선을 생각하자.", "5 자신의 목표를 공언하자."]


def test_number_with_unit_or_alone_kept():
    # 단위가 붙은 수('2년 만에'·'24시간'·'10대와')와 사슬 없는 외딴 번호는 그대로
    text = "\n\n".join(["2년 만에, 나는 수입이 배로 늘었다.", "24시간 현금 지급기가 인기다.", "10대와 20대들은 다르다.",
                       "본문.", "7어떤 경우에나 긍정적이자.", "1쇄인쇄 2005년", "3쇄발행 2006년", "10백만원/월"])
    assert tc.space_list_numbers(text) == (text, 0)


def test_title_split_over_two_short_lines():
    # First,BreakAllTheRules 실측(2026-11-07) — 장 표지 '세번째 열쇠:' / '장점을 개발하라' 두 줄로 갈림
    e = [{"level": 1, "title": "Chapter 4 두번째 열쇠: 합리적 목표를 설정하라", "page": None},
         {"level": 1, "title": "Chapter 5 세번째 열쇠: 장점을 개발하라", "page": None}]
    lines = ["# Chapter 4 두번째 열쇠: 합리적 목표를 설정하라", "본문.", "세번째 열쇠:", "", "장점을 개발하라", "", "본문 2."]
    out = tc.rematch_headings("\n".join(lines), e, {}, respace=False, keep_titles=True, exact=True)
    got = out.split("\n")
    assert got[2] == "# Chapter 5 세번째 열쇠: 장점을 개발하라"
    # 남은 둘째 줄은 제목 끝말 반복이라 drop_heading_echo 가 지운다
    assert "장점을 개발하라" not in tc.drop_heading_echo(out.replace("\n\n", "\n").replace("\n", "\n\n"))[0].split("\n\n")[3:]


def test_noise_line_not_joined_with_title_line():
    # DrivingSalesmanPowerSkill 회귀 — 글자 없는 '#' 잡음 줄 + 진짜 제목 줄이 '두 줄 제목' 으로 잡혀 제목이 잡음 줄에 섰다
    e = [{"level": 2, "title": "03 이제 세일즈를 해볼까", "page": None}]
    lines = ["#", "", "03 이제 세일즈를 해 볼까", "", "본문."]
    out = tc.rematch_headings("\n".join(lines), e, {}, respace=False, keep_titles=True, exact=True)
    assert out.split("\n")[2] == "## 03 이제 세일즈를 해볼까"
    # 한글 없는 줄('2x2 Matrix')도 두 줄 제목의 앞줄이 아니다(2x2Matrix 회귀)
    e = [{"level": 2, "title": "4장 2×2 매트릭스를 만드는 8단계", "page": None}]
    lines = ["2x2 Matrix", "", "매트릭스를 만드는 8단계", "", "본문."]
    out = tc.rematch_headings("\n".join(lines), e, {}, respace=False, keep_titles=True, exact=True)
    assert out.split("\n")[0] == "2x2 Matrix"


def test_two_unrelated_short_lines_not_joined():
    e = [{"level": 1, "title": "Chapter 5 세번째 열쇠: 장점을 개발하라", "page": None}]
    lines = ["세번째 열쇠:", "", "다른 이야기를 하자", "", "본문."]
    out = tc.rematch_headings("\n".join(lines), e, {}, respace=False, keep_titles=True, exact=True)
    assert out.split("\n")[0] == "세번째 열쇠:"


def test_lone_chapter_label_line_used_when_title_absent():
    # FocalPoint 실측(2026-11-07) — 장 표지에 '제1장' 만 있고 큰 제목은 차례에만. 느슨한 패스의 마지막 수단
    e = [{"level": 1, "title": "제1장 습관이 운명을 결정한다", "page": None},
         {"level": 1, "title": "제2장 핵심을 찾아 집중하라", "page": None}]
    lines = ["프롤로그 끝.", "", "제1장", "", "모든 위대한 사람은", "", "## 제2장 핵심을 찾아 집중하라"]
    out = tc.rematch_headings("\n".join(lines), e, {}, respace=False, keep_titles=True)
    assert out.split("\n")[2] == "# 제1장 습관이 운명을 결정한다"
    # 엄격한 앞 패스에서는 쓰지 않는다
    out = tc.rematch_headings("\n".join(lines), e, {}, respace=False, keep_titles=True, exact=True)
    assert out.split("\n")[2] == "제1장"


def test_two_line_title_with_bare_number_label():
    # MBA명강의 실측(2026-11-08) — 차례 '07 고객의 경험을 창조하고 관리하는 CEM', 본문은 번호 없이 두 줄로 갈림.
    # '07 ' 처럼 점 없는 번호표를 떼지 못해 두 줄 잇기가 차례 제목과 견주지 못했다(5곳)
    e = [{"level": 1, "title": "제2부 마케팅", "page": None},
         {"level": 2, "title": "07 고객의 경험을 창조하고 관리하는 CEM", "page": None}]
    lines = ["# 제2부 마케팅", "본문.", "고객의 경험을", "", "창조하고 관리하는CEM", "", "본문 2."]
    out = tc.rematch_headings("\n".join(lines), e, {}, respace=False, keep_titles=True)
    assert "## 07 고객의 경험을 창조하고 관리하는 CEM" in out.split("\n")
    assert tc._strip_label("07 고객의 경험을") == "고객의 경험을"
    assert tc._strip_label("30억 통화 분석") == "30억 통화 분석"   # 단위가 붙은 숫자는 그대로


def test_part_section_chapter_opener_not_toc_remnant():
    # 전쟁론 실측(2026-11-08) — 책 앞쪽의 '제1부' / '제1편 전쟁의 본질' / '제1장 전쟁이란 무엇인가' 는 서로 다른
    # 제목 3개지만 장 여는 쪽이다. 남은 차례 덩어리는 쪽번호가 붙어 있다 — 쪽번호가 없으면 제목이 많을 때만 차례로 본다
    e = [{"level": 1, "title": t, "page": None} for t in ("제1부", "제1편 전쟁의 본질", "제1장 전쟁이란 무엇인가")]
    body = ["머리말 끝 문단이다.", "제1부", "제1편 전쟁의 본질", "제1장 전쟁이란 무엇인가"] \
        + [f"본문 문단 {i}이다." for i in range(60)]
    text = "\n\n".join(body)
    assert tc.drop_toc_remnant(text, e) == (text, 0)


def test_heading_not_relocated_into_end_summary_list():
    # SalesSuperStar 실측(2026-11-08) — 책 끝 '요약 및 정리' 가 장 제목 21개를 그대로 늘어놓았다. 장 제목 바로 아래가
    # 짧은 줄이라 '목록 속 제목' 으로 보고 같은 글자의 요약 줄로 옮겨, 21장이 모두 요약 쪽에 섰다(쪽과 먼 제목 12)
    # 이 단계엔 쪽 표지가 없다 — 차례 쪽(27/150)과 문서 속 자리 비율로 견준다
    e = [{"level": 1, "title": "1. 확신의 힘", "page": 27}, {"level": 1, "title": "나오며 요약 및 정리", "page": 150}]
    body = [f"본문 문단 {i}이다. 길게 이어지는 설명이 여기 있다." for i in range(30)]
    paras = body[:5] + ["# 1. 확신의 힘", "짧은 인용"] + body[5:] + ["1. 확신의 힘", "설명 문단이다."]
    out, n = tc.relocate_list_headings("\n\n".join(paras), e)
    assert n == 0 and out.split("\n\n")[5] == "# 1. 확신의 힘"


def test_title_search_stays_near_toc_page():
    # SalesSuperStar 실측(2026-11-08) — 2장 본문 제목은 OCR 오자('지배하느나')라 앞 패스가 책 끝 요약의 깨끗한
    # 같은 줄을 잡았고, 3~21장이 모두 그 뒤를 따라 요약 쪽에 섰다. 차례 쪽(32/160) 둘레만 찾는다
    e = [{"level": 1, "title": "2. 두려움의 심리 - 지배하느냐, 지배받느냐", "page": 32},
         {"level": 1, "title": "나오며 요약 및 정리", "page": 160}]
    body = [f"본문 줄 {i}이다." for i in range(300)]
    lines = body[:60] + ["2.두려움의 심리 -지배하느나,지배 받느냐"] + body[60:] + ["2. 두려움의 심리-지배하느냐, 지배받느냐"]
    out = tc.rematch_headings("\n".join(lines), e, {}, respace=False, keep_titles=True, exact=True)
    assert out.split("\n")[60] != "## " and out.split("\n")[-1].startswith("2.")   # 끝 요약 줄은 그대로
