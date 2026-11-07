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
