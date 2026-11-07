"""한 글자씩 띄운 쪽 머리글이 본문 문단에 섞여 든 것을 지운다(삶의지혜_자아편 실측, 2026-11-07).
'소 중 한 사 람에 게 주 고 싶 은 삶의 지 혜' — OCR 이 매번 조금씩 달리 읽어('사 람 예 게'·'삶의 지 혜다른')
쪽 맨 위·아래 줄만 보는 strip_running_headers 로는 지워지지 않았다."""
from core import text_cleaner as tc

H = "소 중 한 사 람에 게 주 고 싶 은 삶의 지 혜"


def test_repeated_spaced_header_removed_with_ocr_variants():
    text = "\n\n".join([
        f"{H} 우리는 천재가 될 수 없는 운명을 타고난 것이 아니다.",
        f"앞 문단이 끝났다 {H} 최고의 인재를 써서",
        "소 중 한 사 람 예 게 주 고 싶 은 삶의 지 혜 성숙함은 사람의 외모에서도",
        "소 중 한 사 람에 게 주 고 싶 은 삶의 지 혜다른 사람들의 장점을 모두 따라 하려는 것은",
        "소 중 한 사 람에 게 주 고 싶 은 삶의 지 다음 문단.",
        f"{H} 마지막 문단.",
    ])
    out, n = tc.strip_spaced_running_header(text)
    assert n == 6
    assert out.split("\n\n") == [
        "우리는 천재가 될 수 없는 운명을 타고난 것이 아니다.",
        "앞 문단이 끝났다 최고의 인재를 써서",
        "성숙함은 사람의 외모에서도",
        "다른 사람들의 장점을 모두 따라 하려는 것은",
        "다음 문단.",
        "마지막 문단.",
    ]


def test_normally_spaced_phrase_and_rare_runs_kept():
    # 보통 띄어쓰기로 쓴 같은 말·한 번만 나온 띄운 글자는 본문이다
    text = "\n\n".join([f"{H} 첫째.", f"{H} 둘째.", f"{H} 셋째.",
                        "이 책은 소중한 사람에게 주고 싶은 삶의 지혜를 담았다.",
                        "만 약 천 만 원 을 줄 테 니 장님이 되라면"])
    out, _ = tc.strip_spaced_running_header(text)
    assert "소중한 사람에게 주고 싶은 삶의 지혜를 담았다." in out
    assert "만 약 천 만 원 을 줄 테 니 장님이 되라면" in out


def test_no_header_book_unchanged():
    text = "보통 책의 문단이다.\n\n또 다른 문단이다."
    assert tc.strip_spaced_running_header(text) == (text, 0)
