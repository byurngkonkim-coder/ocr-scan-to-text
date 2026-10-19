"""글머리 기호·구분 표시가 있는 줄은 줄 잇기로 앞 줄에 붙지 않는다 (SI운영방안, 2026-10-19)."""
import numpy as np

from core import noise_rules, postprocessor, rapidocr_engine


def _box(x, y, w=400, h=20):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def test_misread_bullet_restored():
    lines = ["ㅇ협력사, 프리랜서 구매는 합의", "ᄋ 제안팀 구성은 합의", "O영업대표:김정민", "OECD 보고서", "이 문장은 본문"]
    blank = np.full((400, 600, 3), 255, np.uint8)
    # 기호 없는 두 줄은 ○ 줄과 다른 x(본문 여백)에서 시작
    out = rapidocr_engine._restore_bullets(lines, [_box(x, 30 + 40 * i) for i, x in
                                                   enumerate([50, 50, 50, 20, 20])], blank)
    assert out[:3] == ["○ 협력사, 프리랜서 구매는 합의", "○ 제안팀 구성은 합의", "○ 영업대표:김정민"]
    assert out[3] == "OECD 보고서" and out[4] == "이 문장은 본문"


def test_long_acronym_at_bullet_x_is_not_eaten():
    lines = ["ㅇ 첫 항목", "ㅇ 둘째 항목", "OECD 보고서 요약"]
    blank = np.full((400, 600, 3), 255, np.uint8)
    out = rapidocr_engine._restore_bullets(lines, [_box(50, 30 + 40 * i) for i in range(3)], blank)
    assert out[2] == "○ OECD 보고서 요약"


def test_unread_bullet_follows_aligned_bullets():
    # OCR 이 ○ 를 아예 못 읽은 줄 — 같은 x 에서 시작하는 ○ 줄이 2개↑면 항목으로 본다(조금 왼쪽 □ 제목은 제외)
    lines = ["ㅇ 첫 항목", "ㅇ 둘째 항목", "셋째 항목(기호 못 읽음)", "목적", "OSI 사업체제"]
    xs = [50, 50, 52, 40, 50]
    blank = np.full((400, 600, 3), 255, np.uint8)
    out = rapidocr_engine._restore_bullets(lines, [_box(x, 30 + 40 * i) for i, x in enumerate(xs)], blank)
    assert out[2] == "○ 셋째 항목(기호 못 읽음)"
    assert out[3] == "목적"
    assert out[4] == "○ SI 사업체제"


def test_cleanup_keeps_bullets_and_edge_tokens():
    text = ("입찰참여, 협력업체 설정 시 수행팀 합의\n○ 최소 투찰금액, 협력업체 견적가는 수행팀 합의\n"
            "(5) 본부 내 서비스를 제외한 모든 SI매출사업\n-인력투입없는 장비사업: 10%\n"
            "※SI가 아닌 모든 서비스에도 10%를 산정")
    out = noise_rules.clean_page(text)
    assert "○ 최소 투찰금액" in out and "(5) 본부 내" in out
    assert "장비사업: 10%" in out and "※SI가 아닌" in out


def test_join_stops_at_numbered_marker():
    long = "가" * 30
    out = postprocessor.join_broken_lines(f"{long}\n(5) 본부 내 서비스\n{long}\n2) 둘째")
    assert out.splitlines() == [long, "(5) 본부 내 서비스", long, "2) 둘째"]
