"""글자 대응표가 깨진 글꼴의 줄만 이미지로 다시 읽기 (pdf_text_layer) 회귀 테스트.

실측: DBR 잡지 PDF(현대카드 Case Study 등)의 표·제목·캡션 글꼴(DongA3G)이 한글 일부를 구르무키·
말라얄람·아랍 문자 등으로, 영문을 밀린 글자('TVA'→'57A')로 내보냈다. 쪽 점수(garble_score)는
드문 한글 음절·한글+한자만 세어 이 쪽들을 깨진 쪽으로 잡지 못했고, 표 칸이 그대로 깨진 채 저장됐다.
"""
from core import pdf_text_layer as ptl


def test_foreign_script_detected_only_in_garbled_text():
    assert ptl.has_foreign_glyphs("전통੶ 회계")
    assert ptl.has_foreign_glyphs("포인트\x10ഊఔ")
    assert ptl.has_foreign_glyphs("ҳ주의 거울")                # 키릴 확장 문자(군주의)
    assert not ptl.has_foreign_glyphs("톨스토이(Толстой)")       # 현대 러시아어는 정상 글자
    assert not ptl.has_foreign_glyphs("전통적 회계 TVA(Total View Accounting) 1,234원 • ① “인용”")


def test_merge_replaces_only_garbled_parts():
    # 원문의 멀쩡한 글자는 그대로, 깨진 글자·밀린 영문 구간만 OCR 로
    new, ok = ptl.merge_reocr("전통੶ ߏ식의 회계৪ 57A ߏ식의 ମ이", "전통적 방식의 회계와 TVA 방식의 차이")
    assert ok and new == "전통적 방식의 회계와 TVA 방식의 차이"
    new, ok = ptl.merge_reocr("߃기ର 교수는 서울대 경영대학", "박기찬 교수는 서울대 경영대학")
    assert ok and new == "박기찬 교수는 서울대 경영대학"


def test_merge_keeps_original_where_ocr_misreads_clean_text():
    # OCR 이 멀쩡한 원문 글자를 다르게 읽은 곳('경영대학'→'경엉대학')은 원문을 남긴다
    new, ok = ptl.merge_reocr("߃기ର 교수는 서울대 경영대학", "박기찬 교수는 서울대 경엉대학")
    assert ok and new == "박기찬 교수는 서울대 경영대학"


def test_merge_distrusts_unrelated_ocr():
    # 원문의 멀쩡한 글자와 거의 안 맞는 OCR(그림 위 글자 등)은 믿지 않고 깨진 문자만 지운다
    new, ok = ptl.merge_reocr("회계를 ࢚܀게 ࠈঞѵ다는 아이디어는 어ڊ게 나ৱ나.", "룸를 콩저 그을 극이의")
    assert not ok and not ptl.has_foreign_glyphs(new) and "아이디어는" in new


def test_short_line_trusts_plausible_ocr():
    # 맞춰 볼 멀쩡한 글자가 거의 없는 짧은 줄 — 길이가 그럴듯하면 OCR 을 믿는다
    assert ptl.merge_reocr("ঢ়ಊۦ이션 ࠸용", "오퍼레이션 비용") == ("오퍼레이션 비용", True)
    assert ptl.merge_reocr("개ಜ യ", "개편후") == ("개편후", True)


def test_repair_page_text_replaces_garbled_lines_only():
    text = "본문 첫 줄은 멀쩡하다.\n전통੶ 회계\n본문 둘째 줄.\n개ಜ യ"
    lines = [("전통੶ 회계", "R1"), ("개ಜ യ", "R2")]
    fake = {"R1": "전통적 회계", "R2": "개편후"}
    new, n = ptl.repair_page_text(text, lines, lambda rect, orig: fake[rect])
    assert n == 2 and new == "본문 첫 줄은 멀쩡하다.\n전통적 회계\n본문 둘째 줄.\n개편후"


def test_repair_page_text_strips_leftover_garbage():
    # OCR 결과가 없어도 깨진 문자는 남기지 않는다
    new, n = ptl.repair_page_text("표 칸 ମ이", [("표 칸 ମ이", "R")], lambda rect, orig: "")
    assert n == 0 and new == "표 칸 이"
