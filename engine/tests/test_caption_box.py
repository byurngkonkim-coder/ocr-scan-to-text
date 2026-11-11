"""설명 상자(테두리 + 아래 색 띠 설명문) — 현직자를위한무역실무_정재환, 2026-11-11.

이 책은 서류 견본·화면 캡처·도식을 얇은 테두리 상자에 넣고 맨 아래 옅은 파란 띠에 설명문을 단다.
예전 결과는 상자 테두리·견본 서류의 선을 표로 읽어 빈 칸투성이 표가 됐고(25쪽 운송장: 13행 10열, 빈 칸 75%),
쪽 가장자리 그림자가 표로 잡혀 본문 줄 끝이 표 칸으로 들어갔다(278쪽).
표본: sample_docs/caption_box_p25.png(운송장 사진), caption_box_p278.png(작은 글자 표 그림 + 색 바탕 글상자).
"""
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from core import figure_detector as fd
from core import rapidocr_engine, spacing

_DOCS = Path(__file__).parent / "sample_docs"
_needs_ocr = pytest.mark.skipif(not rapidocr_engine.is_available(), reason="RapidOCR 미설치")


def _page(name):
    return Image.open(_DOCS / name).convert("RGB")


def _boxes(img):
    return fd._detect_caption_boxes(img, fd._to_gray_cv(img))


def test_finds_frame_with_caption_band():
    (box, band_y), = _boxes(_page("caption_box_p25.png"))
    x, y, w, h = box
    assert 150 < x < 190 and 700 < y < 750 and 830 < w < 880 and 690 < h < 740, box
    assert y + h * 0.8 < band_y < y + h   # 띠는 상자 맨 아래


def test_tinted_text_box_is_not_a_caption_box():
    # 278쪽 아래 파란 바탕 글상자(「이 고시에서 정한 요건확인의…」)는 띠 위에 그림이 없다 — 본문으로 읽혀야 한다
    boxes = _boxes(_page("caption_box_p278.png"))
    assert len(boxes) == 1 and boxes[0][0][1] < 250, boxes


def _synthetic(band_cells: bool):
    """흰 쪽에 테두리 상자 + 아래 옅은 파란 띠. band_cells 면 띠 안에 세로선(표의 색칠한 행)."""
    img = Image.new("RGB", (1000, 1400), "white")
    d = ImageDraw.Draw(img)
    d.rectangle((150, 300, 850, 800), outline=(90, 90, 90), width=2)
    d.rectangle((152, 740, 848, 798), fill=(205, 228, 247))
    if band_cells:
        for x in (300, 500, 700):
            d.line((x, 300, x, 800), fill=(90, 90, 90), width=2)
    return img


def test_synthetic_frame_detected_and_shaded_table_row_skipped():
    (box, band_y), = _boxes(_synthetic(band_cells=False))
    assert abs(box[1] - 300) <= 10 and abs(box[1] + box[3] - 800) <= 10 and abs(band_y - 740) <= 3, (box, band_y)
    assert _boxes(_synthetic(band_cells=True)) == []


def test_plain_page_has_no_caption_box():
    assert _boxes(Image.new("RGB", (1000, 1400), "white")) == []
    assert _boxes(Image.new("RGB", (1000, 1400), (240, 232, 205))) == []   # 누런 종이 — 바탕 전체가 띠가 되면 안 된다


def test_edge_strip_is_not_a_table():
    assert fd._is_edge_strip((879, 0, 306, 1364), 1238, 1750)        # 278쪽 책등 그림자
    assert not fd._is_edge_strip((206, 558, 827, 412), 1238, 1750)   # 34쪽 진짜 표
    assert not fd._is_edge_strip((0, 0, 400, 300), 400, 300)         # 표만 찍은 사진 — 쪽 전체가 표


def test_open_sided_table_box_widens_to_rule_ends():
    # 121쪽: 가로 괘선은 표 너비 전체, 세로선은 안쪽 둘뿐 — 세로선 안쪽만 잡히면 첫 열 「대금의 지급시기」가 「급시기」로 잘렸다
    img = Image.new("L", (1000, 600), 255)
    d = ImageDraw.Draw(img)
    for y in (200, 260, 320, 380):
        d.line((100, y, 900, y), fill=0, width=2)
    for x in (300, 600):
        d.line((x, 200, x, 380), fill=0, width=2)
    gray = fd.np.array(img)
    x, y, w, h = fd._expand_to_border(gray, (300, 200, 300, 180))
    assert x <= 102 and x + w >= 898 and abs(y - 200) <= 3 and abs(h - 180) <= 6, (x, y, w, h)
    # 그대로 둬야 하는 예: 옆 표 상자를 넘어가지 않는다
    x, y, w, h = fd._expand_to_border(gray, (300, 200, 300, 180), [(650, 190, 200, 200)])
    assert x <= 102 and x + w <= 650, (x, w)


def test_dense_grid_rejects_sparse_form_and_paragraph_cell():
    table = [["HS", "품목", ""], ["4011", "타이어", ""], ["", "", ""]]
    assert fd._dense_grid(table) == [["HS", "품목"], ["4011", "타이어"]]    # 빈 행·빈 열만 빠짐
    form = [["성명", "", "", ""], ["", "", "등기번호", ""], ["주소", "", "", ""], ["", "", "", "접수일"]]
    assert fd._dense_grid(form) == []                                       # 견본 서류: 빈 칸 75%
    assert fd._dense_grid([["가", "나"], ["다", "라" * 61]]) == []           # 문단이 든 칸


def test_split_prose_rows_moves_swallowed_paragraph_out():
    para = "최소담보에서 최대담보로 갈수록 보험자의 책임범위가 커지는 만큼 피보험자가 납부하여야 하는 보험료도 증가합니다."
    grid = [["", "(구) ICC", "(신) ICC"], ["최소담보", "ICC FPA", "ICC C"], [para, "", ""]]
    assert fd._split_prose_rows(grid) == (grid[:2], [para])
    # 그대로 둬야 하는 예: 짧은 한 칸 행(합계·주석), 두 칸 이상 찬 긴 행
    keep = [["구분", "내용"], ["※무역의존도(%). 2018년 기준", ""]]
    assert fd._split_prose_rows(keep) == (keep, [])
    wide = [["구분", "내용"], ["가" * 50, "나"]]
    assert fd._split_prose_rows(wide) == (wide, [])


def test_same_letters_needs_overlap_both_ways():
    assert fd._same_letters("1차 제네바 라운드 1947년", "1차 | 제네바 라운드 | 1947년")
    assert not fd._same_letters("INCHEON KOREA", "PreCarispe br Parof Ro INCHEON KOREA VeseWoyage fanher")   # 표에 잘못 읽은 글자
    assert not fd._same_letters("1차 제네바 라운드 자유무역 보호무역 도하개발아젠다", "1차 제네바 라운드")   # 표에 빠진 글자
    assert not fd._same_letters("", "")


@_needs_ocr
def test_document_photo_becomes_figure_with_caption_not_table():
    det = fd.detect(_page("caption_box_p25.png"))
    blocks = [r.table_md for r in det.regions if r.kind == "table"]
    assert len(blocks) == 1, blocks
    first, *rest = blocks[0].splitlines()
    assert first.startswith("[그림] ") and "운송" in first and "대신합니다" in first, first
    assert "|" not in blocks[0]                                  # 운송장 사진을 표로 쓰지 않는다
    assert all(l.startswith("(그림 속 글자: ") for l in rest), rest


@_needs_ocr
def test_page_with_edge_shadow_keeps_body_out_of_tables():
    det = fd.detect(_page("caption_box_p278.png"))
    blocks = [r for r in det.regions if r.kind == "table"]
    assert [b.table_md.startswith("[그림] 품목") for b in blocks] == [True], [b.table_md[:40] for b in blocks]
    assert all(r.x + r.w < 1100 for r in det.regions), [(r.kind, r.x, r.w) for r in det.regions]   # 가장자리 띠 없음


@pytest.mark.skipif(not spacing.is_available(), reason="kiwipiepy 미설치")
def test_compound_nouns_stay_glued_but_sentence_is_spaced():
    # 책이 붙여 쓴 복합명사는 그대로 — 예전엔 '운송 장'·'특 송 영수증'·'송 하인'·'기한 부' 로 낱말을 깼다
    assert spacing.fix_spacing("택배운송장이운송계약서의 기능을대신합니다")[0] == "택배운송장이 운송계약서의 기능을 대신합니다"
    out = spacing.fix_spacing("특송업체를통해보냈으면그증거로특송영수증을한장줄겁니다")[0]
    assert "특송업체를 통해 보냈으면" in out and "특송영수증을" in out and "특 송" not in out, out
    # 그대로 둬야 하는 예: 명사가 아닌 경계는 전처럼 띄운다
    assert spacing.fix_spacing("인류에게하나밖에없는")[0] == "인류에게 하나밖에 없는"
