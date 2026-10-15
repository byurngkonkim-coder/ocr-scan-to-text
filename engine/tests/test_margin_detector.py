"""스캔 PDF 머리글·바닥글 여백 자동 측정 — 본문 쪽 표본의 OCR 줄 상자로 잰다.

요구(2026-10-15 사용자): 텍스트 레이어 없는 PDF 는 먼저 본문 쪽에서 위·아래 여백(쪽번호·장 제목)을
재고, 여백을 뺀 부분만 인식한다. 앞부분(그림·목차)은 측정에서 건너뛴다.
줄 상자는 (위, 아래, 왼쪽, 오른쪽) — 쪽 높이·너비 대비 비율.
"""
from PIL import Image, ImageDraw

from core import margin_detector as md

PITCH = 0.034   # 줄 간격
LH = 0.025      # 줄 높이


def _body(top=0.18, n=20, last_short=True):
    lines = [(top + i * PITCH, top + i * PITCH + LH, 0.20, 0.88) for i in range(n)]
    if last_short:   # 문단 끝 짧은 줄 — 본문이다
        t = top + n * PITCH
        lines.append((t, t + LH, 0.20, 0.45))
    return lines


HEADER = [(0.066, 0.087, 0.21, 0.25), (0.090, 0.110, 0.22, 0.30)]   # 쪽번호 + 장 제목 (이한줄 실측 모양)
FOOTER = [(0.930, 0.950, 0.75, 0.86)]                                # 쪽 꼬리 (예측기계 실측 모양)


def test_header_and_footer_bands_found():
    pages = [HEADER + _body() + FOOTER for _ in range(8)]
    m = md.measure_from_boxes(pages)
    assert m is not None
    assert 0.110 < m.top < 0.18          # 머리글 아래 ~ 본문 첫 줄 위 사이에서 자른다
    body_bottom = 0.18 + 20 * PITCH + LH
    assert body_bottom < m.bottom < 0.930  # 짧은 마지막 줄까지 살리고 쪽 꼬리 위에서 자른다


def test_no_header_means_no_top_cut():
    pages = [_body(top=0.09) + FOOTER for _ in range(8)]
    m = md.measure_from_boxes(pages)
    assert m.top == 0.0 and m.bottom < 0.930


def test_band_must_repeat_on_many_pages():
    # 한두 쪽에만 있는 위쪽 줄(장 번호 등)로는 자르지 않는다
    pages = [_body() + FOOTER for _ in range(8)]
    pages[0] = HEADER + pages[0]
    m = md.measure_from_boxes(pages)
    assert m.top == 0.0


def test_footnote_below_gap_is_body_not_footer():
    # 본문 아래 간격을 두고 붙은 여러 줄 각주(넓은 줄)는 본문 — 잘리면 안 된다
    note = [(0.86, 0.875, 0.20, 0.88), (0.878, 0.893, 0.20, 0.70)]
    pages = [_body(n=17) + FOOTER for _ in range(8)]
    pages[3] = _body(n=17) + note + FOOTER
    m = md.measure_from_boxes(pages)
    assert m.bottom > 0.893


def test_body_never_cut_on_chapter_start_page():
    # 장 첫 쪽: 본문이 아래에서 시작 — 위 자름선은 다른 쪽 본문 첫 줄보다 위여야 한다
    pages = [HEADER + _body() + FOOTER for _ in range(7)]
    pages.append([(0.10, 0.13, 0.45, 0.55)] + _body(top=0.45, n=12) + FOOTER)
    m = md.measure_from_boxes(pages)
    assert m.top < 0.18


def test_too_few_body_pages_gives_none():
    figure_pages = [[(0.4, 0.43, 0.4, 0.6)] for _ in range(8)]
    assert md.measure_from_boxes(figure_pages) is None


def test_sample_pages_skip_front_matter():
    s = md.sample_pages(325)
    assert min(s) > 325 * 0.1 and max(s) < 325 * 0.95 and len(s) == md.SAMPLE_PAGES
    assert md.sample_pages(7) == list(range(1, 8))      # 짧은 문서는 전 쪽


def _page(*bars):
    """흰 쪽(100×200)에 검은 글줄 막대(y0, y1)."""
    img = Image.new("RGB", (100, 200), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for y0, y1 in bars:
        d.rectangle([10, y0, 90, y1], fill=(0, 0, 0))
    return img


M = md.Margins(top=0.1, bottom=0.9, body_pages=5, sampled=8, line_h=0.03, skip_pages=frozenset({2}))
BLACK, WHITE = (0, 0, 0), (255, 255, 255)


def test_apply_masks_header_and_footer():
    out = md.apply(_page((5, 12), (40, 46), (186, 192)), M, page_num=5)
    assert out.getpixel((50, 8)) == WHITE      # 머리글 가림
    assert out.getpixel((50, 43)) == BLACK     # 본문 그대로
    assert out.getpixel((50, 189)) == WHITE    # 쪽 꼬리 가림


def test_line_crossing_cut_is_kept_whole():
    # 위 자름선(20행)·아래 자름선(180행)을 가로지르는 줄 → 반쪽 자르지 않고 통째로 살린다
    out = md.apply(_page((16, 24), (176, 184)), M, page_num=5)
    assert out.getpixel((50, 17)) == BLACK and out.getpixel((50, 23)) == BLACK
    assert out.getpixel((50, 177)) == BLACK and out.getpixel((50, 183)) == BLACK


def test_skipped_front_page_untouched():
    out = md.apply(_page((5, 12), (186, 192)), M, page_num=2)   # 표지·머리말
    assert out.getpixel((50, 8)) == BLACK and out.getpixel((50, 189)) == BLACK


MS = md.Margins(top=0.1, bottom=0.9, body_pages=5, sampled=8,
                top_x=((0.21, 0.25), (0.22, 0.30)), bottom_x=((0.69, 0.86),))


def test_long_line_in_band_marks_front_page():
    header = [(0.05, 0.07, 0.20, 0.27)]                 # 짧은 머리글 → 가려도 됨
    title = [(0.05, 0.08, 0.15, 0.85)]                  # 표지 제목·머리말 본문 줄 → 그 쪽은 안 가림
    assert not md.front_page_unsafe(header, MS)
    assert md.front_page_unsafe(title, MS)
    assert not md.front_page_unsafe([(0.4, 0.43, 0.1, 0.9)], MS)   # 띠 밖 긴 줄은 상관없음


def test_short_line_off_header_position_marks_front_page():
    # 이한줄 실측: 머리글은 왼쪽 끝(가운데 x 0.13~0.28), 4·5쪽 「엮은이 / 김동범」은 가운데(0.52)
    author = [(0.050, 0.075, 0.48, 0.55), (0.075, 0.100, 0.48, 0.56)]
    assert md.front_page_unsafe(author, MS)
    footer_right = [(0.93, 0.95, 0.75, 0.86)]          # 본문 쪽 쪽 꼬리와 같은 자리 → 가려도 됨
    assert not md.front_page_unsafe(footer_right, MS)


def test_header_positions_recorded_from_body_pages():
    pages = [HEADER + _body() + FOOTER for _ in range(8)]
    m = md.measure_from_boxes(pages)
    assert set(m.top_x) == {(0.21, 0.25), (0.22, 0.30)}
    assert set(m.bottom_x) == {(0.75, 0.86)}


def test_footer_with_longer_chapter_title_still_matches():
    # 오른쪽 맞춤 쪽 꼬리: 장 제목이 길면 왼쪽 끝·가운데는 움직여도 오른쪽 끝은 같다
    assert not md.front_page_unsafe([(0.93, 0.95, 0.55, 0.86)], MS)


def test_gap_inside_a_line_is_not_a_line_break():
    # 한 글줄 안의 1행짜리 빈 틈(16~19 · 21~24 사이 20행)에서 자르지 않는다
    out = md.apply(_page((16, 19), (21, 24)), M, page_num=5)
    assert out.getpixel((50, 17)) == BLACK and out.getpixel((50, 23)) == BLACK


def test_measuring_does_not_switch_off_recognition():
    # RapidOCR 는 engine(img, use_rec=False) 의 설정을 엔진에 그대로 남긴다 — 여백 측정 뒤
    # 본 OCR 이 글자를 하나도 못 읽던 문제(2026-10-15 전체 테스트에서 발견)
    from core import rapidocr_engine
    eng = rapidocr_engine.get_engine()
    before = (eng.use_det, eng.use_cls, eng.use_rec)
    md._detect_boxes(_page((40, 60), (80, 100)))
    assert (eng.use_det, eng.use_cls, eng.use_rec) == before


def test_cli_masks_pages_with_measured_margins(monkeypatch, tmp_path):
    # 명령줄 도구(ocr_tool)도 웹 화면과 같은 측정·가림을 쓴다
    import ocr_tool
    from core import config
    monkeypatch.setattr(config, "get_auto_margin", lambda: True)
    monkeypatch.setattr(md, "measure", lambda path, total: M)
    pages = [_page((5, 12), (40, 46)) for _ in range(3)]
    out = ocr_tool._mask_margins(tmp_path / "x.pdf", pages, verbose=False)
    assert out[1].getpixel((50, 8)) == BLACK      # 2쪽은 skip_pages — 그대로
    assert out[2].getpixel((50, 8)) == WHITE      # 3쪽 머리글 가림
    assert out[2].getpixel((50, 43)) == BLACK     # 본문 그대로
    monkeypatch.setattr(config, "get_auto_margin", lambda: False)
    assert ocr_tool._mask_margins(tmp_path / "x.pdf", pages, verbose=False) is pages
