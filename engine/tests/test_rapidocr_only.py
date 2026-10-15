"""로컬 OCR 은 RapidOCR 한글 v5 하나만 쓴다 — Tesseract 경로는 없다.

배경: RapidOCR 오류·미설치 때 Tesseract 로 조용히 넘어가 한글 품질이 크게 떨어진 결과를 '완료'로 냈고,
smart_run 은 한국어 쪽을 언어만 바꿔 두 번 읽으며 모델을 쪽마다 다시 불러왔다(2026-10-13 예측기계).
RapidOCR 이 실패해도 처리를 멈추지 않는다 — 신뢰도 0 결과로 돌려 Vision 폴백·검토 필요로 잇는다.
"""
from pathlib import Path

from PIL import Image

from core import blank_detector, rapidocr_engine
from core.types import OCREngine, OCRResult


def _ok(text="본문 글자입니다"):
    return OCRResult(text=text, confidence=95.0, engine=OCREngine.RAPIDOCR, has_jamo_split=False)


def test_rapidocr_error_becomes_zero_confidence_result(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("onnx 오류")
    monkeypatch.setattr(rapidocr_engine, "run", boom)
    res = rapidocr_engine.smart_run(Image.new("RGB", (50, 50), "white"))
    assert res.text == "" and res.confidence == 0.0 and res.engine == OCREngine.RAPIDOCR


def test_korean_page_read_once(monkeypatch):
    calls = []
    monkeypatch.setattr(rapidocr_engine, "run",
                        lambda img, lang="kor+eng", fast=False, refine=True: calls.append(lang) or _ok())
    rapidocr_engine.smart_run(Image.new("RGB", (50, 50), "white"))
    assert calls == ["kor+eng"]


def test_english_page_reread_with_english_model(monkeypatch):
    calls = []
    monkeypatch.setattr(rapidocr_engine, "run",
                        lambda img, lang="kor+eng", fast=False, refine=True:
                        calls.append(lang) or _ok("This page is almost entirely English text"))
    rapidocr_engine.smart_run(Image.new("RGB", (50, 50), "white"))
    assert calls == ["kor+eng", "eng"]


def test_blank_check_counts_with_rapidocr(monkeypatch):
    monkeypatch.setattr(rapidocr_engine, "read_boxes",
                        lambda img, lang="kor+eng": [("가나다 abc", 0.9, (0, 0, 10, 10)), ("·", 0.5, (0, 0, 1, 1))])
    assert blank_detector._quick_text_count(Image.new("RGB", (3000, 2000), "white")) == 6


def test_no_tesseract_left_in_engine_code():
    engine = Path(__file__).resolve().parents[1]
    hits = [str(p.relative_to(engine)) for p in engine.rglob("*.py")
            if "venv" not in p.parts and "tests" not in p.parts
            and "pytesseract" in p.read_text(encoding="utf-8", errors="ignore")]
    assert hits == []
    assert not (engine / "core" / "tesseract_engine.py").exists()
    assert not hasattr(OCREngine, "TESSERACT")


def test_textline_angle_classifier_off(monkeypatch):
    # 방향 판별(cls)이 짧은 줄을 뒤집힌 글자로 보고 180° 돌려 읽어 줄이 깨지거나 빠졌다 (2026-10-13 예측기계 30·150쪽)
    import rapidocr
    seen = []
    monkeypatch.setattr(rapidocr, "RapidOCR", lambda params=None: seen.append(params) or object())
    for lang in ("kor+eng", "eng"):
        monkeypatch.setattr(rapidocr_engine, "_ENGINE_INSTANCE", None)
        rapidocr_engine.get_engine(lang)
    assert [p["Global.use_cls"] for p in seen] == [False, False]


def _box(x0, y0, x1, h=30):
    return [[x0, y0], [x1, y0], [x1, y0 + h], [x0, y0 + h]]


def test_indented_line_marked_as_paragraph_start():
    # 예측기계 158쪽 실측: OCR 이 마침표를 빠뜨리고('…따라 달라진다') 줄이 끝까지 차 있어 다음 문단
    # ('인간과 기계는…', 한 글자 들여씀)이 앞 문장 끝에 붙었다 — 상자 좌표의 들여쓰기로 문단 시작을 표시
    lines = ["따라 달라진다", "인간과 기계는 데이터의", "위한 데이터를 축적한다", "2부 의사결정"]
    boxes = [_box(100, 0, 1000), _box(130, 50, 1000), _box(100, 100, 1000), _box(100, 200, 300)]
    out = rapidocr_engine._mark_paragraph_starts(lines, boxes)
    assert out == ["따라 달라진다", "  인간과 기계는 데이터의", "위한 데이터를 축적한다", "2부 의사결정"]


def test_block_quote_and_heading_not_marked():
    # 인용 단락(여러 줄 모두 들여씀)·가운데 제목(크게 들여씀)은 문단 시작으로 보지 않는다
    lines = ["인용 첫 줄", "인용 둘째 줄", "본문 줄", "가운데 제목", "본문 끝 줄"]
    boxes = [_box(130, 0, 1000), _box(130, 50, 1000), _box(100, 100, 1000),
             _box(400, 150, 700), _box(100, 200, 1000)]
    assert rapidocr_engine._mark_paragraph_starts(lines, boxes) == lines


def test_hanging_indent_after_bullet_not_marked():
    # 예측기계 28·34쪽 실측: 각주 '• 다이얼의 … 다이얼이 나' 다음 줄('온다.')이 내어쓰기로 들어가 있어
    # 문단 첫 줄로 잘못 표시돼 '나 ‖ 온다' 로 갈라졌다. OCR 은 글머리 '•' 를 읽지 않는다
    lines = ["본문 줄", "다이얼의 최고 숫자는 보통 10이지만 다이얼이 나", "온다.", "34"]
    boxes = [_box(100, 0, 1000), _box(100, 50, 1000), _box(130, 100, 200), _box(100, 200, 140)]
    assert rapidocr_engine._mark_paragraph_starts(lines, boxes) == lines


def _page(last_text, last_x1, gap_rows):
    # 본문 6줄(줄 간격 50, 높이 30) + 맨 아래 한 줄(gap_rows 줄 간격만큼 떨어짐)
    lines = [f"본문 {k}번째 줄입니다" for k in range(6)] + [last_text]
    boxes = [_box(100, 50 * k, 1000) for k in range(6)] + [_box(100, 250 + 50 * gap_rows, last_x1)]
    return lines, boxes


def test_footer_separated_by_gap_dropped():
    # 예측기계 표본 40쪽 실측: 꼬리말은 앞 줄과 줄 간격 2.15배 이상 떨어져 있고 짧다(39/39).
    # 꼬리말 아닌 쪽 마지막 줄은 0.9배 이하
    for footer in ("28", "피터 드러커의 매니지먼트 029", "1582부의사결정"):
        lines, boxes = _page(footer, 400, 2.3)
        out_lines, out_boxes = rapidocr_engine._drop_page_footer(lines, boxes)
        assert out_lines == lines[:-1] and len(out_boxes) == 6, footer


def test_last_line_kept_when_close_long_or_no_digit():
    for text, x1, gap in (("그림 이름표 12", 400, 0.9),        # 본문과 붙어 있음
                          ("숫자 2026년이 든 긴 각주 줄", 900, 2.3),   # 짧지 않음
                          ("예측", 400, 2.3)):                 # 쪽번호 없음
        lines, boxes = _page(text, x1, gap)
        assert rapidocr_engine._drop_page_footer(lines, boxes)[0] == lines, text


def test_two_column_index_read_left_then_right():
    # 세일즈왕 343~346쪽 실측(2026-10-15): 2단 인명 색인을 같은 높이끼리 좌·우 번갈아 읽어
    # 텍스트층 대비 순서 일치 0.57. 맨 위 쪽번호처럼 한쪽에만 있는 줄도 왼쪽 단에 붙는다
    left = [(f"왼쪽{k}", _box(100, 100 + 50 * k, 400)) for k in range(6)]
    right = [(f"오른쪽{k}", _box(600, 100 + 50 * k, 950)) for k in range(6)]
    mixed = [("340", _box(100, 0, 160))] + [x for pair in zip(left, right) for x in pair]
    lines, boxes = rapidocr_engine._order_columns([t for t, _ in mixed], [b for _, b in mixed])
    assert lines == ["340"] + [t for t, _ in left] + [t for t, _ in right]
    assert len(boxes) == len(lines)


def test_full_width_heading_splits_column_blocks():
    # 단 위·아래의 넓은 줄(제목·본문)은 그 자리를 지키고, 그 사이 블록만 단 순서로 읽는다
    left = [(f"L{k}", _box(100, 100 + 50 * k, 400)) for k in range(4)]
    right = [(f"R{k}", _box(600, 100 + 50 * k, 950)) for k in range(4)]
    mixed = ([("찾아보기", _box(100, 0, 950))] + [x for pair in zip(left, right) for x in pair]
             + [("본문 마지막 넓은 줄", _box(100, 400, 950))])
    lines, _ = rapidocr_engine._order_columns([t for t, _ in mixed], [b for _, b in mixed])
    assert lines == ["찾아보기"] + [t for t, _ in left] + [t for t, _ in right] + ["본문 마지막 넓은 줄"]


def test_single_column_and_two_cell_table_untouched():
    # 본문 쪽: 대부분 줄이 가운데를 가로지른다 → 단 없음
    lines = [f"본문 {k}" for k in range(6)] + ["짧은 끝줄", "오른쪽 쪽번호", "또 짧은", "오른 끝"]
    boxes = ([_box(100, 50 * k, 950) for k in range(6)]
             + [_box(100, 300, 300), _box(800, 300, 950), _box(100, 350, 250), _box(800, 350, 950)])
    assert rapidocr_engine._order_columns(lines, boxes)[0] == lines
    # 표: 좁은 항목 칸 + 넓은 내용 칸 → 행 순서 유지(칸 너비가 크게 다르면 단으로 보지 않는다)
    rows = [(f"항목{k}", _box(100, 50 * k, 200)) for k in range(5)]
    vals = [(f"내용 설명 {k}", _box(300, 50 * k, 950)) for k in range(5)]
    mixed = [x for pair in zip(rows, vals) for x in pair]
    t = [s for s, _ in mixed]
    assert rapidocr_engine._order_columns(t, [b for _, b in mixed])[0] == t
