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
