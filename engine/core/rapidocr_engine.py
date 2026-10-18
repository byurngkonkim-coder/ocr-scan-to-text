"""RapidOCR (ONNX Runtime) 엔진 — 이 도구의 유일한 로컬 OCR (한글 v5 모델).

별도 프로그램 설치 없이 순수 Python 및 ONNX Runtime만으로 CPU에서 고속·고정확도 텍스트 인식을 수행합니다.
본문 판독(smart_run)·그림 감지·빈 쪽 판정(read_boxes)이 모두 이 엔진을 쓴다.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np
from PIL import Image

from core import config, hanja_reader
from core.char_normalizer import has_jamo_decomposition
from core.types import OCREngine, OCRResult

logger = logging.getLogger(__name__)

# 지연 로드 싱글톤 인스턴스
_ENGINE_INSTANCE = None
_ENGINE_LANG = None
_RETRY_LOCK = threading.Lock()   # 판독 실패 쪽 재시도는 한 번에 하나씩


def is_available() -> bool:
    """RapidOCR 및 ONNX Runtime 사용 가능 여부 확인."""
    try:
        import rapidocr
        import onnxruntime
        return True
    except ImportError:
        return False


def engine_params(lang: str = "kor+eng") -> dict:
    """RapidOCR 설정 — 본문 판독과 표 칸 판독(core.table_structure)이 함께 쓴다."""
    from rapidocr.utils.typings import OCRVersion, LangRec, ModelType

    # 방향 판별(cls) 끔 — 짧은 줄을 뒤집힌 글자로 보고 180° 돌려 읽어 줄이 깨지거나 빠졌다(2026-10-13)
    params = {"Global.use_cls": False}
    # 한국어 또는 한영 혼합인 경우 한국어 특화 v5 모바일 인식 모델 적용
    if any(k in lang.lower() for k in ("kor", "korean", "ko")):
        params.update({
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Rec.lang_type": LangRec.KOREAN,
            "Rec.model_type": ModelType.MOBILE,
        })
    return params


def get_engine(lang: str = "kor+eng"):
    """지연 초기화된 RapidOCR 엔진 인스턴스 반환."""
    global _ENGINE_INSTANCE, _ENGINE_LANG

    # 언어 매핑이 동일하면 기존 인스턴스 재사용
    if _ENGINE_INSTANCE is not None and _ENGINE_LANG == lang:
        return _ENGINE_INSTANCE

    from rapidocr import RapidOCR

    try:
        _ENGINE_INSTANCE = RapidOCR(params=engine_params(lang))
        _ENGINE_LANG = lang
    except Exception as e:
        logger.warning(f"RapidOCR Korean 모델 초기화 실패, 기본 모델로 전환: {e}")
        _ENGINE_INSTANCE = RapidOCR()
        _ENGINE_LANG = lang

    return _ENGINE_INSTANCE


def _read(engine, rgb_array: np.ndarray):
    """한글 엔진 호출 — 언제나 같은 인자로 부른다.

    RapidOCR 은 호출 인자(text_score·return_word_box 등)를 엔진 설정에 남긴다. 쪽 판독은 여러 스레드가
    한 엔진을 함께 쓰므로 호출마다 인자가 다르면 서로 설정을 덮어쓴다 — 점수 문턱은 받은 뒤 직접 건다.
    """
    return engine(rgb_array, use_det=True, use_cls=False, use_rec=True,
                  return_word_box=True, return_single_char_box=True, text_score=0.0)


def read_boxes(image: Image.Image, lang: str = "kor+eng") -> list[tuple[str, float, tuple[int, int, int, int]]]:
    """이미지의 글자 줄마다 (글자, 신뢰도 0~1, 상자 x·y·w·h) — 그림·표 감지에서 쓴다."""
    result = _read(get_engine(lang=lang), np.array(image.convert("RGB")))
    if result is None or getattr(result, "txts", None) is None or getattr(result, "boxes", None) is None:
        return []
    out = []
    for text, score, box in zip(result.txts, result.scores, result.boxes):
        if float(score) < hanja_reader.TEXT_SCORE:
            continue
        pts = np.asarray(box)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        out.append((str(text).strip(), float(score), (int(x0), int(y0), int(x1 - x0), int(y1 - y0))))
    return out


def _drop_page_footer(lines: list[str], boxes) -> tuple[list[str], list]:
    """쪽 맨 아래 줄이 본문과 떨어진 짧은 쪽번호 줄(꼬리말)이면 뺀다.

    꼬리말은 본문 아래로 넉넉히 떨어져 있다 — 예측기계 표본 40쪽 실측: 꼬리말 39개는 앞 줄과
    줄 간격의 2.15배 이상, 꼬리말 아닌 마지막 줄은 0.9배 이하(2026-10-14). 모양이 책마다 달라
    ('2장 …19'·'1582부의사결정'·'피터 드러커의 매니지먼트 029') 글자 규칙만으로는 다 못 잡고,
    줄 잇기 전에 떼야 본문 끝에 붙지 않는다.
    꼬리말 = 마지막 줄(같은 높이 상자 묶음) + 앞 줄과 1.5배 이상 떨어짐 + 본문 너비 40% 이하 + 숫자 포함.
    위쪽 머리글은 장 시작 쪽의 '11장'·'3부' 와 구분되지 않아 다루지 않는다.
    """
    if len(lines) != len(boxes) or len(lines) < 4:
        return lines, boxes
    y0 = [min(p[1] for p in b) for b in boxes]
    y1 = [max(p[1] for p in b) for b in boxes]
    x0 = [min(p[0] for p in b) for b in boxes]
    x1 = [max(p[0] for p in b) for b in boxes]
    h = float(np.median([y1[i] - y0[i] for i in range(len(lines))]))
    low = max(y0)
    foot = [i for i in range(len(lines)) if y0[i] >= low - 0.5 * h]
    rest = [i for i in range(len(lines)) if i not in foot]
    rows = sorted(y0[i] for i in rest)
    steps = [b - a for a, b in zip(rows, rows[1:]) if b - a > 0.5 * h]   # 같은 줄 상자는 건너뜀
    if len(steps) < 2:
        return lines, boxes
    pitch = float(np.median(steps))
    gap = min(y0[i] for i in foot) - max(y1[i] for i in rest)
    width = max(x1[i] for i in foot) - min(x0[i] for i in foot)
    body = max(x1[i] for i in rest) - min(x0[i] for i in rest)
    text = " ".join(lines[i] for i in foot)
    if gap >= 1.5 * pitch and width <= 0.4 * body and any(c.isdigit() for c in text):
        return [lines[i] for i in rest], [boxes[i] for i in rest]
    return lines, boxes


def _order_columns(lines: list[str], boxes) -> tuple[list[str], list]:
    """2단으로 짠 쪽(색인 등)을 왼쪽 단 → 오른쪽 단 순서로 다시 늘어놓는다.

    RapidOCR 은 위에서 아래로, 같은 높이면 왼쪽→오른쪽으로 내보내 두 단의 줄이 번갈아 섞인다
    (세일즈왕 343~346쪽 2단 인명 색인: 글자는 99% 맞는데 텍스트층 대비 순서 일치 0.57, 2026-10-15).
    단 사이 여백 = 쪽 가운데 30~70% 에서 상자가 가장 적게 가로지르는 세로 띠.
    단으로 보는 조건: 여백을 가로지르는 줄이 20% 이하 · 양쪽 3줄 이상 · 양쪽이 같은 높이대에 나란히 ·
    두 단 너비 비 0.6 이상(좁은 항목 칸 + 넓은 내용 칸 표는 행 순서를 지킨다).
    여백을 가로지르는 넓은 줄(제목·본문)은 제자리를 지키고, 그 사이 블록마다 왼쪽 단 → 오른쪽 단.
    단 안의 순서는 RapidOCR 순서를 그대로 쓴다.
    """
    n = len(lines)
    if n != len(boxes) or n < 8:
        return lines, boxes
    x0 = [min(p[0] for p in b) for b in boxes]
    x1 = [max(p[0] for p in b) for b in boxes]
    yc = [(min(p[1] for p in b) + max(p[1] for p in b)) / 2 for b in boxes]
    lo, hi = min(x0), max(x1)
    span = hi - lo
    if span <= 0:
        return lines, boxes
    xs = np.linspace(lo + 0.3 * span, lo + 0.7 * span, 81)
    cross_n = [sum(1 for i in range(n) if x0[i] < x < x1[i]) for x in xs]
    best = min(cross_n)
    # 가장 덜 가로지르는 x 들 중 가장 긴 연속 구간의 가운데를 여백으로 본다
    runs, start = [], None
    for k, c in enumerate(cross_n + [best + 1]):
        if c == best and start is None:
            start = k
        elif c != best and start is not None:
            runs.append((k - start, start, k - 1))
            start = None
    _, a, b = max(runs)
    g = (xs[a] + xs[b]) / 2

    cross = [i for i in range(n) if x0[i] < g < x1[i]]
    left = [i for i in range(n) if x1[i] <= g]
    right = [i for i in range(n) if x0[i] >= g]
    if len(cross) > 0.2 * n or len(left) < 3 or len(right) < 3:
        return lines, boxes
    lw = max(x1[i] for i in left) - min(x0[i] for i in left)
    rw = max(x1[i] for i in right) - min(x0[i] for i in right)
    if min(lw, rw) < 0.6 * max(lw, rw):
        return lines, boxes
    ly = (min(yc[i] for i in left), max(yc[i] for i in left))
    ry = (min(yc[i] for i in right), max(yc[i] for i in right))
    overlap = min(ly[1], ry[1]) - max(ly[0], ry[0])
    if overlap < 0.5 * min(ly[1] - ly[0], ry[1] - ry[0]):
        return lines, boxes

    seps = sorted(cross, key=lambda i: yc[i])
    def block(i):   # 이 줄 위에 있는 넓은 줄 수 = 블록 번호
        return sum(1 for s in seps if yc[s] < yc[i])
    order = []
    for k in range(len(seps) + 1):
        order += [i for i in left if block(i) == k] + [i for i in right if block(i) == k]
        if k < len(seps):
            order.append(seps[k])
    return [lines[i] for i in order], [boxes[i] for i in order]


def _mark_paragraph_starts(lines: list[str], boxes) -> list[str]:
    """들여 쓴 줄(문단 첫 줄) 앞에 공백 2칸을 붙인다 — postprocessor.join_broken_lines 가 앞 줄과 잇지 않는다.

    OCR 이 문단 끝 마침표를 빠뜨리면('…따라 달라진다') 줄 잇기 규칙이 다음 문단을 앞 문장 끝에 붙였다
    (예측기계 158쪽, 2026-10-14). 글자만으로는 문단 시작을 알 수 없어 상자 좌표의 들여쓰기를 쓴다.
    문단 첫 줄 = 왼쪽 여백보다 0.6~3.5줄높이 들어가 있고, 오른쪽 끝까지 차 있고, 앞·뒤 줄은 여백에 붙은 줄.
    여러 줄이 모두 들여 쓴 인용 단락과 크게 들여 쓴 가운데 제목은 표시하지 않는다.
    오른쪽까지 차야 하는 까닭: 각주 둘째 줄('온다.')이 내어쓰기로 들어가 있는데 OCR 이 글머리 '•' 를
    읽지 않아 문단 첫 줄로 잘못 표시됐다(예측기계 28·34쪽).
    """
    if len(lines) != len(boxes) or len(lines) < 3:
        return lines
    x0 = [min(p[0] for p in b) for b in boxes]
    x1 = [max(p[0] for p in b) for b in boxes]
    h = float(np.median([max(p[1] for p in b) - min(p[1] for p in b) for b in boxes]))
    full = max(x1[i] - x0[i] for i in range(len(lines))) * 0.8
    wide = [i for i in range(len(lines)) if x1[i] - x0[i] >= full]
    margin = float(np.median([x0[i] for i in wide]))
    right = float(np.median([x1[i] for i in wide]))
    at_margin = [abs(x - margin) < 0.4 * h for x in x0]
    out = []
    for i, line in enumerate(lines):
        indent = x0[i] - margin
        if (0.6 * h <= indent <= 3.5 * h and x1[i] >= right - h and (i == 0 or at_margin[i - 1])
                and i + 1 < len(lines) and at_margin[i + 1]):
            line = "  " + line
        out.append(line)
    return out


def run(image: Image.Image, lang: str = "kor+eng", fast: bool = False, refine: bool = True) -> OCRResult:
    """
    이미지에 RapidOCR(ONNX)을 실행하고 OCRResult 반환.

    Args:
        image: PIL 이미지
        lang: 언어 코드 (kor+eng, kor, eng 등)
        fast: 호환성 유지용 플래그 (RapidOCR은 기본적으로 고속 실행됨)
        refine: 한국어 띄어쓰기, 맞춤법 및 OCR 노이즈 정제 규칙 자동 적용 여부

    Returns:
        OCRResult (text, confidence, engine, has_jamo_split)
    """
    engine = get_engine(lang=lang)

    # PIL Image -> RGB numpy array 변환
    if image.mode != "RGB":
        image_rgb = image.convert("RGB")
    else:
        image_rgb = image

    rgb_array = np.array(image_rgb)

    # RapidOCR 실행
    result = _read(engine, rgb_array)

    if result is None or getattr(result, "txts", None) is None or not result.txts:
        return OCRResult(
            text="",
            confidence=100.0,
            engine=OCREngine.RAPIDOCR,
            has_jamo_split=False,
        )

    # 한국어 모델이면 한자 자리를 중국어 모델로 다시 읽어 끼운다(core.hanja_reader). 끄면 한글만.
    korean = any(k in lang.lower() for k in ("kor", "korean", "ko"))
    if korean and config.get_hanja_reading():
        lines, line_boxes, line_scores = hanja_reader.fill(rgb_array, result)
    else:
        lines, line_boxes, line_scores = [], [], []
        for text, score, box in zip(result.txts, result.scores, result.boxes):
            if float(score) >= hanja_reader.TEXT_SCORE and str(text).strip():
                lines.append(str(text).strip()); line_boxes.append(box); line_scores.append(float(score))
    # 0.0~1.0 score -> 0.0~100.0 스케일. 한자로 되살린 한문 줄(점수 0)은 쪽 신뢰도에 넣지 않는다
    confs = [s * 100.0 for s in line_scores if s > 0]

    if len(line_boxes) == len(lines):
        lines, line_boxes = _order_columns(lines, line_boxes)
        lines, line_boxes = _drop_page_footer(lines, line_boxes)
    full_text = "\n".join(_mark_paragraph_starts(lines, line_boxes))
    avg_conf = sum(confs) / len(confs) if confs else 0.0

    # MyOCR의 고도화된 한국어 정제기 적용
    if refine and full_text.strip():
        try:
            from core.korean_corrector import refine_korean_text
            full_text = refine_korean_text(full_text)
        except Exception as e:
            logger.debug(f"korean_corrector 적용 생략: {e}")

    # 자모 분리 감지
    jamo_split = has_jamo_decomposition(full_text)
    if jamo_split:
        avg_conf = 0.0

    return OCRResult(
        text=full_text,
        confidence=avg_conf,
        engine=OCREngine.RAPIDOCR,
        has_jamo_split=jamo_split,
    )


def smart_run(image: Image.Image, default_lang: str = "kor+eng", fast: bool = False, refine: bool = True) -> OCRResult:
    """
    로컬 OCR 진입점 — 한 번 읽고, 거의 영문인 쪽만 영문 모델로 다시 읽는다.

    한국어 모델은 'kor'·'kor+eng' 가 같은 모델이라 한국어 쪽은 다시 읽지 않는다.
    RapidOCR 이 없거나 실패하면 빈 결과(신뢰도 0)를 돌려준다 — 처리를 멈추지 않고,
    호출부가 저신뢰 쪽처럼 Vision 폴백하거나 '검토 필요'로 남긴다.
    """
    def once():
        res = run(image, lang=default_lang, fast=fast, refine=refine)
        if default_lang != "eng" and _mostly_english(res.text):
            res = run(image, lang="eng", fast=fast, refine=refine)
        return res
    try:
        return once()
    except Exception as e:
        # 일시 오류면 한 번 더 — 다른 작업과 겹치지 않게 혼자 읽는다. 대한민국CEO명언집 일괄 처리에서 글자가
        # 또렷한 53쪽이 신뢰도 0 으로 비었는데, 단독으로 다시 읽으니 90~96% 로 정상이었다(2026-10-17)
        logger.warning(f"RapidOCR 판독 실패 — 한 번 더 읽음: {e}")
        try:
            with _RETRY_LOCK:
                return once()
        except Exception as e2:
            logger.warning(f"RapidOCR 판독 재시도 실패 — 신뢰도 0 으로 넘김: {e2}")
            return OCRResult(text="", confidence=0.0, engine=OCREngine.RAPIDOCR, has_jamo_split=False)


def _mostly_english(text: str) -> bool:
    """영문 글자가 영문+한글의 85% 를 넘는 쪽."""
    eng = sum(1 for c in text if ord(c) < 128 and c.isalpha())
    kor = sum(1 for c in text if "가" <= c <= "힣")
    return eng + kor > 0 and eng / (eng + kor) > 0.85

