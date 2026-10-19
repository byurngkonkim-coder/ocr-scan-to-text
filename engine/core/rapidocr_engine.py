"""RapidOCR (ONNX Runtime) 엔진 — 이 도구의 유일한 로컬 OCR (한글 v5 모델).

별도 프로그램 설치 없이 순수 Python 및 ONNX Runtime만으로 CPU에서 고속·고정확도 텍스트 인식을 수행합니다.
본문 판독(smart_run)·그림 감지·빈 쪽 판정(read_boxes)이 모두 이 엔진을 쓴다.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np
from PIL import Image

from core.char_normalizer import has_jamo_decomposition
from core.types import OCREngine, OCRResult

logger = logging.getLogger(__name__)

# 지연 로드 싱글톤 인스턴스
_ENGINE_INSTANCE = None
_ENGINE_LANG = None


def is_available() -> bool:
    """RapidOCR 및 ONNX Runtime 사용 가능 여부 확인."""
    try:
        import rapidocr
        import onnxruntime
        return True
    except ImportError:
        return False


def get_engine(lang: str = "kor+eng"):
    """지연 초기화된 RapidOCR 엔진 인스턴스 반환."""
    global _ENGINE_INSTANCE, _ENGINE_LANG

    # 언어 매핑이 동일하면 기존 인스턴스 재사용
    if _ENGINE_INSTANCE is not None and _ENGINE_LANG == lang:
        return _ENGINE_INSTANCE

    from rapidocr import RapidOCR
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

    try:
        _ENGINE_INSTANCE = RapidOCR(params=params)
        _ENGINE_LANG = lang
    except Exception as e:
        logger.warning(f"RapidOCR Korean 모델 초기화 실패, 기본 모델로 전환: {e}")
        _ENGINE_INSTANCE = RapidOCR()
        _ENGINE_LANG = lang

    return _ENGINE_INSTANCE


def read_boxes(image: Image.Image, lang: str = "kor+eng") -> list[tuple[str, float, tuple[int, int, int, int]]]:
    """이미지의 글자 줄마다 (글자, 신뢰도 0~1, 상자 x·y·w·h) — 그림·표 감지에서 쓴다."""
    result = get_engine(lang=lang)(np.array(image.convert("RGB")))
    if result is None or getattr(result, "txts", None) is None or getattr(result, "boxes", None) is None:
        return []
    out = []
    for text, score, box in zip(result.txts, result.scores, result.boxes):
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


# 글머리 ○ 를 RapidOCR 이 한글 자모·영문 O 로 읽은 것 — 줄 머리에서만('ㅇ협력사'·'ᄋ제안서'·'O영업대표').
# 'OSI' 처럼 O 뒤가 영문이면 낱말일 수 있어 두지 않는다(SI운영방안, 2026-10-19)
_MISREAD_BULLET_RE = re.compile(r"^(?:[ㅇᄋ]\s*(?=[가-힣A-Za-z0-9(])|O(?=[가-힣]))")


def _restore_bullets(lines: list[str], boxes, rgb: np.ndarray) -> list[str]:
    """글머리 기호를 되살려 줄 잇기가 항목을 한 문단으로 붙이지 않게 한다.

    ① 잘못 읽힌 ○(ㅇ·ᄋ·O) → '○ '. ② 기호를 아예 못 읽은 줄 — 줄 머리 바로 왼쪽에 글자 하나 크기의
    네모진 잉크 덩어리가 있고, 같은 x 에 그런 줄이 2줄 이상이면 목록으로 보고 '· ' 를 붙인다
    (한 줄짜리는 쪽 가장자리 티끌일 수 있어 책 본문을 지키려고 둔다). SI운영방안 1쪽 ○ 11줄이
    한 문단으로 붙던 문제(2026-10-19)."""
    lines = [_MISREAD_BULLET_RE.sub("○ ", l) for l in lines]
    if len(lines) != len(boxes) or not lines:
        return lines
    gray = rgb.mean(axis=2) if rgb.ndim == 3 else rgb
    H, W = gray.shape
    marks: dict[int, float] = {}
    for i, b in enumerate(boxes):
        if _LIST_LEAD_RE.match(lines[i]):
            continue
        x0 = int(min(p[0] for p in b))
        y0, y1 = int(min(p[1] for p in b)), int(max(p[1] for p in b))
        h = max(1, y1 - y0)
        a = max(0, x0 - int(1.8 * h))
        strip = gray[y0:y1, a:max(a, x0 - int(0.15 * h))] < 128
        ys, xs = np.nonzero(strip)
        if len(xs) < 4:
            continue
        bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        # 글자 하나 크기·정사각형 비슷·위아래로 줄 밖까지 이어지지 않음(쪽 가장자리 그림자 제외)
        above = gray[max(0, y0 - h):y0, a + xs.min():a + xs.max() + 1] < 128
        if (0.25 * h <= bw <= 1.0 * h and 0.25 * h <= bh <= 1.0 * h and 0.6 <= bw / bh <= 1.6
                and not above[-max(1, h // 4):].any()):
            marks[i] = a + (xs.min() + xs.max()) / 2
    for i, cx in marks.items():
        h = max(p[1] for p in boxes[i]) - min(p[1] for p in boxes[i])
        if sum(1 for j, c in marks.items() if j != i and abs(c - cx) < 0.5 * h) >= 1:
            lines[i] = "· " + lines[i]
    # ③ 기호를 못 읽었지만 OCR 상자가 기호 자리부터 시작하는 줄 — 같은 쪽에서 ○ 로 시작하는 줄이
    # 2줄 이상 같은 x 에 있으면 그 x 에서 시작하는 줄도 항목이다(SI운영방안 1쪽 ○ 11줄 중 6줄이 기호 없이 읽힘)
    x0 = [min(p[0] for p in b) for b in boxes]
    hs = [max(p[1] for p in b) - min(p[1] for p in b) for b in boxes]
    anchors = [x0[i] for i, l in enumerate(lines) if l.startswith("○ ")]
    if len(anchors) >= 2:
        ax = float(np.median(anchors))
        for i, l in enumerate(lines):
            # 0.5줄높이로 넓히면 조금 왼쪽의 □ 제목('목적'·'이익개념')까지 ○ 로 붙었다 — 0.3 유지
            if not _LIST_LEAD_RE.match(l) and abs(x0[i] - ax) < 0.3 * hs[i]:
                # 기호 자리의 'O'+대문자 두 자('OSI Biz팀')는 ○ 를 O 로 읽은 것 — 덧붙이지 않고 바꾼다.
                # 'OECD'·'OPEC' 처럼 세 자 넘는 약어는 낱말이라 그대로 두고 앞에 붙인다
                lines[i] = ("○ " + l[1:]) if re.match(r"O[A-Z]{2}(?![A-Za-z])", l) else "○ " + l
    return lines


_LIST_LEAD_RE = re.compile(r"^\s*[-–—•·*▶▷►>»※□■○●◆◇▪①-⑳]|^\s*\(?\d{1,2}[.)]")


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


def _insert_anchors(lines: list[str], boxes, height: int, anchors: list[tuple[float, str]]) -> list[str]:
    """표 자리 표지를 그 높이(쪽 높이 대비 비율 — 전처리의 확대·여백과 무관)에 오는 줄 앞에 끼운다.
    표지는 앞뒤 빈 줄로 따로 세워 줄 잇기에 붙지 않게 한다. 상자가 없으면 끝에 붙인다."""
    out = list(lines)
    tops = [min(p[1] for p in b) / height for b in boxes] if boxes is not None else []
    for frac, token in sorted(anchors, reverse=True):   # 아래 표부터 끼워 앞 표지의 위치가 밀리지 않게
        i = next((k for k, t in enumerate(tops) if t > frac), len(out))
        out[i:i] = ["", token, ""]
    return out


def run(image: Image.Image, lang: str = "kor+eng", fast: bool = False, refine: bool = True,
        anchors: list[tuple[float, str]] | None = None) -> OCRResult:
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
    result = engine(rgb_array)

    if result is None or getattr(result, "txts", None) is None or not result.txts:
        return OCRResult(
            text="\n\n".join(t for _f, t in sorted(anchors or [])),   # 표만 있는 쪽
            confidence=100.0,
            engine=OCREngine.RAPIDOCR,
            has_jamo_split=False,
        )

    txts = result.txts
    scores = getattr(result, "scores", None) or [1.0] * len(txts)

    # 개별 라인 텍스트 및 신뢰도 수집
    lines: list[str] = []
    confs: list[float] = []
    line_boxes = []
    boxes = getattr(result, "boxes", None)

    for idx, text in enumerate(txts):
        line_str = str(text).strip()
        if line_str:
            lines.append(line_str)
            if boxes is not None and idx < len(boxes):
                line_boxes.append(boxes[idx])
            score = float(scores[idx]) if idx < len(scores) else 1.0
            # 0.0~1.0 score -> 0.0~100.0 스케일
            confs.append(score * 100.0)

    if len(line_boxes) == len(lines):
        lines, line_boxes = _order_columns(lines, line_boxes)
        lines, line_boxes = _drop_page_footer(lines, line_boxes)
        lines = _restore_bullets(lines, line_boxes, rgb_array)
    marked = _mark_paragraph_starts(lines, line_boxes)
    if anchors:
        marked = _insert_anchors(marked, line_boxes if len(line_boxes) == len(lines) else None,
                                 image_rgb.height, anchors)
    full_text = "\n".join(marked)
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


def smart_run(image: Image.Image, default_lang: str = "kor+eng", fast: bool = False, refine: bool = True,
              anchors: list[tuple[float, str]] | None = None) -> OCRResult:
    """
    로컬 OCR 진입점 — 한 번 읽고, 거의 영문인 쪽만 영문 모델로 다시 읽는다.

    한국어 모델은 'kor'·'kor+eng' 가 같은 모델이라 한국어 쪽은 다시 읽지 않는다.
    RapidOCR 이 없거나 실패하면 빈 결과(신뢰도 0)를 돌려준다 — 처리를 멈추지 않고,
    호출부가 저신뢰 쪽처럼 Vision 폴백하거나 '검토 필요'로 남긴다.
    """
    try:
        extra = {"anchors": anchors} if anchors else {}
        res = run(image, lang=default_lang, fast=fast, refine=refine, **extra)
        if default_lang != "eng" and _mostly_english(res.text):
            res = run(image, lang="eng", fast=fast, refine=refine, **extra)
        return res
    except Exception as e:
        logger.warning(f"RapidOCR 판독 실패 — 신뢰도 0 으로 넘김: {e}")
        return OCRResult(text="", confidence=0.0, engine=OCREngine.RAPIDOCR, has_jamo_split=False)


def _mostly_english(text: str) -> bool:
    """영문 글자가 영문+한글의 85% 를 넘는 쪽."""
    eng = sum(1 for c in text if ord(c) < 128 and c.isalpha())
    kor = sum(1 for c in text if "가" <= c <= "힣")
    return eng + kor > 0 and eng / (eng + kor) > 0.85

