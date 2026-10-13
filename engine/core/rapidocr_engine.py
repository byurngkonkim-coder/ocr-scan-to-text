"""RapidOCR (ONNX Runtime) 엔진 — 이 도구의 유일한 로컬 OCR (한글 v5 모델).

별도 프로그램 설치 없이 순수 Python 및 ONNX Runtime만으로 CPU에서 고속·고정확도 텍스트 인식을 수행합니다.
본문 판독(smart_run)·그림 감지·빈 쪽 판정(read_boxes)이 모두 이 엔진을 쓴다.
"""
from __future__ import annotations

import logging
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
    result = engine(rgb_array)

    if result is None or getattr(result, "txts", None) is None or not result.txts:
        return OCRResult(
            text="",
            confidence=100.0,
            engine=OCREngine.RAPIDOCR,
            has_jamo_split=False,
        )

    txts = result.txts
    scores = getattr(result, "scores", None) or [1.0] * len(txts)

    # 개별 라인 텍스트 및 신뢰도 수집
    lines: list[str] = []
    confs: list[float] = []

    for idx, text in enumerate(txts):
        line_str = str(text).strip()
        if line_str:
            lines.append(line_str)
            score = float(scores[idx]) if idx < len(scores) else 1.0
            # 0.0~1.0 score -> 0.0~100.0 스케일
            confs.append(score * 100.0)

    full_text = "\n".join(lines)
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
    try:
        res = run(image, lang=default_lang, fast=fast, refine=refine)
        if default_lang != "eng" and _mostly_english(res.text):
            res = run(image, lang="eng", fast=fast, refine=refine)
        return res
    except Exception as e:
        logger.warning(f"RapidOCR 판독 실패 — 신뢰도 0 으로 넘김: {e}")
        return OCRResult(text="", confidence=0.0, engine=OCREngine.RAPIDOCR, has_jamo_split=False)


def _mostly_english(text: str) -> bool:
    """영문 글자가 영문+한글의 85% 를 넘는 쪽."""
    eng = sum(1 for c in text if ord(c) < 128 and c.isalpha())
    kor = sum(1 for c in text if "가" <= c <= "힣")
    return eng + kor > 0 and eng / (eng + kor) > 0.85

