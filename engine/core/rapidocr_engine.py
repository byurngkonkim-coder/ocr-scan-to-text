"""RapidOCR (ONNX Runtime) 엔진 — 초경량·고속 로컬 OCR (기본 권장 엔진).

Tesseract와 달리 C++ 바이너리 시스템 설치가 필요 없으며,
순수 Python 및 ONNX Runtime만으로 CPU에서 고속·고정확도 텍스트 인식을 수행합니다.
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

    params = {}
    # 한국어 또는 한영 혼합인 경우 한국어 특화 v5 모바일 인식 모델 적용
    if any(k in lang.lower() for k in ("kor", "korean", "ko")):
        params = {
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Rec.lang_type": LangRec.KOREAN,
            "Rec.model_type": ModelType.MOBILE,
        }

    try:
        _ENGINE_INSTANCE = RapidOCR(params=params if params else None)
        _ENGINE_LANG = lang
    except Exception as e:
        logger.warning(f"RapidOCR Korean 모델 초기화 실패, 기본 모델로 전환: {e}")
        _ENGINE_INSTANCE = RapidOCR()
        _ENGINE_LANG = lang

    return _ENGINE_INSTANCE


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
    스마트 실행: RapidOCR을 사용하여 최적 인식 수행.
    (tesseract_engine.smart_run과 동일한 시그니처 제공)
    """
    return run(image, lang=default_lang, fast=fast, refine=refine)

