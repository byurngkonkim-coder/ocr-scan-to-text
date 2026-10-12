"""Tesseract OCR 엔진 — 블록별 언어 감지 + 신뢰도 반환"""
import pytesseract
from PIL import Image

from core.types import OCREngine, OCRResult
from core.char_normalizer import has_jamo_decomposition

# Windows Tesseract 기본 경로
import platform
if platform.system() == "Windows":
    import os
    _default_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(_default_path):
        pytesseract.pytesseract.tesseract_cmd = _default_path

# psm 3: 자동 레이아웃 분석 (느림·정확), psm 6: 단일 텍스트 블록 (빠름)
# preserve_interword_spaces=1: 한글·한영 혼합문의 어절 띄어쓰기 보존 (CJK 권장)
_CONFIG_FULL = "--psm 3 --oem 1 -c preserve_interword_spaces=1"
_CONFIG_FAST = "--psm 6 --oem 1 -c preserve_interword_spaces=1"


def run(image: Image.Image, lang: str = "kor+eng", fast: bool = False, force_tesseract: bool = False) -> OCRResult:
    """
    이미지에 로컬 OCR을 실행하고 OCRResult 반환.
    RapidOCR(ONNX)이 사용 가능한 경우 최우선 실행하고, 미설치 또는 force_tesseract=True인 경우 Tesseract로 폴백합니다.

    Args:
        image: 전처리된 PIL 이미지
        lang: 언어 코드 (기본 kor+eng)
        fast: True면 빠른 설정 사용
        force_tesseract: True면 RapidOCR 대신 Tesseract 강제 사용

    Returns:
        OCRResult (text, confidence, engine)
    """
    if not force_tesseract:
        try:
            from core import rapidocr_engine
            if rapidocr_engine.is_available():
                res = rapidocr_engine.run(image, lang=lang, fast=fast)
                return OCRResult(
                    text=res.text,
                    confidence=res.confidence,
                    engine=OCREngine.TESSERACT,
                    has_jamo_split=res.has_jamo_split,
                )
        except Exception:
            pass

    cfg = _CONFIG_FAST if fast else _CONFIG_FULL

    # 블록별 신뢰도 데이터 수집
    data = pytesseract.image_to_data(
        image, lang=lang,
        output_type=pytesseract.Output.DICT,
        config=cfg,
    )

    confs = [int(c) for c in data["conf"] if str(c).lstrip("-").isdigit() and int(c) != -1]
    avg_conf = sum(confs) / len(confs) if confs else 0.0

    # 전체 텍스트 추출
    text = pytesseract.image_to_string(
        image, lang=lang,
        config=cfg,
    )

    # 자모 분리 감지 → 신뢰도 강제 0 (폴백 트리거)
    jamo_split = has_jamo_decomposition(text)
    if jamo_split:
        avg_conf = 0.0

    return OCRResult(
        text=text,
        confidence=avg_conf,
        engine=OCREngine.TESSERACT,
        has_jamo_split=jamo_split,
    )


def smart_run(image: Image.Image, default_lang: str = "kor+eng",
              fast: bool = False) -> OCRResult:
    """
    1차 OCR 후 블록별 언어 비율을 분석하여
    최적 언어팩으로 재처리하는 스마트 OCR.
    """
    rough_result = run(image, lang=default_lang, fast=fast)
    optimal_lang = _detect_language(rough_result.text)
    if optimal_lang != default_lang:
        return run(image, lang=optimal_lang, fast=fast)
    return rough_result


def _detect_language(text: str) -> str:
    """
    텍스트에서 영문/한글 비율을 분석하여 최적 언어팩 코드 반환.
    - 영문 85%+ → 'eng'
    - 한글 85%+ → 'kor'
    - 혼합   → 'kor+eng'
    """
    ascii_count = sum(1 for c in text if ord(c) < 128 and c.isalpha())
    kor_count   = sum(1 for c in text if "가" <= c <= "힣")
    total = ascii_count + kor_count

    if total == 0:
        return "kor+eng"

    eng_ratio = ascii_count / total
    if eng_ratio > 0.85:
        return "eng"
    if eng_ratio < 0.15:
        return "kor"
    return "kor+eng"
