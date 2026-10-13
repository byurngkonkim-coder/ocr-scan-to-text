"""로컬 OCR 진입점 — RapidOCR 한글 v5 만 쓴다 (모듈 이름은 호출부 호환을 위해 유지).

예전엔 RapidOCR 오류·미설치 때 Tesseract 로 조용히 넘어가 한글 품질이 크게 떨어진 결과를
'완료'로 냈다(2026-10-13). 이제 RapidOCR 이 없거나 실패하면 그대로 오류를 낸다 —
호출부가 Vision 폴백하거나 그 쪽을 실패로 드러낸다.
"""
from PIL import Image

from core import rapidocr_engine
from core.types import OCREngine, OCRResult


def run(image: Image.Image, lang: str = "kor+eng", fast: bool = False) -> OCRResult:
    """이미지에 RapidOCR 을 실행하고 OCRResult 반환. 미설치면 RuntimeError."""
    if not rapidocr_engine.is_available():
        raise RuntimeError("RapidOCR 이 설치돼 있지 않습니다 — engine\\install.bat 을 실행하세요")
    res = rapidocr_engine.run(image, lang=lang, fast=fast)
    return OCRResult(
        text=res.text,
        confidence=res.confidence,
        engine=OCREngine.TESSERACT,   # 처리 기록·화면의 '로컬 엔진' 구분값 (실제 엔진은 RapidOCR)
        has_jamo_split=res.has_jamo_split,
    )


def smart_run(image: Image.Image, default_lang: str = "kor+eng",
              fast: bool = False) -> OCRResult:
    """한 번 읽고, 거의 영문인 쪽만 영문 모델로 다시 읽는다.

    한국어 모델은 'kor'·'kor+eng' 가 같은 모델이라, 한국어 쪽을 언어만 바꿔 다시 읽으면
    같은 결과에 모델만 쪽마다 다시 불러온다(예전 동작).
    """
    rough_result = run(image, lang=default_lang, fast=fast)
    if _detect_language(rough_result.text) == "eng" and default_lang != "eng":
        return run(image, lang="eng", fast=fast)
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
