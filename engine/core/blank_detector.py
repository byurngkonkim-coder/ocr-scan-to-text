"""
빈 페이지 감지 — OCR 전 차단으로 API 크레딧 절약

감지 전략 (단계별, 빠른 순서):
  1. 픽셀 밀도   : 화소의 97% 이상이 흰색 → 완전 공백
  2. 엣지 밀도   : 텍스트/선분 윤곽이 거의 없음 → 공백
  3. Tesseract   : 빠른 OCR 후 의미 있는 문자 < 10자 → 사실상 공백

각 단계는 이전 단계보다 느리지만 더 정확함.
단계 1·2는 CPU 연산만으로 0.1초 미만 완료.
"""
from __future__ import annotations
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


# ── 임계값 상수 (실측 기반 보정) ──────────────────────────────────────────────
# 실측치:
#   완전 빈 페이지    : white=100%,  edge=0.00000
#   스캔 노이즈만     : white=99.97%, edge=0.00000
#   페이지 번호만     : white=99.98%, edge=0.00010
#   텍스트 4줄        : white=99.29%, edge=0.00434  ← 이 경계가 핵심
#   책 본문 (빽빽)    : white=92.98%, edge=0.04348
WHITE_RATIO_BLANK = 0.9995  # 99.95% 이상 흰색 → 완전 공백 (stage 1)
WHITE_RATIO_SUSPECT = 0.999   # 99.9~99.95% → 의심 → edge 검사 (stage 2)
EDGE_DENSITY_BLANK = 0.0005  # 엣지 밀도 0.05% 미만 → 텍스트 없음
MIN_MEANINGFUL_CHARS = 10      # Tesseract 결과 의미 문자 수 최소값


@dataclass
class BlankResult:
    is_blank: bool
    reason: str          # 감지 이유 (로그용)
    white_ratio: float   # 흰색 픽셀 비율 0.0~1.0
    stage: int           # 감지된 단계 (1·2·3 / 0=공백 아님)


def check(image: Image.Image, fast: bool = False) -> BlankResult:
    """
    이미지가 빈 페이지인지 확인.

    Args:
        image : PIL 이미지
        fast  : True면 단계 1·2만 확인 (Tesseract 미실행, 더 빠름)

    Returns:
        BlankResult
    """
    gray = _to_gray(image)

    # ── 단계 1: 픽셀 밀도 ────────────────────────────────────────────────────
    white_ratio = _white_ratio(gray)

    if white_ratio >= WHITE_RATIO_BLANK:
        return BlankResult(
            is_blank=True,
            reason=f"공백 픽셀 {white_ratio:.1%} (완전 빈 페이지)",
            white_ratio=white_ratio,
            stage=1,
        )

    if white_ratio < WHITE_RATIO_SUSPECT:
        # 93% 미만이면 확실히 내용 있음 → 즉시 통과
        return BlankResult(
            is_blank=False,
            reason="",
            white_ratio=white_ratio,
            stage=0,
        )

    # ── 단계 2: 엣지 밀도 ────────────────────────────────────────────────────
    edge_density = _edge_density(gray)

    if edge_density < EDGE_DENSITY_BLANK:
        return BlankResult(
            is_blank=True,
            reason=f"텍스트 윤곽 없음 (엣지 밀도 {edge_density:.4f})",
            white_ratio=white_ratio,
            stage=2,
        )

    # fast 모드는 여기서 종료
    if fast:
        return BlankResult(is_blank=False, reason="", white_ratio=white_ratio, stage=0)

    # ── 단계 3: Tesseract 빠른 스캔 ──────────────────────────────────────────
    char_count = _quick_tesseract_count(image)

    if char_count < MIN_MEANINGFUL_CHARS:
        return BlankResult(
            is_blank=True,
            reason=f"의미 있는 문자 {char_count}자 미만 (사실상 빈 페이지)",
            white_ratio=white_ratio,
            stage=3,
        )

    return BlankResult(is_blank=False, reason="", white_ratio=white_ratio, stage=0)


# ── 내부 함수 ──────────────────────────────────────────────────────────────────

def _to_gray(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)


def _white_ratio(gray: np.ndarray) -> float:
    """흰색(240 이상) 픽셀 비율."""
    return float(np.sum(gray >= 240) / gray.size)


def _edge_density(gray: np.ndarray) -> float:
    """Canny 엣지 픽셀 비율 — 낮을수록 텍스트·선이 없음."""
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    return float(np.sum(edges > 0) / edges.size)


def _quick_tesseract_count(image: Image.Image) -> int:
    """
    Tesseract 최소 설정으로 의미 있는 문자 수 반환.
    psm 11: 텍스트 행 없이 개별 문자 탐지 → 빠름.

    '문자 수'만 필요하므로 장변 1500px로 다운스케일 후 실행한다.
    (300 DPI 풀해상도 psm 11 은 페이지당 수 초 소요 — 빈페이지 판정엔 과함)
    """
    try:
        import pytesseract
        small = image.convert("RGB")
        longest = max(small.size)
        if longest > 1500:
            scale = 1500 / longest
            small = small.resize(
                (max(1, round(small.width * scale)), max(1, round(small.height * scale)))
            )
        text = pytesseract.image_to_string(
            small, lang="kor+eng",
            config="--psm 11 --oem 1",
        )
        meaningful = sum(
            1 for c in text
            if c.isalpha() or "가" <= c <= "힣"
        )
        return meaningful
    except Exception:
        return MIN_MEANINGFUL_CHARS  # 오류 시 공백 아닌 것으로 처리 (안전 방향)
