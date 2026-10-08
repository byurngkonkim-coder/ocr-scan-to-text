"""스캔 품질 자동 진단 — 흐림·노출·해상도·기울기·세로쓰기"""
import cv2
import numpy as np
from PIL import Image

from core.types import QualityReport


def diagnose(image: Image.Image) -> QualityReport:
    """
    PIL 이미지를 분석하여 QualityReport 반환.

    진단 항목:
      - 흐림: Laplacian Variance < 100
      - 과노출: 히스토그램 평균 > 220
      - 저노출: 히스토그램 평균 < 50
      - 기울기: Hough Transform 각도 > 45°
      - 저해상도: 단변 < 500px
      - 세로쓰기: 바운딩박스 종횡비 avg_h/avg_w > 2.0
    """
    img_cv = _to_gray_cv(image)
    h, w = img_cv.shape

    blur_score     = _calc_blur(img_cv)
    avg_brightness = _calc_brightness(img_cv)
    skew_angle     = _calc_skew(img_cv)
    is_vertical    = _detect_vertical_text(img_cv)

    return QualityReport(
        blur_score=blur_score,
        avg_brightness=avg_brightness,
        skew_angle=skew_angle,
        is_vertical_text=is_vertical,
        width_px=w,
        height_px=h,
    )


# ── 내부 함수 ──────────────────────────────────────────────────────────────────

def _to_gray_cv(image: Image.Image) -> np.ndarray:
    arr = np.array(image.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def _calc_blur(gray: np.ndarray) -> float:
    """Laplacian Variance — 낮을수록 흐림"""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _calc_brightness(gray: np.ndarray) -> float:
    """히스토그램 평균 밝기"""
    return float(gray.mean())


def _calc_skew(gray: np.ndarray) -> float:
    """
    Hough Transform으로 기울기 추정.
    검출 실패 시 0.0 반환.
    """
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    if lines is None:
        return 0.0

    angles = []
    for line in lines[:20]:  # 상위 20개 선분만 사용
        _, theta = line[0]
        angle = np.degrees(theta) - 90
        angles.append(angle)

    if not angles:
        return 0.0

    # 중앙값으로 이상치 제거
    return float(np.median(angles))


def _detect_vertical_text(gray: np.ndarray) -> bool:
    """
    바운딩박스 종횡비로 세로쓰기 감지.
    개별 문자 바운딩박스의 avg_h/avg_w > 2.0 이면 세로쓰기로 판단.
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ratios = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        # 너무 작거나 큰 영역 제외 (노이즈 · 페이지 전체)
        if cw < 5 or ch < 5 or cw > gray.shape[1] * 0.8:
            continue
        ratios.append(ch / cw)

    if not ratios:
        return False

    avg_ratio = float(np.median(ratios))
    return avg_ratio > 2.0


def describe(report: QualityReport) -> str:
    """진단 결과를 사람이 읽기 쉬운 문자열로 반환"""
    lines = [
        f"  해상도    : {report.width_px} × {report.height_px} px",
        f"  흐림 점수 : {report.blur_score:.1f}  {'⚠ 흐림' if report.blur_score < 100 else 'OK'}",
        f"  밝기 평균 : {report.avg_brightness:.1f}  "
        + ("⚠ 과노출" if report.avg_brightness > 220
           else "⚠ 저노출" if report.avg_brightness < 50
           else "OK"),
        f"  기울기    : {report.skew_angle:.1f}°  {'⚠ 심함' if abs(report.skew_angle) > 45 else 'OK'}",
        f"  세로쓰기  : {'⚠ 감지' if report.is_vertical_text else 'OK'}",
        f"  → {'폴백 권장' if report.needs_fallback else '전처리 후 Tesseract 처리'}",
    ]
    return "\n".join(lines)
