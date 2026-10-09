"""이미지 전처리 — 6단계 전체 / 3단계 빠른 모드"""
import cv2
import numpy as np
from PIL import Image, ImageDraw

from core import config
from core.types import QualityReport


def mask_margins(image: Image.Image, top_ratio: float, bottom_ratio: float) -> Image.Image:
    """이미지의 상/하단 여백 영역을 흰색(White)으로 마스킹합니다."""
    if top_ratio <= 0.0 and bottom_ratio <= 0.0:
        return image

    w, h = image.size
    draw = ImageDraw.Draw(image)

    if top_ratio > 0.0:
        top_h = int(h * top_ratio)
        draw.rectangle([0, 0, w - 1, top_h - 1], fill=(255, 255, 255))

    if bottom_ratio > 0.0:
        bottom_h = int(h * (1.0 - bottom_ratio))
        draw.rectangle([0, bottom_h, w - 1, h - 1], fill=(255, 255, 255))

    return image


def preprocess_for_rapidocr(image: Image.Image, report: QualityReport | None = None) -> Image.Image:
    """
    RapidOCR (딥러닝 PP-OCRv5) 전용 경량 전처리 파이프라인.

    딥러닝 모델은 이진화(Binarization)나 그레이스케일 변환 시 텍스트 특징 맵이 훼손되므로,
    RGB 원본 컬러를 보존한 채 여백 마스킹 및 필요시 최소 해상도 업스케일만 수행합니다.
    """
    img = image.convert("RGB")

    # 여백 마진 제거 필터 적용
    if config.get_enable_margin_filter():
        top_ratio = config.get_margin_top_ratio()
        bottom_ratio = config.get_margin_bottom_ratio()
        img = mask_margins(img, top_ratio, bottom_ratio)

    # 해상도 업스케일 (단변 < 500px만 2x 보정)
    img = _upscale_if_needed(img, report)
    return img


def preprocess(image: Image.Image, report: QualityReport | None = None) -> Image.Image:
    """
    6단계 전처리 파이프라인.
    report가 있으면 진단 결과에 따라 각 단계를 선택적으로 적용.

    순서:
      1. 해상도 업스케일 (단변 < 500px → 2× Lanczos)
      2. 그레이스케일 변환
      3. 노출 보정 (과노출: Gamma / 저노출: CLAHE)
      4. 이진화 (깨끗한 스캔: Otsu 전역 / 저품질: Adaptive Gaussian)
      5. 노이즈 제거 (저품질 스캔만 Median 3×3 — 깨끗한 스캔은 생략)
      6. 기울기 보정 (Hough Transform Deskew, ±45°)
      7. 테두리 패딩 (20px 흰색)
    """
    img = image.convert("RGB")

    # 여백 마진 제거 필터 적용
    if config.get_enable_margin_filter():
        top_ratio = config.get_margin_top_ratio()
        bottom_ratio = config.get_margin_bottom_ratio()
        img = mask_margins(img, top_ratio, bottom_ratio)

    # STEP 1 — 해상도 업스케일
    img = _upscale_if_needed(img, report)

    # STEP 2 — 그레이스케일
    gray = _to_gray(img)

    # STEP 3 — 노출 보정
    if report:
        if report.avg_brightness > 220:
            gray = _gamma_correction(gray, gamma=1.5)
        elif report.avg_brightness < 50:
            gray = _clahe(gray)

    # STEP 4·5 — 이진화 + 노이즈 제거 (품질 진단 기반 조건부 적용)
    if _is_clean_scan(report):
        # 선명·균일 조명 스캔: Otsu 전역 이진화로 얇은 한글 획 보존, 잡음제거 생략
        _, denoised = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        # 저품질·불균일 조명: 적응형 이진화 + Median 잡음제거 (저품질 스캔 구제)
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31, C=10,
        )
        denoised = cv2.medianBlur(binary, 3)

    # STEP 6 — 기울기 보정 (±45° 이내)
    if report is None or abs(report.skew_angle) <= 45:
        deskewed = _deskew(denoised)
    else:
        deskewed = denoised  # 45° 초과는 폴백에서 처리

    # STEP 7 — 테두리 패딩
    padded = cv2.copyMakeBorder(
        deskewed, 20, 20, 20, 20,
        cv2.BORDER_CONSTANT, value=255,
    )

    return Image.fromarray(padded)


def quick_preprocess(image: Image.Image) -> Image.Image:
    """
    빠른 3단계 전처리 (미리보기 전용).
    기울기 보정(가장 느린 단계) 생략 → 전체 대비 약 2배 빠름.
    ① 업스케일 → ② 그레이스케일 → ③ 적응형 이진화
    """
    img = image.convert("RGB")

    # 여백 마진 제거 필터 적용
    if config.get_enable_margin_filter():
        top_ratio = config.get_margin_top_ratio()
        bottom_ratio = config.get_margin_bottom_ratio()
        img = mask_margins(img, top_ratio, bottom_ratio)

    w, h = img.size
    if min(w, h) < 500:
        img = img.resize((w * 2, h * 2), Image.LANCZOS)

    gray = _to_gray(img)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31, C=10,
    )
    padded = cv2.copyMakeBorder(binary, 10, 10, 10, 10,
                                cv2.BORDER_CONSTANT, value=255)
    return Image.fromarray(padded)


# ── 내부 함수 ──────────────────────────────────────────────────────────────────

def _is_clean_scan(report: QualityReport | None) -> bool:
    """선명하고 균일하게 노출된 '깨끗한 스캔'인지 판정.

    깨끗한 스캔에는 적응형 이진화+Median 잡음제거가 오히려 얇은 한글 획을
    갉아내 인식률을 떨어뜨릴 수 있다. 이 경우 Otsu 전역 이진화만 적용한다.
    진단 정보가 없으면(보수적으로) False → 기존 적응형 경로 사용.

    판정 핵심은 '선명도(blur_score)'다. 문서 스캔은 흰 배경이 많아 평균 밝기가
    높은 게 정상이므로 상한을 두지 않고, 과도하게 어두운 경우(<60)만 적응형으로
    보낸다. 흐릿·저대비 스캔은 blur_score가 낮아 자동으로 적응형 경로로 간다.
    """
    if report is None:
        return False
    return report.blur_score >= 150 and report.avg_brightness >= 60


def _to_gray(image: Image.Image) -> np.ndarray:
    arr = np.array(image)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def _upscale_if_needed(image: Image.Image, report: QualityReport | None) -> Image.Image:
    if report and report.needs_upscale:
        w, h = image.size
        return image.resize((w * 2, h * 2), Image.LANCZOS)
    return image


def _gamma_correction(gray: np.ndarray, gamma: float = 1.5) -> np.ndarray:
    inv_gamma = 1.0 / gamma
    table = np.array([
        (i / 255.0) ** inv_gamma * 255
        for i in range(256)
    ], dtype=np.uint8)
    return cv2.LUT(gray, table)


def _clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _deskew(binary: np.ndarray) -> np.ndarray:
    """Hough Transform으로 기울기 감지 후 Affine 회전 보정"""
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    if lines is None:
        return binary

    angles = []
    for line in lines[:20]:
        _, theta = line[0]
        angle = np.degrees(theta) - 90
        if abs(angle) < 45:
            angles.append(angle)

    if not angles:
        return binary

    skew = float(np.median(angles))
    if abs(skew) < 0.5:  # 0.5° 미만은 무시
        return binary

    h, w = binary.shape
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, skew, 1.0)
    rotated = cv2.warpAffine(
        binary, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    return rotated
