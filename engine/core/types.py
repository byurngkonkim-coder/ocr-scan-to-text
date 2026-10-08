"""공통 데이터 타입 정의"""
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class OCREngine(str, Enum):
    RAPIDOCR = "rapidocr"      # 로컬 고성능 ONNX 엔진 (기본 로컬 엔진)
    TESSERACT = "tesseract"
    CLAUDE_VISION = "claude_vision"
    GEMINI = "gemini"
    OLLAMA = "ollama"          # 로컬 Ollama 비전 모델 (core.ollama_vision)
    PDF_TEXT = "pdf_text"      # 디지털 PDF 텍스트 레이어 직접 추출(OCR·API 미사용)


class OutputFormat(str, Enum):
    TXT = "txt"
    MD = "md"


@dataclass
class OCRResult:
    """단일 페이지 OCR 결과"""
    text: str
    confidence: float          # 0.0 ~ 100.0
    engine: OCREngine
    page_num: int = 1
    has_jamo_split: bool = False


@dataclass
class QualityReport:
    """스캔 품질 진단 결과"""
    blur_score: float          # Laplacian variance (< 100 = 흐림)
    avg_brightness: float      # 히스토그램 평균 (> 220 = 과노출, < 50 = 저노출)
    skew_angle: float          # 기울기 각도 (> 45 = 심각)
    is_vertical_text: bool     # 세로쓰기 감지
    width_px: int
    height_px: int

    @property
    def needs_fallback(self) -> bool:
        return (
            self.blur_score < 100
            or abs(self.skew_angle) > 45
            or self.is_vertical_text
            or min(self.width_px, self.height_px) < 500
        )

    @property
    def needs_upscale(self) -> bool:
        return min(self.width_px, self.height_px) < 500


@dataclass
class PageResult:
    """파일 1페이지 처리 결과 (OCR + 보정 포함)"""
    page_num: int
    raw_text: str
    corrected_text: str
    engine: OCREngine
    confidence: float
    figures: list[str] = field(default_factory=list)  # 플레이스홀더 문자열 목록
    needs_review: bool = False    # 저신뢰도·자모분리 등 수동 검토 필요
    review_reason: str = ""       # 검토 필요 사유
    is_blank: bool = False        # 빈 페이지 감지 → OCR 건너뜀
    blank_reason: str = ""        # 빈 페이지 감지 이유


@dataclass
class FileResult:
    """파일 전체 처리 결과"""
    source_path: Path
    output_path: Optional[Path]
    pages: list[PageResult] = field(default_factory=list)
    success: bool = True
    error_message: str = ""

    @property
    def total_pages(self) -> int:
        return len(self.pages)
