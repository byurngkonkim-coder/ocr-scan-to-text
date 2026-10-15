"""Vision 폴백 파사드 — config.json 의 vision_provider 에 따라 엔진 라우팅.

기존 호출부는 이 모듈 하나만 사용하면 되고(claude_vision 와 동일한 인터페이스),
엔진 전환은 코드 수정 없이 config.json 의 "vision_provider" 값만 바꾸면 된다.

    "vision_provider": "gemini"  → Google Gemini (core.gemini_vision)
    "vision_provider": "claude"  → Anthropic Claude (core.claude_vision)
    "vision_provider": "ollama"  → 로컬 Ollama (core.ollama_vision) — API 키 불필요
"""
import math
import os
import threading
from PIL import Image

from core import config
from core import gemini_vision as _gemini
from core import claude_vision as _claude
from core import ollama_vision as _ollama
from core.types import OCRResult


def _impl():
    """현재 설정된 provider 모듈 반환."""
    provider = config.get_vision_provider()
    if provider == "claude":
        return _claude
    if provider == "ollama":
        return _ollama
    # 기본값 및 알 수 없는 값 → gemini
    return _gemini


class VisionBudget:
    """한 문서의 Vision 폴백 쪽 수 상한(GUI·CLI 공통). 쪽 읽기 스레드끼리 나눠 쓴다."""

    def __init__(self, total_pages: int):
        self.left = max(3, math.ceil(total_pages * config.get_vision_max_ratio()))
        self._lock = threading.Lock()

    def take(self) -> bool:
        with self._lock:
            if self.left <= 0:
                return False
            self.left -= 1
            return True


def provider_name() -> str:
    """'gemini' | 'claude' | 'ollama'."""
    p = config.get_vision_provider()
    return p if p in ("claude", "ollama") else "gemini"


def provider_label() -> str:
    """사람이 읽는 엔진 이름."""
    return {"claude": "Anthropic Claude", "ollama": "로컬 Ollama"}.get(provider_name(), "Google Gemini")


def is_available() -> bool:
    """현재 provider 의 키가 설정되어 있으면 True."""
    return _impl().is_available()


def run(image: Image.Image, lang_hint: str = "kor+eng") -> OCRResult:
    """현재 provider 로 OCR 실행.

    머리글·바닥글 마스킹(preprocessor.mask_margins)을 여기서 적용한다.
    Vision 경로는 전처리(preprocessor.preprocess)를 통째로 건너뛰므로 마스킹이
    걸리지 않아, 마진 필터를 켜도 러닝헤더가 그대로 OCR에 들어가던 결함이 있었다
    (Vision 우선/품질폴백/저신뢰폴백 + CLI = 호출부 5곳). 호출부가 모두 이
    파사드로 모이므로 여기 한 곳에서 처리한다.
    """
    if config.get_enable_margin_filter():
        from core import preprocessor
        # convert("RGB")가 사본을 돌려주므로 호출부의 원본 이미지는 보존된다
        # (preprocessor.preprocess 와 동일 관용구).
        image = preprocessor.mask_margins(
            image.convert("RGB"),
            config.get_margin_top_ratio(),
            config.get_margin_bottom_ratio(),
        )
    return _impl().run(image, lang_hint=lang_hint)


def run_text(prompt: str) -> str:
    """현재 provider 로 텍스트→텍스트 호출 (문서 정제 단계용).

    지원하지 않는 엔진(예: Claude 경로 미구현)이면 RuntimeError.
    """
    fn = getattr(_impl(), "run_text", None)
    if fn is None:
        raise RuntimeError(f"{provider_label()} 엔진은 텍스트 정제를 지원하지 않습니다.")
    return fn(prompt)


def reset_client() -> None:
    """양쪽 provider 의 캐시된 클라이언트를 초기화 (키 변경 후 호출)."""
    _gemini._client = None
    _gemini._blocked = None     # 새 키면 결제 차단도 풀어 다시 시도
    _claude._client = None
    _ollama._client = None


def set_session_key(key: str) -> None:
    """이 세션에만 적용되는 임시 키 등록 — 현재 provider 에 맞는 환경변수로 설정."""
    key = key.strip()
    if not key or provider_name() == "ollama":  # 로컬 엔진은 키가 없다
        return
    if provider_name() == "claude":
        os.environ["ANTHROPIC_API_KEY"] = key
    else:
        os.environ["GEMINI_API_KEY"] = key
    reset_client()
