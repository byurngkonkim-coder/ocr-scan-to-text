"""Google Gemini Vision 폴백 — 저품질 스캔 고정확도 처리

claude_vision.py 와 동일한 인터페이스(run / is_available / _client)를 제공한다.
엔진 선택은 core.vision 파사드가 config.json 의 vision_provider 로 결정한다.

성능 최적화 (2026-06-29):
  - 호출 타임아웃 지정 → 네트워크 멈춤(hang) 시 무한 대기 방지 (최대 지연 상한 보장)
  - 전송 이미지: PNG(무손실·대용량) → JPEG + 장변 다운스케일 → 업로드 페이로드 대폭 감소
  - thinking(추론) 비활성화 → OCR엔 불필요한 지연·토큰 제거
  - 1회 자동 재시도 (재시도 시 thinking 옵션 제거 → 모델 호환성 폴백 겸용)
"""
import io
import re
from PIL import Image

from core.types import OCREngine, OCRResult
from core import config

_client = None


def _get_client():
    global _client
    if _client is None:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise RuntimeError(
                "google-genai 패키지가 설치되어 있지 않습니다.\n"
                "pip install google-genai 를 실행해 주세요."
            )
        from core.config import get_google_api_key
        api_key = get_google_api_key()
        if not api_key:
            raise RuntimeError(
                "Google Gemini API 키가 설정되지 않았습니다.\n"
                "설정 방법 1: config.json 의 \"google_api_key\" 에 키 입력\n"
                "설정 방법 2: 환경변수 set GEMINI_API_KEY=AIza...\n"
                "키 발급: https://aistudio.google.com/apikey"
            )
        # 타임아웃 지정(ms) — 네트워크 멈춤 시 무한 대기 방지.
        timeout_ms = config.get_gemini_timeout_s() * 1000
        _client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=timeout_ms),
        )
    return _client


VISION_PROMPT = """이 이미지는 스캔된 문서입니다. 이미지에서 텍스트를 정확하게 추출해 주세요.

규칙:
- 이미지에 보이는 텍스트를 그대로 추출하세요. 내용을 요약하거나 변경하지 마세요.
- 한국어와 영어가 혼합된 경우 그대로 유지하세요.
- 줄바꿈은 원본 문서의 구조를 따르세요.
- 표가 있으면 Markdown 표 형식으로 변환하세요.
- 그림·사진이 있으면 [그림: 설명] 형태로 표시하세요.
- 텍스트만 출력하고 다른 설명은 쓰지 마세요."""


def _prepare_jpeg_bytes(image: Image.Image, max_edge: int | None = None) -> bytes:
    """
    전송용 이미지 바이트 생성.
      1. 알파/팔레트 → 흰 배경 위에 합성하여 RGB 로 평탄화
      2. 장변이 max_edge 초과 시 Lanczos 다운스케일 (Gemini는 내부적으로
         축소·타일링하므로 과도한 해상도는 정확도 이득 없이 업로드만 느려짐)
      3. JPEG 인코딩 (PNG 대비 스캔 노이즈 이미지에서 페이로드 수배~수십배 감소)
    """
    # 1) RGB 평탄화 (투명 배경이 검게 변하지 않도록 흰색으로 합성)
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[-1])
        img = bg
    else:
        img = image.convert("RGB")

    # 2) 다운스케일
    if max_edge is None:  # 호출부(ollama_vision)가 자체 상한을 넘길 수 있다
        max_edge = config.get_gemini_max_edge()
    if max_edge and max_edge > 0:
        w, h = img.size
        longest = max(w, h)
        if longest > max_edge:
            scale = max_edge / longest
            img = img.resize(
                (max(1, round(w * scale)), max(1, round(h * scale))),
                Image.LANCZOS,
            )

    # 3) JPEG 인코딩
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=config.get_gemini_jpeg_quality())
    return buf.getvalue()


def _record_usage(response, kind: str = "vision") -> None:
    """사용량 기록 (best-effort). kind='vision'(OCR) | 'llm'(텍스트 정제)."""
    try:
        from core.usage_tracker import record_vision, record_llm
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return
        inp = getattr(usage, "prompt_token_count", 0) or 0
        out = getattr(usage, "candidates_token_count", 0) or 0
        model = config.get_gemini_model()
        if kind == "llm":
            record_llm(input_tokens=inp, output_tokens=out, model=model)
        else:
            record_vision(input_tokens=inp, output_tokens=out, model=model)
    except Exception:
        pass


def _finish_reason(response) -> str:
    """응답 finish_reason 을 짧은 문자열로. 후보가 없으면 'NO_CANDIDATES'.

    실측 사고(2026-07-07, 메시지성경 신약 842p): 저작권 있는 출판물 스캔에서
    Gemini가 finish_reason=RECITATION(인용 차단)으로 **예외 없이 빈 텍스트**를
    반환 — 이를 신뢰도 95%로 수용해 167p가 통째로 소실됐다. temperature·
    thinking 조정으로도 우회되지 않음(실측). 비정상 종료(STOP 이외)는
    텍스트가 있어도 부분 절단일 수 있으므로 수용하면 안 된다.
    """
    try:
        cands = getattr(response, "candidates", None) or []
        if not cands:
            return "NO_CANDIDATES"
        fr = getattr(cands[0], "finish_reason", None)
        if fr is None:
            return "UNKNOWN"
        return str(fr).rsplit(".", 1)[-1]
    except Exception:  # noqa: BLE001 — SDK 구조 변화 대비, 판단 불가면 통과
        return "UNKNOWN"


def response_ok(response) -> tuple[bool, str]:
    """(정상 종료 여부, finish_reason). STOP/판단불가(UNKNOWN)만 수용."""
    fr = _finish_reason(response)
    return fr in ("STOP", "UNKNOWN"), fr


_MD_RE = re.compile(r'\*\*[^*\n]{2,60}\*\*|^#{1,4}\s', re.M)
_BULLET_RE = re.compile(r'^\s*[•\-*]\s', re.M)


def looks_fabricated(text: str) -> bool:
    """이미지를 읽지 않고 글을 지어낸 응답 의심 여부.

    실측(2026-10-05, 드러커 매니지먼트 471쪽): '디지털 전환·프로젝트 관리' 보고서를 지어내
    실제 쪽 내용이 사라졌다. 그 출력에만 마크다운 강조와 글머리표 목록이 함께 있었다
    (결과 10건 4,197쪽 중 이 둘을 함께 가진 쪽은 그 1쪽뿐). 한쪽 신호만으로는 정상 쪽도 걸린다.
    """
    return bool(_MD_RE.search(text)) and len(_BULLET_RE.findall(text)) >= 3


def run(image: Image.Image, lang_hint: str = "kor+eng") -> OCRResult:
    """
    Google Gemini API로 이미지에서 텍스트 추출.

    Args:
        image: PIL 이미지
        lang_hint: 언어 힌트 (프롬프트에 반영)

    Returns:
        OCRResult (engine=GEMINI, confidence=95.0)

    Raises:
        비정상 종료(RECITATION·SAFETY·MAX_TOKENS 등)가 재시도에도 반복되면
        예외를 던진다 — 호출부(_process_page)가 RapidOCR 폴백으로 전환해
        페이지 소실을 막는다.
    """
    from google.genai import types

    client = _get_client()
    jpeg_bytes = _prepare_jpeg_bytes(image)
    part = types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")

    prompt = VISION_PROMPT
    if "kor" in lang_hint:
        prompt += "\n- 한국어 인식에 특히 주의하세요."

    model = config.get_gemini_model()
    disable_thinking = config.get_gemini_disable_thinking()

    # 최대 2회 시도. 재시도 시 thinking 옵션을 제거하여
    # ① 일시적 네트워크 오류 회복 ② thinking 미지원 모델 호환 폴백을 동시에 처리.
    last_err: Exception | None = None
    for attempt in range(2):
        try:
            cfg_kwargs: dict = {"temperature": 0.0}
            if disable_thinking:
                cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            response = client.models.generate_content(
                model=model,
                contents=[part, prompt],
                config=types.GenerateContentConfig(**cfg_kwargs),
            )
            _record_usage(response)
            ok, fr = response_ok(response)
            if not ok:
                last_err = RuntimeError(
                    f"Gemini 비정상 종료 finish_reason={fr} "
                    f"(텍스트 {len(response.text or '')}자) — 페이지 소실 방지 차단")
                disable_thinking = False
                continue
            # 정상 종료(STOP/UNKNOWN)인데 본문이 비면 쪽이 경고 없이 사라진다 → 폴백
            # (빈 페이지는 blank_detector 가 OCR 전에 걸러낸다)
            if not (response.text or "").strip():
                last_err = RuntimeError(
                    f"Gemini 빈 응답 finish_reason={fr} — 페이지 소실 방지 차단")
                disable_thinking = False
                continue
            if looks_fabricated(response.text):
                last_err = RuntimeError(
                    "Gemini 출력이 책과 무관한 생성 글로 의심됨(마크다운 강조·글머리표) — 원문 소실 방지 차단")
                disable_thinking = False
                continue
            return OCRResult(
                text=response.text or "",
                confidence=95.0,
                engine=OCREngine.GEMINI,
                has_jamo_split=False,
            )
        except Exception as e:  # noqa: BLE001 — 호출부에서 RapidOCR 폴백 처리
            last_err = e
            disable_thinking = False  # 2차 시도는 thinking 옵션 없이

    raise last_err  # type: ignore[misc]


def run_text(prompt: str) -> str:
    """
    텍스트 → 텍스트 Gemini 호출 (이미지 없음). OCR 후처리 '정제(교정)' 단계용.

    run()과 동일한 정책 적용: temperature=0, thinking off(설정 시), 타임아웃,
    1회 자동 재시도(재시도 시 thinking 제거). 사용량은 'llm'(정제)로 기록.
    """
    from google.genai import types

    client = _get_client()
    model = config.get_gemini_model()
    disable_thinking = config.get_gemini_disable_thinking()

    last_err: Exception | None = None
    for attempt in range(2):
        try:
            cfg_kwargs: dict = {"temperature": 0.0}
            if disable_thinking:
                cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(**cfg_kwargs),
            )
            _record_usage(response, kind="llm")
            ok, fr = response_ok(response)
            if not ok:
                # RECITATION 등 비정상 종료 — 빈/부분 출력 수용 금지.
                # 호출부(text_cleaner)가 예외를 받아 원문 보존으로 강등한다.
                last_err = RuntimeError(
                    f"Gemini 비정상 종료 finish_reason={fr} "
                    f"(텍스트 {len(response.text or '')}자)")
                disable_thinking = False
                continue
            return response.text or ""
        except Exception as e:  # noqa: BLE001
            last_err = e
            disable_thinking = False  # 2차 시도는 thinking 옵션 없이

    raise last_err  # type: ignore[misc]


def is_available() -> bool:
    """Gemini 사용 가능 여부 (환경변수 또는 설정 파일에 키가 있으면 True)."""
    from core.config import has_google_api_key
    return has_google_api_key()
