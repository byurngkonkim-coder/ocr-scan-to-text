"""Claude Vision 폴백 — 저품질 스캔 고정확도 처리"""
import base64
import io
from PIL import Image

from core.types import OCREngine, OCRResult

_client = None

# ── LLM 종료 사유 가드 (G01) ──────────────────────────────────────────────
# 근거: "예외 없음 ≠ 성공". Anthropic은 stop_reason=max_tokens·refusal 에서
#       예외 없이 잘린/빈 텍스트를 반환한다. gemini_vision.response_ok() 의
#       Anthropic 경로 대응물 — 한쪽만 방어하면 다른 경로가 남는다.
#       실측 사고: 842p 중 167p 조용한 소실.
_OK_STOP = {"end_turn", "stop_sequence"}


def response_ok(response) -> tuple[bool, str]:
    """(정상 종료 여부, stop_reason). end_turn/stop_sequence/판단불가만 수용.

    SDK 구조가 바뀌어 stop_reason 을 읽을 수 없으면 통과시킨다(안전측:
    오탐으로 정상 페이지를 전부 실패시키는 것보다 통과가 낫다).
    """
    sr = getattr(response, "stop_reason", None)
    if sr is None:
        return True, "UNKNOWN"
    return str(sr) in _OK_STOP, str(sr)


def _get_client():
    global _client
    if _client is None:
        try:
            import anthropic
            from core.config import get_api_key
            api_key = get_api_key()
            if not api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY가 설정되지 않았습니다.\n"
                    "설정 방법 1: python ocr_tool.py --setup\n"
                    "설정 방법 2: set ANTHROPIC_API_KEY=sk-ant-..."
                )
            # 시간 제한 — SDK 기본(10분 × 재시도 2회)이면 응답 없는 쪽 하나에 최대 약 30분 멈춘다
            _client = anthropic.Anthropic(api_key=api_key, timeout=120.0, max_retries=1)
        except ImportError:
            raise RuntimeError(
                "anthropic 패키지가 설치되어 있지 않습니다.\n"
                "pip install anthropic 를 실행해 주세요."
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


def run(image: Image.Image, lang_hint: str = "kor+eng") -> OCRResult:
    """
    Claude Vision API로 이미지에서 텍스트 추출.

    Args:
        image: PIL 이미지
        lang_hint: 언어 힌트 (프롬프트에 반영)

    Returns:
        OCRResult (engine=CLAUDE_VISION, confidence=95.0)
    """
    client = _get_client()

    # PIL → base64 PNG
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    img_b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")

    # 언어 힌트 추가
    prompt = VISION_PROMPT
    if "kor" in lang_hint:
        prompt += "\n- 한국어 인식에 특히 주의하세요."

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_b64,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )

    ok, sr = response_ok(response)
    if not ok:
        raise RuntimeError(
            f"Claude Vision 비정상 종료: stop_reason={sr} "
            f"(부분/빈 텍스트 수용 거부 — 재시도 또는 폴백 필요)"
        )

    text = response.content[0].text if response.content else ""
    if not text.strip():   # 예외 없음 ≠ 성공 — 빈 응답을 신뢰도 95로 받으면 쪽이 조용히 빈다
        raise RuntimeError("Claude Vision 빈 응답 — 로컬 결과로 넘김")

    # 사용량 기록
    try:
        from core.usage_tracker import record_vision
        record_vision(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
    except Exception:
        pass

    return OCRResult(
        text=text,
        confidence=95.0,
        engine=OCREngine.CLAUDE_VISION,
        has_jamo_split=False,
    )


def is_available() -> bool:
    """Claude Vision 사용 가능 여부 확인 (환경변수 또는 설정 파일)"""
    from core.config import has_api_key
    return has_api_key()
