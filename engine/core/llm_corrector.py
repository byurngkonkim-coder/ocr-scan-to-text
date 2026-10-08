"""L3 LLM 보정 — Claude Haiku로 OCR 잔존 오류 문맥 교정"""
from __future__ import annotations
import re

from core.config import get_api_key

# 페이지당 최대 토큰 청크 크기 (A4 1p ≈ 500토큰, Haiku 기준 약 $0.0004/p)
_CHUNK_CHARS = 1_500
_MODEL       = "claude-haiku-4-5-20251001"
_MIN_RATIO, _MAX_RATIO = 0.75, 1.25  # 출력/입력 비공백 분량 허용 범위

CORRECTION_PROMPT = """\
아래는 OCR로 추출된 한국어·영어 혼합 텍스트입니다.
다음 규칙을 엄격히 따라 교정하세요:

1. 글자 깨짐·오타·자모 분리·조사·어미 경계 오류만 수정하세요.
2. 내용·의미·단어 선택을 임의로 변경하지 마세요.
3. 확실하지 않으면 수정하지 마세요.
4. [그림 N:...] [표 N:...] [수식 N:...] 플레이스홀더는 그대로 유지하세요.
5. 수정된 텍스트만 출력하고 설명·코멘트는 쓰지 마세요.

OCR 텍스트:
{text}"""

_client = None
_call_count = 0  # 세션 내 호출 횟수 (비용 제어용)


def correct(text: str, budget: int = 0) -> str:
    """
    L3 LLM 보정 실행.

    Args:
        text   : 보정할 텍스트
        budget : 최대 호출 횟수 (0=무제한). 초과 시 원본 반환.

    Returns:
        교정된 텍스트. API 오류·예산 초과 시 원본 반환.
    """
    global _call_count

    if not text.strip():
        return text

    api_key = get_api_key()
    if not api_key:
        return text  # API 키 없으면 그냥 통과

    chunks = _split_chunks(text, _CHUNK_CHARS)
    corrected_parts: list[str] = []

    for chunk in chunks:
        if budget and _call_count >= budget:
            corrected_parts.append(chunk)
            continue
        try:
            result = _call_api(chunk, api_key)
        except Exception:
            result = chunk  # API 오류 시 원본 유지 (배치 안정성)
        # 분량 가드 — 정상 종료(end_turn)여도 문장을 빼먹거나 설명을 덧붙이면 원본 유지
        if not _length_ok(result, chunk):
            result = chunk
        # 모델은 끝 줄바꿈을 지운다 — 그대로 이으면 다음 청크 첫 줄과 붙는다
        result = result.rstrip() + chunk[len(chunk.rstrip()):]
        corrected_parts.append(result)
        _call_count += 1

    return "".join(corrected_parts)


def _length_ok(result: str, chunk: str) -> bool:
    """교정은 글자만 고치므로 비공백 분량이 원문의 0.75~1.25배 안이어야 한다.
    (하한 0.75는 text_cleaner G-01과 같은 값 — 자모 분리 복원으로 줄어드는 몫을 허용)"""
    n_in = len(re.sub(r"\s", "", chunk))
    if n_in == 0:
        return True
    ratio = len(re.sub(r"\s", "", result)) / n_in
    return _MIN_RATIO <= ratio <= _MAX_RATIO


def reset_call_count() -> None:
    """호출 횟수 초기화 (테스트·재시작 용)."""
    global _call_count
    _call_count = 0


def get_call_count() -> int:
    return _call_count


def is_available() -> bool:
    return bool(get_api_key())


# ── 내부 함수 ──────────────────────────────────────────────────────────────────

def _get_client(api_key: str):
    global _client
    if _client is None:
        try:
            import anthropic
            _client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise RuntimeError("pip install anthropic 를 실행해 주세요.")
    return _client


# ── LLM 종료 사유 가드 (G01) ──────────────────────────────────────────
# 근거: "예외 없음 ≠ 성공". max_tokens 절단·refusal은 예외 없이 잘린 텍스트를
#       정상처럼 반환한다. 실측 사고: 842p 중 167p 조용한 소실.
#       gemini_vision.response_ok() 의 Anthropic 경로 대응물.
_OK_STOP = {"end_turn", "stop_sequence"}


def _check_stop(response) -> None:
    """stop_reason 이 정상 종료가 아니면 예외. 호출부가 원본 청크를 유지한다."""
    sr = getattr(response, "stop_reason", None)
    if sr is not None and sr not in _OK_STOP:
        raise RuntimeError(f"LLM 비정상 종료: stop_reason={sr}")


def _call_api(chunk: str, api_key: str) -> str:
    """단일 청크를 Claude Haiku로 보정. 실패 시 원본 반환."""
    client = _get_client(api_key)
    response = client.messages.create(
        model=_MODEL,
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": CORRECTION_PROMPT.format(text=chunk),
        }],
    )
    _check_stop(response)
    # 사용량 기록
    try:
        from core.usage_tracker import record_llm
        record_llm(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=_MODEL,
        )
    except Exception:
        pass
    return response.content[0].text if response.content else chunk


def _split_chunks(text: str, max_chars: int) -> list[str]:
    """
    문장 경계 유지 청크 분할.
    - 플레이스홀더 행([그림/표/수식 N])은 독립 청크
    - 나머지는 max_chars 이하로 문장 단위 분할
    """
    # 줄 단위로 분리 (빈 줄 포함)
    lines = text.splitlines(keepends=True)
    if not lines:
        return [text]

    chunks: list[str] = []
    current = ""

    for line in lines:
        is_placeholder = bool(re.match(r"\[(그림|표|수식)\s*\d+", line))

        if is_placeholder:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line)
            continue

        # 현재 청크에 추가하면 한계 초과 → 먼저 저장
        if current and len(current) + len(line) > max_chars:
            chunks.append(current)
            current = line
        else:
            current += line

    if current:
        chunks.append(current)

    # 단일 행이 max_chars 초과하는 경우 강제 분할
    result: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            result.append(chunk)
        else:
            for i in range(0, len(chunk), max_chars):
                result.append(chunk[i:i + max_chars])

    return result if result else [text]
