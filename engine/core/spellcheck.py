"""L2 사전 기반 맞춤법 검사 — py-hanspell (네이버 맞춤법 검사기)"""
from __future__ import annotations
import re
import time

_hanspell_ok: bool | None = None


def _check_hanspell() -> bool:
    global _hanspell_ok
    if _hanspell_ok is None:
        try:
            from hanspell import spell_checker  # noqa: F401
            _hanspell_ok = True
        except ImportError:
            _hanspell_ok = False
    return _hanspell_ok


# 한 번 호출에 보낼 최대 글자 수 (API 제한)
_CHUNK_SIZE = 500


def correct(text: str) -> tuple[str, list[str]]:
    """
    L2 맞춤법 교정 실행.

    Returns:
        (교정된 텍스트, 변경 목록)
        hanspell 미설치 시 원본 텍스트 반환.
    """
    if not _check_hanspell():
        # 오프라인 로컬 한국어 정제 엔진(korean_corrector) 실행
        try:
            from core.korean_corrector import refine_korean_text
            refined = refine_korean_text(text)
            changes = []
            if refined != text:
                changes.append("로컬 맞춤법·띄어쓰기 정제 규칙 적용")
            return refined, changes
        except Exception:
            return text, []

    from hanspell import spell_checker

    chunks = _split_chunks(text, _CHUNK_SIZE)
    corrected_parts: list[str] = []
    changes: list[str] = []

    for chunk in chunks:
        if not chunk.strip():
            corrected_parts.append(chunk)
            continue
        try:
            result = spell_checker.check(chunk)
            # 교정기가 조각 끝 공백·줄바꿈을 떼어 돌려줘도 원래 것을 붙인다(문장·문단 경계 보존)
            corrected_parts.append(result.checked.rstrip() + chunk[len(chunk.rstrip()):])
            # 변경된 단어 수집
            original_words = chunk.split()
            corrected_words = result.checked.split()
            for orig, corr in zip(original_words, corrected_words):
                if orig != corr:
                    changes.append(f"{orig} → {corr}")
            # API 과부하 방지
            time.sleep(0.2)
        except Exception:
            # 실패 시 원본 청크 유지 (배치 안정성)
            corrected_parts.append(chunk)

    return "".join(corrected_parts), changes


def correct_safe(text: str) -> str:
    """변경 목록 없이 교정 텍스트만 반환. 예외 발생 시 원본 반환."""
    try:
        corrected, _ = correct(text)
        return corrected
    except Exception:
        return text


def is_available() -> bool:
    return _check_hanspell()


# ── 내부 유틸 ──────────────────────────────────────────────────────────────────

def _split_chunks(text: str, size: int) -> list[str]:
    """
    문장 경계를 유지하면서 size 글자 이하로 분할.
    문장 구분자: 마침표·줄바꿈·느낌표·물음표
    """
    # 문장 뒤 공백·줄바꿈을 그 문장에 붙여 둔다 — re.split(r"…\s*") 은 공백을 먹어, 조각을 다시 이으면
    # '문장이다.둘째 문장'·문단 줄바꿈 소실이 났다(hanspell 설치 시, 감사 잠복 1건)
    sentences = [s for s in re.findall(r".*?(?:[.!?\n]\s*|$)", text, flags=re.S) if s]
    chunks: list[str] = []
    current = ""

    for sent in sentences:
        if len(current) + len(sent) <= size:
            current += sent
        else:
            if current:
                chunks.append(current)
            # 단일 문장이 size 초과 시 강제 분할
            while len(sent) > size:
                chunks.append(sent[:size])
                sent = sent[size:]
            current = sent

    if current:
        chunks.append(current)

    return chunks
