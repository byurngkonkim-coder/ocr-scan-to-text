"""L1 규칙 기반 문자 보정 — Unicode 정규화·합자·특수기호·한글분절·영문혼동"""
import re
import unicodedata

# ── 1. Unicode 합자 분해 ───────────────────────────────────────────────────────
LIGATURES: dict[str, str] = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "st",
    "æ": "ae",
    "œ": "oe",
}

# ── 2. 특수기호·단위·약어 치환 ─────────────────────────────────────────────────
SYMBOL_MAP: dict[str, str] = {
    # 법적·상표 기호
    "(c)": "©", "(C)": "©",
    "(r)": "®", "(R)": "®",
    "(tm)": "™", "(TM)": "™",
    # 수학·과학
    "+-": "±", "->": "→", "<-": "←",
    ">=": "≥", "<=": "≤", "!=": "≠",
    # 단위
    "oC": "℃", "deg C": "℃",
    "m2": "m²", "km2": "km²",
    # 약어
    "e,g,": "e.g.", "i,e,": "i.e.",
    "etc,": "etc.", "No,": "No.",
    # [A안] 구두점 변환(...→…, --→—)은 원문 보존을 위해 제거.
    #       말줄임표·대시 정규화는 하위 정제 파이프라인(02_01)에서 처리.
}

def normalize_all(text: str) -> str:
    """L1 보정 전체 파이프라인 실행"""
    text = unicodedata.normalize("NFKC", text)
    text = _remove_invisible(text)          # N-06: 비가시 문자 제거
    text = _expand_ligatures(text)
    text = _apply_symbol_map(text)
    text = fix_english_confusion(text)
    text = fix_korean_spaces(text)
    text = fix_hyphenation(text)
    return text


def has_jamo_decomposition(text: str) -> bool:
    """
    자모 분리 감지 — 전체 한글 관련 문자 중 단독 자모 비율 > 5% 이면 True.
    폴백 트리거 조건.
    """
    jamo_count = sum(1 for c in text if "㄰" <= c <= "㆏")
    total = sum(
        1 for c in text
        if "가" <= c <= "힣" or "㄰" <= c <= "㆏"
    )
    return total > 0 and (jamo_count / total) > 0.05


# ── 내부 함수 ──────────────────────────────────────────────────────────────────

def _remove_invisible(text: str) -> str:
    """폭 없는/조인 문자 제거 (BOM·ZWSP·ZWNJ·ZWJ·WORD JOINER).

    OCR·복사붙여넣기 과정에서 섞여 들어와 검색·중복판정·후처리를 방해하는
    비가시 문자를 정리한다. (텍스트_정규화_규칙서 N-06 참조)
    """
    return re.sub("[\ufeff\u200b\u200c\u200d\u2060]", "", text)


def _expand_ligatures(text: str) -> str:
    for ligature, replacement in LIGATURES.items():
        text = text.replace(ligature, replacement)
    return text


def _apply_symbol_map(text: str) -> str:
    for src, dst in SYMBOL_MAP.items():
        text = text.replace(src, dst)
    return text


def fix_english_confusion(text: str) -> str:
    """영문 세리프 혼동 컨텍스트 기반 교정 (L1, ~70~80% 자동)"""
    # 숫자 사이 I→1, l→1, S→5, B→8, G→6, O→0
    confusion = {"I": "1", "l": "1", "S": "5", "B": "8", "G": "6"}
    for wrong, correct in confusion.items():
        text = re.sub(
            rf"(?<=[0-9]){re.escape(wrong)}(?=[0-9])",
            correct, text
        )
    text = re.sub(r"(?<=[0-9])O(?=[0-9])", "0", text)

    # [제거] 어두 l/1 → 대문자 I 규칙: "let"→"Iet", "look"→"Iook" 처럼
    #        정상 영단어를 손상시켜 삭제함 (저정밀 규칙, 원문 보존 원칙)

    # 소문자 단어 내부 I/1 → l (고정밀: 양쪽이 소문자일 때만, "shouId"→"should")
    text = re.sub(r"(?<=[a-z])[I1](?=[a-z])", "l", text)

    return text


def fix_korean_spaces(text: str) -> str:
    """한글 줄끝 하이픈 분리 연결.

    [변경] 기존 '한글-공백-한글-공백-한글 → 공백 제거' 규칙은 정상적으로 띄어 쓴
    단음절 단어("이 그 저" 등)까지 붙여버려 원문을 훼손하므로 제거함.
    (원문 보존 원칙 — 텍스트_정규화_규칙서·명언_정제_규칙집과 동일 철학)
    """
    # 줄끝 하이픈으로 분리된 한글 연결만 수행
    text = re.sub(
        r"([가-힣])-\n([가-힣])",
        r"\1\2", text
    )
    return text


def fix_hyphenation(text: str) -> str:
    """영어 줄끝 하이픈 연결"""
    return re.sub(r"([a-zA-Z])-\n([a-zA-Z])", r"\1\2", text)
