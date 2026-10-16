"""비용 우선(내장 정제규칙) 경로의 OCR 잡음·단락 규칙 — 2026-10-16.

실측(01_입력 명언집 5권, RapidOCR → _rules_clean 결과 02_결과):
  잡음 줄 990 · 잡음 토큰 1,918 · 전각 부호 3,410 · 깨진 머리글 잔류 725 ·
  한 단어씩 끊긴 영어 줄 194, 그리고 번호 붙은 명언 항목이 빈 줄 없이 이어지거나
  한 줄에 다음 항목이 붙어(… 기술교 408 | 시간을…) 단락 구분이 거의 없었다.

원칙: 모양이 뚜렷한 잡음만 지운다. 본문일 수 있으면 남긴다(원문 사본은 storage 에 보존).
  - 쪽 단위(clean_page): 전각→반각, 잡음 토큰, 깨진 항목 번호 복원, 잡음 줄, 영어 단어 줄 잇기
  - 문서 단위(strip_fuzzy_headers): 쪽마다 다르게 깨져 읽힌 러닝헤더('실라콘繼리 천재醫의…')
  - 문서 단위(paragraph_entries): 연속 번호(n, n+1, …) 항목마다 단락(빈 줄) 복원
"""
from __future__ import annotations

import re
from bisect import bisect_left as _bisect
from collections import Counter

# ── 쪽 단위 ────────────────────────────────────────────────────────────────────
_FULLWIDTH = str.maketrans({"，": ",", "：": ":", "；": ";", "（": "(", "）": ")",
                            "？": "?", "！": "!", "［": "[", "］": "]", "｜": "|",
                            "、": ",", "－": "-", "～": "~", "＞": ">", "＜": "<",
                            "。": ".", "＂": '"', "＇": "'"})

# 옛한글 자모(U+1100~11FF)는 현대 한국어 본문에 오지 않는다 — OCR 이 획을 읽은 것('ᄀ 하지만', 'ᄂf 이다')
_JAMO_TOKEN_RE = re.compile(r"(?<!\S)[ᄀ-ᇿ][A-Za-z]{0,2}(?!\S)")
_JAMO_CHAR_RE = re.compile(r"[ᄀ-ᇿ]")
# 장식 기호 덩어리('■■■', '•', '^', '«') — 토큰 전체가 기호일 때만
_SYMBOL_TOKEN_RE = re.compile(r"(?<!\S)[■□▪▫◆◇●○•^«»°◊※*]+(?!\S)")
_SYMBOL_LEAD_RE = re.compile(r"(?<!\S)[■□▪▫◆◇●○•^«»]+(?=[가-힣A-Za-z“‘\"'(])")
# 스캔 밑줄·점선('______', '-------')
_RULE_RE = re.compile(r"[_\-–—―一]{5,}")
# 깨진 항목 번호: 'OO52'·'O457' → '0052'·'0457', '0 0 5 1' → '0051'
_O_NUM_RE = re.compile(r"(?<![\w])([0O]{1,3})(\d{1,3})(?![\w])")
_SPACED_NUM_RE = re.compile(r"(?<![\w\d])(\d) (\d) (\d)(?: (\d))?(?![\w\d])")

_ALLOWED_PUNCT = set(".,!?'\"“”‘’()[]<>〈〉『』「」《》:;-–—~·%/…|_+=&#@")


# 숫자만 있는 줄의 o·i 오인식('ooi3' → '0013', 'OO7O' → '0070') — 줄 전체가 번호이고 숫자가 하나
# 이상 섞였을 때만(영단어 오변환 방지)
_GARBLED_NUM_LINE_RE = re.compile(r"(?m)^[ \t]*((?=[0-9OoilI]*\d)[0-9Oo][0-9OoilI]{1,3})[ \t]*$")


def _fix_numbers(text: str) -> str:
    text = _GARBLED_NUM_LINE_RE.sub(
        lambda m: m.group(1).translate(str.maketrans("OoilI", "00111"))
        if re.search(r"\d", m.group(1)) else m.group(0), text)
    text = _O_NUM_RE.sub(lambda m: m.group(1).replace("O", "0") + m.group(2), text)
    return _SPACED_NUM_RE.sub(lambda m: "".join(g for g in m.groups() if g), text)


def is_garbage_line(line: str) -> bool:
    """이미지·장식을 읽은 잡음 줄인가. 의미 있는 토큰(한글 2음절↑·영문 3자↑)이 없거나,
    짧은 줄에서 허용 외 기호가 30% 이상이면 잡음. 숫자만 있는 줄(항목 번호·연도)은 남긴다."""
    s = line.strip()
    if not s:
        return False
    ns = re.sub(r"\s", "", s)
    if re.fullmatch(r"[\d\-.,:/()|]+", ns):
        return False
    if not re.search(r"[가-힣]{2}|[A-Za-z]{3}", s):
        return True
    bad = sum(1 for c in ns if not (re.match(r"[가-힣A-Za-z0-9]", c) or c in _ALLOWED_PUNCT))
    return len(ns) <= 60 and bad / len(ns) >= 0.3


_KEEP_HANGUL = 6           # 한글이 이만큼 있는 줄은 본문 — 통째로 지우지 않고 양끝 잡음 토큰만 뗀다


def _is_junk_token(t: str) -> bool:
    return not re.search(r"[가-힣]|[A-Za-z]{3}|^\d{1,4}[.,]?$|^[|“”‘’\"'(]$", t)


def _trim_junk_edges(line: str) -> str:
    """'/1 r* xr \\ x - 나항상 선택의…' → '나항상 선택의…' (본문 앞뒤에 붙은 잡음 토큰 제거)."""
    toks = line.split()
    a, b = 0, len(toks)
    while a < b and _is_junk_token(toks[a]):
        a += 1
    while b > a and _is_junk_token(toks[b - 1]):
        b -= 1
    return " ".join(toks[a:b])


_HANJA_RE = re.compile(r"[一-鿿㐀-䶿々]")


def _is_noise_token(t: str) -> bool:
    """본문 줄 안에 섞인 잡음 토큰('가令:乂■아', '乂', '#법을')인가. 한자 병기('(溫故知新)')는 본문."""
    if re.fullmatch(r"[(\[]?[一-鿿]{2,}[)\]]?[.,]?", t):
        return False
    if re.search(r"[가-힣]{2}", t):      # 한글 낱말이 있으면 토큰은 남기고 앞뒤 잡음만 뗀다(_strip_token_edges)
        return False
    hanja = len(_HANJA_RE.findall(t))
    bad = sum(1 for c in t if not (re.match(r"[가-힣A-Za-z0-9]", c) or c in _ALLOWED_PUNCT)) - hanja
    good = re.search(r"[A-Za-z]{3}|\d", t)
    return (not good and bool(bad or hanja)) or (bad + hanja >= 2 and bad + hanja >= len(t) / 2)


def _strip_token_edges(t: str) -> str:
    """'一오스틴' → '오스틴', '臣麗因주기도문은' → '주기도문은' (한글 어절 앞뒤에 붙은 한자·기호)."""
    if not re.search(r"[가-힣]{2}", t) or re.search(r"\([一-鿿]+\)", t):
        return t
    return re.sub(r"^[一-鿿㐀-䶿々^■•*«»~`]+|[一-鿿㐀-䶿々^■•*«»~`]+$", "", t)


def _clean_line(line: str) -> str:
    """잡음 줄이면 '', 본문 줄이면 섞인 잡음 토큰과 양끝 잡음을 뗀 줄."""
    line = " ".join(_strip_token_edges(t) for t in line.split() if not _is_noise_token(t))
    hangul = len(re.findall(r"[가-힣]", line))
    if hangul >= _KEEP_HANGUL:
        return _trim_junk_edges(line)
    # 한글이 적은 짧은 줄: 한자가 섞였거나 한글 낱말·영단어(모음 포함 3자↑)가 없으면 잡음
    if not re.fullmatch(r"[\d\-.,:/()| ]+", line.strip()) and (
            _HANJA_RE.search(line)
            or not (re.search(r"[가-힣]{2}", line)
                    or re.search(r"(?=[A-Za-z]{3})[A-Za-z]*[aeiouyAEIOUY][A-Za-z]*", line))):
        return ""
    return "" if is_garbage_line(line) else line


def _join_latin_word_lines(lines: list[str]) -> list[str]:
    """한 단어씩 끊긴 영어 줄('Our' / 'success' / 'has' …)을 앞 줄에 잇는다."""
    out: list[str] = []
    for l in lines:
        s = l.strip()
        if (out and re.fullmatch(r"[A-Za-z’'.,;:!?-]{1,20}", s)
                and re.search(r"[A-Za-z’'.,;:-]$", out[-1].rstrip())):
            out[-1] = out[-1].rstrip() + " " + s
        else:
            out.append(l)
    return out


def clean_page(text: str) -> str:
    """쪽 하나의 OCR 잡음을 걷어낸다(순서 중요: 부호 정리 → 번호 복원 → 잡음 줄 판정)."""
    text = text.translate(_FULLWIDTH)
    text = _RULE_RE.sub(" ", text)
    text = _JAMO_TOKEN_RE.sub(" ", text)
    text = _JAMO_CHAR_RE.sub("", text)
    text = _SYMBOL_TOKEN_RE.sub(" ", text)
    text = _SYMBOL_LEAD_RE.sub("", text)
    text = _fix_numbers(text)
    lines = [re.sub(r"[ \t]{2,}", " ", l).strip() for l in text.splitlines()]
    lines = [c for c in (_clean_line(l) for l in lines) if c]
    return "\n".join(_join_latin_word_lines(lines))


# ── 문서 단위: 깨진 러닝헤더 ─────────────────────────────────────────────────────
_GRAM = 4
_HEADER_MIN_SPAN = 8       # 머리글로 지울 구간의 최소 글자(한글·영문) 수('천재■의생각아포리즘' 9자)
_HEADER_COVER = 0.5        # 구간 4글자 조각 중 머리글 조각 비율
DEBUG_LOG: list | None = None   # 점검용 — 리스트를 넣으면 (원래 줄, 지운 뒤 줄)을 모은다


def _skel(s: str) -> str:
    return re.sub(r"[^가-힣A-Za-z]", "", s).lower()


def _grams(sk: str) -> set[str]:
    return {sk[i:i + _GRAM] for i in range(len(sk) - _GRAM + 1)}


def _header_grams(pages: list[str]) -> set[str]:
    """쪽 가장자리(첫·끝 2줄)에 여러 쪽 반복되는 4글자 조각 중, 본문 쪽에는 드문 것."""
    edge_df, all_df = Counter(), Counter()
    for p in pages:
        lines = [l for l in p.splitlines() if l.strip()]
        # 가장자리 줄의 양끝 30자만 — 머리글은 본문 줄 앞뒤에 붙어 읽힌다. 줄 전체를 보면
        # 쪽 끝에 자주 오는 영어 명언 줄의 흔한 조각('that'·'ally')까지 학습된다
        edge = [s for l in lines[:2] + lines[-2:] for s in (l.strip()[:30], l.strip()[-30:])]
        edge_df.update(set().union(*(_grams(_skel(l)) for l in edge)) if edge else set())
        all_df.update(_grams(_skel(p)))
    need = max(5, int(len(pages) * 0.03))
    return {g for g, n in edge_df.items() if n >= need and n >= 0.6 * all_df[g]}


# 쪽번호 토큰 — 깨져 읽힌 꼴 포함('•)88', '8(i', '|]8', '9()(5', '(J2', ']$)($')
_PAGE_NO_TOKEN_RE = re.compile(r"^(?=.*[\d$])[•·|I()\[\]$]*[\dIlOoSiJ()\[\]$?]{1,5}[•·|)\]]*$|^\d{1,4}$")


def _strip_header_spans(line: str, hg: set[str]) -> str:
    """머리글 조각에 글자 대부분이 덮인 어절이 이어진 구간(합계 _HEADER_MIN_SPAN자↑)만 지운다.
    구간 판정을 어절 단위로 해야 머리글 앞뒤의 본문 어절이 함께 지워지지 않는다."""
    toks = line.split()
    sks = [_skel(t) for t in toks]
    line_sk = "".join(sks)
    covered = [False] * len(line_sk)
    for p in range(len(line_sk) - _GRAM + 1):
        if line_sk[p:p + _GRAM] in hg:
            for q in range(p, p + _GRAM):
                covered[q] = True
    hg3 = {g[p:p + 3] for g in hg for p in range(2) if re.fullmatch(r"[가-힣]{3}", g[p:p + 3])}
    hdr, pos = [], 0
    for sk in sks:
        c = sum(covered[pos:pos + len(sk)])
        hdr.append(len(sk) >= 2 and c / len(sk) >= 0.6)
        pos += len(sk)
    drop: set[int] = set()
    i = 0
    while i < len(toks):
        if not hdr[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(toks) and (hdr[j + 1] or (not sks[j + 1] and j + 2 < len(toks) and hdr[j + 2])):
            j += 1
        if sum(len(sks[k]) for k in range(i, j + 1) if hdr[k]) >= _HEADER_MIN_SPAN:
            # 바로 앞 어절이 더 심하게 깨진 머리글 조각('실리콘IB리')이면 함께 — 머리글 3글자 조각을 품을 때만
            if i > 0 and len(sks[i - 1]) >= 3 and any(sks[i - 1][p:p + 3] in hg3
                                                       for p in range(len(sks[i - 1]) - 2)):
                i -= 1
            span = set(range(i, j + 1))
            # 머리글 바로 앞뒤의 쪽번호 토큰 하나씩('164 실리콘…', '… sellers •)88')
            k = i - 1
            if k >= 0 and toks[k] in ("I", "|", "•", ":", "'", "’", '"'):   # '146 I 1000 WISE…'
                span.add(k)
                k -= 1
            if k >= 0 and _PAGE_NO_TOKEN_RE.match(toks[k]):
                span.add(k)
            # 영문 머리글 끝 단어가 깨져 조각이 안 맞는 경우('… MASTERPIECE RLMS')
            if j + 1 < len(toks) and re.fullmatch(r"[A-Z]{2,6}", toks[j + 1]) \
                    and re.fullmatch(r"[A-Za-z]+", toks[j]):
                j += 1
                span.add(j)
            # 뒤따르는 쪽번호 하나 — 단 '407 I 제목'처럼 항목 번호+구분자이면 남긴다
            if j + 1 < len(toks) and _PAGE_NO_TOKEN_RE.match(toks[j + 1]) \
                    and not (j + 2 < len(toks) and toks[j + 2] in ("I", "|")):
                span.add(j + 1)
            drop |= span
        i = j + 1
    return " ".join(t for k, t in enumerate(toks) if k not in drop) if drop else line


# 'PART 3 실라콘뱅리 혁신가들의 통학 331' — 장 머리글+쪽번호가 본문 줄 가운데에 끼어든 것.
# 쪽번호(1~3자리)가 뒤따를 때만 머리글로 본다. 번호가 둘이면('명대사 51 115 언제') 앞 것만 쪽번호.
_PART = r"(?:P ?A ?R ?T|PA.T|P.RT|part)"
_PART_HEADER_RE = re.compile(rf"\b{_PART}\s?\d{{1,2}}\b[^\n]{{0,40}}?(?<!\S)[•·|I]?\s*\d{{1,3}}(?=\s|$)")
_PART_LINE_RE = re.compile(rf"^{_PART}\s?(\d{{1,2}})\b.{{0,40}}$")


def _strip_part_headers(pages: list[str]) -> tuple[list[str], int]:
    seen: set[str] = set()
    out, n = [], 0
    for p in pages:
        new = []
        for l in p.splitlines():
            s = re.sub(r" {2,}", " ", _PART_HEADER_RE.sub(" ", l)).strip()
            if s == l.strip():
                s = l
            m = _PART_LINE_RE.match(s.strip())
            if m:                   # 번호 없이 'PART N 장제목' 만 있는 줄 — 첫 등장(장 표제)만 남긴다
                if m.group(1) in seen:
                    s = ""
                seen.add(m.group(1))
            if s != l:
                n += 1
            if s.strip() or not l.strip():
                new.append(s)
        out.append("\n".join(new))
    return out, n


def strip_fuzzy_headers(pages: list[str]) -> tuple[list[str], int]:
    """쪽마다 다르게 깨져 읽힌 머리글 구간을 지운다. (쪽들, 지운 줄 수)"""
    pages, n_part = _strip_part_headers(pages)
    hg = _header_grams(pages)
    if not hg:
        return pages, n_part
    out, n = [], n_part
    for p in pages:
        new = []
        for l in p.splitlines():
            s = _strip_header_spans(l, hg)
            if s != l and is_garbage_line(s):       # 머리글을 떼고 남은 자투리('1 期')
                s = ""
            if s != l and DEBUG_LOG is not None:
                DEBUG_LOG.append((l, s))
            if s != l:
                n += 1
            if s.strip() or not l.strip():
                new.append(s)
        out.append("\n".join(new))
    return out, n


# ── 문서 단위: 앞뒤 부속물(규칙 판정) ─────────────────────────────────────────────
# Gemini 문단 분류(drop_front_paragraphs)는 비용 우선에서 쓰지 않는다. 실측(명언집 5권, 부속물 삭제
# 켜고 처리): 판권(마지막 쪽)과 '차례' 제목 쪽 1장만 지워지고 표지·저자 소개·다른 책 광고·저작권 쪽·
# 차례 이어지는 쪽이 그대로 남았다. 모양이 뚜렷한 것만 규칙으로 고른다 — 머리말·프롤로그는 본문.
_FRONT_MAX_PAGES = 40
_PRICE_AD_RE = re.compile(r"\d[\d,.]{3,6}\s*원")
_PRICE_CTX_RE = re.compile(r"지음|옮김|(?<![가-힣])저(?![가-힣])|(?<![가-힣])역(?![가-힣])|\d\s*면\b|\d\s*쪽")
_COPYRIGHT_RE = re.compile(r"Copyright|©|저작권법|무단\s*(?:전재|복제)|ISBN|All rights reserved|출판권", re.I)
_BIO_WORDS = ("저자", "지은이", "옮긴이", "대표이사", "큐레이터", "졸업", "학위", "저서", "출간", "강연",
              "페이스북", "블로그", "http", "인문학자", "facebook")
_BODY_START_RE = re.compile(r"^(?:prologue|프롤로그|머리말|서문|들어가며|들어가는 글|여는 글)", re.I)


def _is_toc_page(p: str) -> bool:
    """차례 쪽 꼴: 짧은 줄이 많고 상당수가 쪽번호(숫자)로 끝나거나 숫자뿐인 쪽."""
    lines = [l.strip() for l in p.splitlines() if l.strip()]
    if len(lines) < 12:
        return False
    avg = sum(len(l) for l in lines) / len(lines)
    numeric = sum(1 for l in lines if re.search(r"(?:^|\s|[가-힣.])\d{1,3}$", l))
    return avg <= 30 and numeric / len(lines) >= 0.2


def _is_toc_part_page(p: str) -> bool:
    """'Part 10 / 장 제목 / #태그 #태그 …' 꼴 차례 쪽(장별 차례)."""
    return bool(re.search(r"(?i)\bpart\s?\d", p)) and len(re.findall(r"#[가-힣A-Za-z]", p)) >= 3


def drop_book_matter_rules(pages: list[str], notes: list[str]) -> list[str]:
    """비용 우선(규칙) 경로의 앞뒤 부속물 제거. 지운 쪽은 빈 자리로 두고 notes 에 보고."""
    out = list(pages)
    n = len(out)
    win = min(n, _FRONT_MAX_PAGES, max(10, n // 6))
    found: dict[str, list[int]] = {}

    def drop(i: int, kind: str) -> None:
        if out[i].strip():
            out[i] = ""
            found.setdefault(kind, []).append(i + 1)

    # 본문 시작 경계: 머리말·프롤로그 제목 쪽과 첫 차례 쪽 중 앞선 것. 둘 다 없으면 600자 넘는 첫 쪽
    first = lambda pred: next((i for i in range(win) if pred(out[i])), None)
    head_lines = lambda p: [l.strip() for l in p.splitlines() if l.strip()][:3]
    marks = [x for x in (first(lambda p: any(_BODY_START_RE.match(l) for l in head_lines(p))),
                         first(_is_toc_page)) if x is not None]
    bound = min(marks) if marks else (first(lambda p: len(p.strip()) >= 600) or 0)
    # 문장으로 된 쪽(머리말 등)은 표지·소개로 보지 않는다 — '…다.' 꼴 종결이 둘 이상이면 본문
    sentences = lambda p: len(re.findall(r"[다요죠][.!?]?(?:\s|$)", p))

    for i in range(win):
        p = out[i]
        if not p.strip():
            continue
        if _PRICE_AD_RE.search(p) and _PRICE_CTX_RE.search(p):
            drop(i, "다른 책 광고")
        elif len(_COPYRIGHT_RE.findall(p)) >= 2:
            drop(i, "저작권·판권")
        elif i < bound and len(p) < 1500 and (
                re.search(r"(?m)^\s*(?:저자|지은이|옮긴이|역자)\s", p)
                or sum(w in p for w in _BIO_WORDS) >= 3):
            drop(i, "저자 소개")
        elif i < bound and len(p.strip()) < 500 and sentences(p) < 2:
            drop(i, "표지·제목")
    # 차례: 앞쪽의 첫 차례 덩어리만 — 차례 꼴(또는 장별 차례 꼴) 쪽이 이어지는 동안.
    # 본문에도 '번호 / 짧은 제목' 줄이 많은 쪽이 있어(기독교명언) 덩어리 하나에서 멈춘다
    start = first(_is_toc_page)
    if start is not None:
        j = start
        while j < win and (not out[j].strip() or _is_toc_page(out[j]) or _is_toc_part_page(out[j])):
            drop(j, "차례")
            j += 1
    # 뒤쪽 다른 책 광고
    for i in range(max(win, n - 5), n):
        if _PRICE_AD_RE.search(out[i]) and _PRICE_CTX_RE.search(out[i]):
            drop(i, "다른 책 광고")
    for kind, idxs in found.items():
        notes.append(f"앞부속·뒤부속 자동 제거 — {kind} {', '.join(map(str, idxs))}번째 쪽"
                     " (정제 전 원문 사본에서 확인 가능)")
    return out


# ── 문서 단위: 번호 항목 단락 ──────────────────────────────────────────────────
# 항목 번호 후보: 단독 숫자 토큰(뒤에 단위·구두점이 붙지 않음)
# (단위는 숫자에 바로 붙은 것만 본다 — '0001 명언' 의 '명'을 단위로 오인하지 않게)
_ENTRY_RE = re.compile(r"(?<![\w(\-:.,/])(\d{1,4})(?![\w%)\-:.,/~])")
_ENTRY_GAP = 4             # OCR 이 놓친 번호를 건너뛰어도 되는 폭
_ENTRY_LOOKBACK = 400      # DP 에서 되돌아보는 후보 수
_ENTRY_MIN_CHAIN = 30      # 이보다 짧은 연쇄는 번호 항목 책이 아니다(본문 숫자 우연 일치)


def _entry_chain(text: str) -> list[re.Match]:
    cands = [m for m in _ENTRY_RE.finditer(text)]
    vals = [int(m.group(1)) for m in cands]
    best = [1] * len(cands)
    prev = [-1] * len(cands)
    for j in range(len(cands)):
        for i in range(max(0, j - _ENTRY_LOOKBACK), j):
            if 1 <= vals[j] - vals[i] <= _ENTRY_GAP and best[i] + 1 > best[j]:
                best[j], prev[j] = best[i] + 1, i
    if not cands:
        return []
    k = max(range(len(cands)), key=lambda x: best[x])
    chain = []
    while k != -1:
        chain.append(cands[k])
        k = prev[k]
    return chain[::-1]


# ── 문서 단위: 몰려 읽힌 번호를 본문에 되돌리기 ──────────────────────────────────
# 실측(시네마명언 14곳·실리콘밸리 68곳): 번호가 본문 왼쪽 여백에 따로 인쇄된 책은 RapidOCR 이
# 한 쪽의 번호들('014' '015')을 먼저 몰아 읽고 본문을 뒤에 읽는다 → 단락 복원 후 번호만 있는
# 빈 항목(실리콘밸리 173개)이 생기고 본문은 번호 없이 앞 항목에 붙었다. 내용 손실이 아니라 순서 어긋남.
# 번호만 있는 줄. 쪽 경계에서 앞 쪽 쪽번호가 붙어 '19 014' 가 된 줄도 포함(쪽번호는 버린다)
_NUM_LINE_RE = re.compile(r"^\s*(?:\d{1,3}\s+)?(\d{1,4})\s*$")
_SENT_END_RE = re.compile(r"(?:[.!?”\"’')\]]|[다요죠까니])\s*$")
_BOUND_HEAD_RE = re.compile(
    r"^(?:며|으며|니|으니|지만|면서|도록|거나|는데|은데|으면|는|은|을|를|에|의|에서|에게|으로|로"
    r"|었다|았다|였다|했다|한다|된다|합니다|입니다|습니다|니다|리는|하는|이다)(?![가-힣])")


def _is_unit_start(line: str, nxt: str = "") -> bool:
    """본문 덩어리의 첫 줄 — 한글로 시작. 12자 이하 짧은 줄은 소제목('사라지지 않는다')일 수 있어
    바로 뒤에 영어 번역 줄이 올 때만 덩어리로 본다."""
    s = line.strip()
    if not re.match(r"[“\"'‘(]?[가-힣]", s):
        return False
    return len(s) > 12 or bool(re.match(r"\s*[A-Z][a-z]", nxt))


def redistribute_number_stacks(text: str) -> tuple[str, int]:
    """'014 / 015 / 본문A / 본문B' 처럼 몰려 읽힌 연속 번호를 뒤따르는 본문 덩어리 머리에 차례로 붙인다.
    번호 바로 뒤의 앞 항목 꼬리(영어 번역 줄·끊긴 문장 잇는 줄)는 건너뛴다. (텍스트, 옮긴 번호 수)"""
    raw = text.split("\n")
    lines: list[str] = []
    for k, l in enumerate(raw):
        # 쪽 경계에서 몰림의 앞 번호들이 앞 문장 끝에 붙은 경우('…날인 0027' / '0028',
        # '…한다 0032 0033' / '0034') — 연속 번호면 떼어 낸다
        m_tail = re.match(r"^(.*?[가-힣A-Za-z.,!?”\"’')][^\d]*?)((?:\s+\d{2,4})+)\s*$", l)
        if m_tail:
            tail = [int(x) for x in m_tail.group(2).split()]
            nxt = _NUM_LINE_RE.match(raw[k + 1]) if k + 1 < len(raw) else None
            seq = all(b - a == 1 for a, b in zip(tail, tail[1:]))
            if seq and ((nxt and int(nxt.group(1)) == tail[-1] + 1) or len(tail) >= 2):
                lines += [m_tail.group(1).rstrip()] + m_tail.group(2).split()
                continue
        # 영어 번역 줄 끝에 다음 항목의 한글이 붙은 줄('…define it 행동가들은 …') — 한글 앞에서 나눈다
        m_mix = re.match(r"^([A-Za-z‘'\"].*?[A-Za-z.,!?’'\"])\s+([“\"'‘]?[가-힣]{2,}.*)$", l)
        if m_mix and len(re.findall(r"[가-힣]", m_mix.group(2))) >= 6 \
                and len(re.findall(r"[A-Za-z]{2,}", m_mix.group(1))) >= 3:   # 영어 쪽이 문장일 때만('Boss 매출이…' 제외)
            lines += [m_mix.group(1), m_mix.group(2)]
            continue
        lines.append(l)
    out: list[str] = []
    moved = 0
    i = 0
    prev_num = None
    while i < len(lines):
        m = _NUM_LINE_RE.match(lines[i])
        j = i
        while j < len(lines) and _NUM_LINE_RE.match(lines[j]):
            j += 1
        vals = [int(_NUM_LINE_RE.match(lines[k]).group(1)) for k in range(i, j)]
        if not m or len(vals) < 2 or any(b - a != 1 for a, b in zip(vals, vals[1:])):
            lm = re.match(r"^\s*(\d{1,4})(?=\s|$)", lines[i])
            if lm:
                prev_num = int(lm.group(1))
            out.append(lines[i])
            i += 1
            continue
        # 번호 몰림 [i, j) — 뒤따르는 덩어리 범위: 다음 번호 줄 전까지. 빈 줄(쪽 경계)은 덩어리가
        # 아직 모자랄 때만 넘는다(최대 40줄) — 본문이 다음 쪽으로 넘어가 읽힌 경우
        end = j
        found = 0
        while end < len(lines) and end - j < 40 and not _NUM_LINE_RE.match(lines[end]) \
                and not re.match(r"^\s*\d{2,4}\s+\S", lines[end]):
            if not lines[end].strip() and found >= len(vals) + 1:
                break
            if _is_unit_start(lines[end], lines[end + 1] if end + 1 < len(lines) else ""):
                found += 1
            end += 1
        while end > j and not lines[end - 1].strip():
            end -= 1
        starts = [k for k in range(j, end)
                  if _is_unit_start(lines[k], lines[k + 1] if k + 1 < end else "")]
        # 앞 항목 꼬리: 첫 한글 줄이 조사·어미로 시작하거나, 번호 몰림 앞 본문이 문장 중간에서 끊겼으면.
        # 앞 본문 = 번호 몰림 바로 앞 줄(영어 번역 줄이면 그 앞 한 줄). 'PART 1 … 명대사' 같은
        # 머리글·20자 미만 소제목이면 끊긴 문장이 아니다
        prev_lines = [l for l in out[-3:] if l.strip()]
        before = prev_lines[-1] if prev_lines else ""
        if before and not re.search(r"[가-힣]", before) and len(prev_lines) >= 2:
            before = prev_lines[-2]
        if len(before.strip()) < 20 or _PART_LINE_RE.match(before.strip()) or re.match(rf"{_PART}\s?\d", before.strip()):
            before = ""
        # 앞 문장이 끊겼다는 정황만으로는 덩어리가 번호보다 많을 때만 건너뛴다 — 모자라면 첫 덩어리도
        # 새 항목이다(실리콘밸리 0198~0202 실측: '저는 여러분이…'를 꼬리로 보내 0201·0202 가 비었다)
        if starts and (_BOUND_HEAD_RE.match(lines[starts[0]].strip())
                       or (before and not _SENT_END_RE.search(before) and starts[0] == j
                           and len(starts) > len(vals))):
            starts = starts[1:]
        k_n = len(vals)
        gap = vals[0] - prev_num - 1 if prev_num is not None and 0 < vals[0] - prev_num <= 5 else 0
        offset = min(gap, len(starts) - k_n) if len(starts) > k_n else 0
        nums = [_NUM_LINE_RE.match(lines[k]).group(1) for k in range(i, j)]
        assign = dict(zip(starts[offset:], nums))
        left = nums[len(assign):]   # 붙일 덩어리가 모자라면 남은 번호는 덩어리 뒤에 그대로(순서 유지)
        for k in range(j, end):
            out.append(f"{assign[k]} {lines[k].lstrip()}" if k in assign else lines[k])
        out.extend(left)
        moved += len(assign)
        prev_num = vals[-1]
        i = end
    return "\n".join(out), moved


def paragraph_entries(text: str) -> tuple[str, int]:
    """연속 번호 항목(0451, 0452 …)마다 앞에 빈 줄을 넣어 단락을 복원한다.
    번호 바로 앞에 같은 줄로 붙은 쪽번호('157 453')는 함께 뗀다. (텍스트, 단락 수)"""
    chain = _entry_chain(text)
    if len(chain) < _ENTRY_MIN_CHAIN:
        return text, 0
    # 항목 번호 연쇄에 속하지 않는, 숫자만 있는 줄 중 바로 앞 항목 번호보다 확연히 작은 수 = 홀로 남은
    # 쪽번호('0238 …' 뒤의 '91') — 지운다. 항목 번호와 쪽번호 값대가 겹치는 책(기독교명언 앞부분)은
    # 구분할 수 없으므로 남긴다
    pos = [m.start() for m in chain]
    vals = [int(m.group(1)) for m in chain]

    def _keep(m: re.Match) -> str:
        k = _bisect(pos, m.start(1))
        if k < len(pos) and pos[k] == m.start(1):
            return m.group(0)
        prev = vals[k - 1] if k > 0 else None
        v = int(m.group(1))
        return "" if prev is not None and v <= prev * 0.6 else m.group(0)

    # 쪽번호 꼴: 앞 줄은 본문, 뒤는 빈 줄·끝 — 단락 끝에 홀로 붙은 숫자 줄만 본다
    text = re.sub(r"(?m)(?<=[^\d\s]\n)[ \t]*(\d{1,3})[ \t]*(?=\n\s*\n|\Z)", _keep, text)
    chain = _entry_chain(text)
    out, last = [], 0
    for m in chain:
        before = text[last:m.start()]
        # 같은 줄에서 번호 바로 앞이 쪽번호뿐이면('157 453', '• 272') 그 쪽번호를 뗀다
        before = re.sub(r"(^|\n)[^\S\n]*[•·|I]?\s*\d{1,3}[^\S\n]+$", r"\1", before)
        before = before.rstrip(" \t")
        # 번호 바로 앞 줄이 짧은 소제목('습관')이면 앞 항목 꼬리가 아니라 다음 항목의 머리 — 함께 묶는다
        head = ""
        tm = re.search(r"\n([가-힣][가-힣 ]{0,11})\n?$", before)
        if tm and before[:tm.start()].strip():
            head, before = tm.group(1) + "\n", before[:tm.start()]
        out.append(before)
        out.append("\n\n" + head)
        last = m.start()
    out.append(text[last:])
    res = re.sub(r"[ \t]+\n", "\n", "".join(out))
    res = re.sub(r"\n{3,}", "\n\n", res).lstrip("\n")
    return res, len(chain)
