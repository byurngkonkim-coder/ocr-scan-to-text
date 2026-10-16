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


def _fix_numbers(text: str) -> str:
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
_HEADER_MIN_SPAN = 10      # 머리글로 지울 구간의 최소 글자(한글·영문) 수
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


def paragraph_entries(text: str) -> tuple[str, int]:
    """연속 번호 항목(0451, 0452 …)마다 앞에 빈 줄을 넣어 단락을 복원한다.
    번호 바로 앞에 같은 줄로 붙은 쪽번호('157 453')는 함께 뗀다. (텍스트, 단락 수)"""
    chain = _entry_chain(text)
    if len(chain) < _ENTRY_MIN_CHAIN:
        return text, 0
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
