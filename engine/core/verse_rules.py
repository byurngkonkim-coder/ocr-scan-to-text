"""장절 책(성경 등) 정제 규칙 R-15 — 2026-10-16.

실측(01_입력 king_james_bible.pdf, 1,568쪽 디지털 PDF → 02_결과 20261016_190831):
  - '제 1 장'·'제 1 편' 장 머리 1,184개와 관주 줄 '(창2:4­9; 욥38:4­11)'이 전부 사라짐
    (쪽 단위 잡음 판정이 '한글 낱말 없는 짧은 줄'로 지움)
  - 쪽 머리글 '창세기 3'과 쪽번호가 본문 문장 가운데에 끼어듦('빛이 있었고 창세기 1')
  - 앞 절 끝에 다음 절이 붙음('먹어도 되나 3 동산의…') — 줄 잇기가 절 번호를 모름
  - 한 음절로 넘어간 줄('라.')이 잡음으로 지워져 절 끝이 잘림('찾지 못하였더')

원칙: 디지털 PDF 원문 줄을 그대로 쓴다. 줄 끝 공백 = 낱말 경계(띄워 잇기), 공백 없음 = 낱말 중간(붙여 잇기).
  단위: 장 머리 / 소제목·해설 / 관주 / 절('N 본문', 번호는 n → n+1 순서일 때만 절로 본다).
"""
from __future__ import annotations

import re

_CHAPTER_RE = re.compile(r"^제\s*(\d{1,3})\s*([장편])$")
# 절 번호만 한 줄에 놓이고 본문이 다음 줄로 넘어간 경우('35' / '이 날 내가…')도 절 머리다
_VERSE_RE = re.compile(r"^(\d{1,3})(?:\s+\S|$)")
# 관주: 괄호 안에 '장:절'이 있는 줄 — 길면 다음 줄로 넘어가 닫는 괄호가 없을 수 있다
_XREF_RE = re.compile(r"^\([^()]*\d+:\d+[^()]*\)?$")
# 쪽 머리글('창세기 3'·'예레미야 애가 2'·'시편 6⋅7'·'시편 118­119')과 쪽번호 — 쪽 맨 위 두 줄 안에서만 뗀다
_HEADER_RE = re.compile(r"^[가-힣]{1,8}(?:\s[가-힣]{1,8})?\s\d{1,3}(?:[⋅·­\-–]\d{1,3})?$")
# 책 해설의 항목 이름('목적:'·'핵심 장:') — 앞 줄 끝 공백이 없어도 띄워 잇는다('…하는 것목적:' 방지)
_LABEL_RE = re.compile(r"^[가-힣]{2,5}(?:\s[가-힣]{1,5})?:")
_PAGENO_RE = re.compile(r"^\d{1,4}$")
_SENT_END_RE = re.compile(r"[.?!][”’\"')」』]*$")
_PUNCT_END_RE = re.compile(r"[.,?!;:)”’\"」』]$")
_SHY_RANGE_RE = re.compile(r"(?<=\d)­(?=\d)")   # '1:18­25' — 범위 표시가 소프트 하이픈으로 들어 있다
# 절 끝 이음말('…? / 하시니', '…낳았으며', '사랑하사') — 소제목은 명사로 끝난다('라헬의 죽음'·'형제자매'·'경고')
_CONNECTIVE_END_RE = re.compile(r"(?:(?<!자)매|니|거늘|므로|더니|데|으나|며|되|즉|[아어하보오]사|(?<![회경])고)$")
_FULL_RATIO = 0.8   # 단 너비의 이만큼 이상 찬 줄만 다음 줄로 이어진다고 본다

# 'verse' 는 이어 쓰기 가능, 'text'(소제목·해설) 도 가능, 'chapter' 는 닫힌 단위
_CHAPTER, _VERSE, _TEXT, _XREF = "chapter", "verse", "text", "xref"


def is_verse_book(pages: list[str]) -> bool:
    """장 머리가 3개 이상이고, 장 머리 뒤 첫 번호 줄이 대부분 '1'로 다시 시작하면 장절 책.
    번호 항목 명언집('제1장' 뒤에도 0052·0053 으로 이어짐)과 가른다."""
    chapters = resets = 0
    after_chapter = False
    for p in pages:
        for line in p.splitlines():
            s = line.strip()
            if _CHAPTER_RE.match(s):
                chapters += 1
                after_chapter = True
            elif after_chapter and (m := _VERSE_RE.match(s)):
                resets += m.group(1) == "1"
                after_chapter = False
    return chapters >= 3 and resets >= 0.8 * chapters


def _column_width(pages: list[str]) -> int:
    """절 첫 줄 길이의 75% 지점 — 절 첫 줄은 대개 다음 줄로 넘어가므로 단 너비를 대표한다."""
    lens = sorted(len(l.strip()) for p in pages for l in p.splitlines() if _VERSE_RE.match(l.strip()))
    return lens[len(lens) * 3 // 4] if lens else 0


def _strip_page_edge(lines: list[str]) -> tuple[list[str], int]:
    """쪽 맨 위 두 줄 안의 머리글·쪽번호를 뗀다."""
    k = 0
    while k < min(2, len(lines)) and (_PAGENO_RE.match(lines[k].strip()) or _HEADER_RE.match(lines[k].strip())):
        k += 1
    return lines[k:], k


def format_book(pages: list[str]) -> tuple[str, dict]:
    """원문 쪽들 → 장·소제목·관주·절이 한 줄씩 선 본문, 통계(장·절·뗀 머리글 수)."""
    width = _column_width(pages)
    units: list[list[str]] = []      # [종류, 본문]
    expected: int | None = None     # 다음 절 번호
    prev = ""                       # 직전 원문 줄(줄 끝 공백 포함)
    dropped = 0
    flat: list[tuple[str, int]] = []   # (원문 줄, 그 쪽에서 가장 긴 줄 길이)
    for p in pages:
        lines, k = _strip_page_edge([l for l in p.splitlines() if l.strip()])
        dropped += k
        # 해설·소제목은 한 단 짜리 넓은 줄일 수 있어 쪽에서 가장 긴 줄을 기준으로 — 그래야 책 표제
        # ('기원이라 하는 모세의 첫째 책' / '창 세 기')가 해설 줄처럼 이어 붙지 않는다
        page_width = max((len(l.strip()) for l in lines), default=0)
        flat.extend((l, page_width) for l in lines)
    for i, (line, page_width) in enumerate(flat):
        s = line.strip()
        kind = units[-1][0] if units else None
        m = _VERSE_RE.match(s)
        nxt, nxt2 = (flat[j][0].strip() if j < len(flat) else "" for j in (i + 1, i + 2))
        if c := _CHAPTER_RE.match(s):
            units.append([_CHAPTER, f"제 {c.group(1)} {c.group(2)}"])
            expected = 1
        elif kind in (_CHAPTER, _TEXT, _XREF) and _XREF_RE.match(s):
            units.append([_XREF, s])
        elif m and expected is not None and (expected <= int(m.group(1)) <= expected + 1
                                             or (m.group(1) == "1" and kind != _VERSE)):
            # 장 머리 없는 한 장짜리 책(오바댜·빌레몬 등)은 해설·소제목 뒤 '1'로 시작한다.
            # 한 절을 못 알아봐도 다음 절에서 순서를 되찾는다(n+1 허용)
            units.append([_VERSE, s])
            expected = int(m.group(1)) + 1
        elif kind == _VERSE and units[-1][1].isdigit():
            units[-1][1] += " " + s                 # 번호만 있던 절 머리의 본문
        elif kind == _XREF and not units[-1][1].endswith(")"):
            units[-1][1] += s                       # 넘어간 관주 줄(닫는 괄호가 빠진 원문은 절에서 멈춘다)
        elif kind == _VERSE and _is_heading(prev, s, nxt, nxt2, expected):
            units.append([_TEXT, s])                # 문장으로 끝난 절 뒤, 다음 절 바로 앞의 소제목
        elif kind in (_VERSE, _TEXT) and _continues(prev, width if kind == _VERSE else max(width, page_width)):
            spaced = prev[-1:].isspace() or _PUNCT_END_RE.search(prev.rstrip()) or _LABEL_RE.match(s)
            units[-1][1] += (" " if spaced else "") + s
        else:
            units.append([_TEXT, s])                # 소제목·책 해설 단락
        prev = line
    return _render(units), {
        "chapters": sum(u[0] == _CHAPTER for u in units),
        "verses": sum(u[0] == _VERSE for u in units),
        "headers": dropped,
    }


def _heading_like(s: str) -> bool:
    """문장 부호·이음말 없이 끝나는 줄. 관주를 붙인 소제목('순금 등잔대(출37:17­24)')은 ')'로 끝나도 된다."""
    return ((not _PUNCT_END_RE.search(s) or bool(re.search(r"\([^()]*\d+:\d+[^()]*\)$", s)))
            and not _CONNECTIVE_END_RE.search(s))


def _is_heading(prev: str, s: str, nxt: str, nxt2: str, expected: int | None) -> bool:
    """앞 줄이 문장으로 끝났고(줄 끝 공백이 있어도), 이 줄이 소제목 꼴이며, 다음 줄(두 줄짜리 소제목이면
    그다음 줄)이 바로 다음 절·관주면 소제목. '…하였더라. ' 뒤 '라헬의 죽음' 같은 소제목 444개가 앞 절에
    붙었다(글꼴 크기 정답지 3,038개 대조)."""
    def opens(line: str) -> bool:
        m = _VERSE_RE.match(line)
        return bool(_XREF_RE.match(line)) or (m is not None and int(m.group(1)) == expected)
    return (bool(_SENT_END_RE.search(prev.strip())) and _heading_like(s)
            and (opens(nxt) or (_heading_like(nxt) and not _VERSE_RE.match(nxt) and opens(nxt2))))


def _continues(prev: str, width: int) -> bool:
    """직전 줄이 단 끝까지 찼고, 줄 끝 공백 없이 문장으로 끝난 '단위의 마지막 줄'이 아니면 다음 줄이 이어진다."""
    s = prev.strip()
    if len(s) < width * _FULL_RATIO:
        return False
    return not (_SENT_END_RE.search(s) and not prev[-1:].isspace())


def _render(units: list[list[str]]) -> str:
    out: list[str] = []
    for i, (kind, text) in enumerate(units):
        before = units[i - 1][0] if i else None
        # 장 머리 앞, 그리고 절 뒤에 오는 소제목·해설 앞은 빈 줄
        if out and (kind == _CHAPTER or (kind == _TEXT and before == _VERSE)):
            out.append("")
        out.append(_SHY_RANGE_RE.sub("-", re.sub(r"[ \t]{2,}", " ", text)).strip())
    return "\n".join(out)
