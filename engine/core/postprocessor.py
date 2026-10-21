"""
후처리 — OCR 엔진별 전체 처리 체인

처리 순서 (엔진별 분기):
  ┌─ RapidOCR ──────────────────────────────────────────────────┐
  │  L1 → 쪽번호·마진노이즈제거 → 단락복원 → 공백정규화         │
  └─────────────────────────────────────────────────────────────┘
  ┌─ Vision (Gemini·Claude·Ollama) ─────────────────────────────┐
  │  L1(char_normalizer) → 단락복원 → 공백정규화                 │
  │  (Vision은 이미 깨끗한 텍스트 → 마진제거 생략)               │
  └─────────────────────────────────────────────────────────────┘
  공통: L2(spellcheck) · L3(llm_corrector) 는 _finalize()에서 적용
"""
from __future__ import annotations
import re
from functools import lru_cache
from pathlib import Path

from core.figure_detector import Region, build_placeholder

# 줄 끝 글자+다음 줄 첫 글자가 정제본에서 주로 붙어 쓰이는 한글 쌍 — tools/build_wrap_spacing.py 로 만든다.
# 책 97권 실측: 글자 경계에서 늘 띄우면 30.1%, 이 표로 정하면 95.1% 맞았다.
_WRAP_TABLE = Path(__file__).parent / "data" / "wrap_nospace.txt"
# 쪽 머리·끝의 쪽 번호 줄 ('30', '- 30 -')
_PAGE_NUM_LINE = re.compile(r'^[\s\-–—]*\d{1,3}[\s\-–—]*$')
# 쪽 꼬리 '장·부 제목+쪽번호' ('4장왜지능이라고 하는가?55', '13 1부 예측') — 쪽 마지막 줄이면 앞 줄에 잇지 않는다.
# 따로 남아야 문서 단위 머리글 제거(text_cleaner.strip_running_headers)가 모양으로 알아보고 지운다(예측기계 141곳)
# 짝수 쪽 꼬리는 OCR 이 쪽번호와 부 번호를 붙여 읽는다('1582부의사결정' = 158쪽 2부) → 사이 공백은 없어도 된다
_PAGE_FOOTER_LINE = re.compile(r'^(?:\d{1,3}\s*)?\d{1,2}\s?[장부]\s?\S.{0,35}?\s?\d{1,3}$|^\d{1,3}\s*\d{1,2}\s?[장부]\s?\S.{0,35}$')

# 단락 끝 판단 (이 문자로 끝나면 새 단락 시작 가능)
_SENT_END = re.compile(r'[.!?。！？…\"\'\)\]》」』]\s*$')
# 단락 시작 판단 (들여쓰기 없이 시작하는 한글·영문)
_PARA_START = re.compile(r'^[가-힣A-Za-z0-9\"\'\(「『《]')
# 줄이 오른쪽 끝까지 찼다고 볼 비율 — 쪽에서 흔한 줄 길이(중앙값)의 이만큼 이상이어야 다음 줄과 잇는다.
# MyOCR 정제본 23권 실측: 길이만 보면 제목('목표는 완전한 이해다')·표 행('국내 극장 4,928,000')을 본문에 붙였다
_FULL_LINE_RATIO = 0.75
# 항목 시작(글머리 기호·일련번호·조항) — 이 줄은 앞 줄에 잇지 않는다. 쪽 경계 잇기·OCR 문단 잇기도 같이 쓴다.
# '1.5배'·'가나다'·'제3조에' 처럼 뒤에 공백·괄호가 없으면 항목 표시가 아니다.
ITEM_START_RE = re.compile(
    r'^(?:[-–—•·*▶▷►>»※□■○●◆◇▪◦☞➤✓]'
    r'|\(?\d{1,3}[.．)]\s'                          # 1. 2) (3)
    r'|[①-⑳㉠-㉭㈀-㈍]'                              # ① ㉠ ㈀
    r'|제\s?\d{1,3}\s?[조장절항편부관](?:\s|\(|$))')   # 제1조 제2장
# 한글 항목(가. 나) (다)) — 줄 끝에서 잘린 '…운용되었|다. 그러나' 와 모양이 같아,
# 앞이나 뒤 글자도 다른 줄 머리에 항목으로 나올 때만 항목으로 본다(목록은 두 개 이상이다)
_KO_ITEM_RE = re.compile(r'^\(?([가나다라마바사아자차카타파하])[.)]\s')
_KO_SEQ = "가나다라마바사아자차카타파하"


def item_letters(lines) -> frozenset[str]:
    """줄 머리에 한글 항목 표시로 나온 글자들 — is_item_start 에 넘긴다."""
    return frozenset(m[1] for l in lines if (m := _KO_ITEM_RE.match(l.lstrip())))


def is_item_start(line: str, letters: frozenset[str] = frozenset()) -> bool:
    """항목 시작 줄인가(글머리 기호·일련번호·조항·한글 항목)."""
    s = line.lstrip()
    if ITEM_START_RE.match(s):
        return True
    m = _KO_ITEM_RE.match(s)
    if not m:
        return False
    k = _KO_SEQ.index(m[1])
    return bool(letters & {_KO_SEQ[k - 1] if k else "", _KO_SEQ[k + 1] if k + 1 < len(_KO_SEQ) else ""})
# 노이즈 줄 (2자 미만 의미 있는 문자)
_NOISE_LINE = re.compile(r'^[\s\W]{0,2}$')
# 구조 마커 패턴 (미리보기에서 제거할 것들)
_STRUCTURAL = [
    re.compile(r'^\[페이지\s*\d+\]'),           # [페이지 N]
    re.compile(r'^[─\-=]{3,}'),                  # 구분선
    re.compile(r'^\[그림\s*\d+[^\]]*\]'),         # [그림 N: ...]
    re.compile(r'^\[표\s*\d+[^\]]*\]'),           # [표 N...]
    re.compile(r'^\[수식\s*\d+[^\]]*\]'),         # [수식 N...]
    re.compile(r'^⟨TABLE:\d+-\d+⟩'),              # 본문 표 자리 표지
    re.compile(r'^[─]+\s*비텍스트\s*영역'),        # ── 비텍스트 영역 ──
    re.compile(r'^\s*\|'),                        # Markdown 표 행
    re.compile(r'^\s*\|?\s*:?-{3,}:?\s*\|?'),    # Markdown 구분자
    re.compile(r'^[★⚠]'),                         # 검토 경고 헤더
    re.compile(r'python ocr_tool\.py'),           # 설정 안내
    re.compile(r'^>\s*\*\*'),                     # MD 인용 헤더
    re.compile(r'^>\s*사유:'),                     # MD 인용 사유
    re.compile(r'^>\s*이\s*페이지'),               # MD 인용 안내
]


# ── 공개 API ──────────────────────────────────────────────────────────────────

def apply(text: str, regions: list[Region] | None = None) -> str:
    """OCR 텍스트 후처리 전체 파이프라인 (파일 출력용)."""
    if regions:
        text = _insert_placeholders(text, regions)
    text = _normalize_whitespace(text)
    text = _strip_line_ends(text)
    text = remove_noise_lines(text)
    text = _normalize_whitespace(text)
    return text.strip()


def process_full(text: str, engine: str = "rapidocr") -> str:
    """
    OCR 결과 전체 처리 체인 — 저장 파일·미리보기 공통 사용.

    engine='rapidocr':
      쪽번호·마진노이즈제거(extract_korean_text) → 단락복원 → 공백정규화
    Vision 엔진('gemini'·'claude_vision'·'ollama')·'pdf_text':
      단락복원 → 공백정규화  (이미 깨끗한 텍스트)

    L1(char_normalizer)·L2(spellcheck)·L3(llm_corrector)는
    _finalize()에서 이미 적용되므로 여기서는 호출하지 않음.
    """
    if not text.strip():
        return text

    if engine == "rapidocr":
        # 로컬 OCR: 쪽 번호·마진 숫자·스캔 잡음·노이즈 줄 제거
        text = _strip_page_number_lines(text)
        text = extract_korean_text(text)

    # 공통: 단락 복원 + 공백 정규화
    text = join_broken_lines(text)
    if engine == "rapidocr":
        text = re.sub(r'^[ \t]+', '', text, flags=re.M)   # 문단 첫 줄 표시는 줄 잇기에서 다 썼다
    text = _fix_spacing(text)
    text = remove_noise_lines(text)
    text = _normalize_whitespace(text)
    return text.strip()


def for_preview(text: str) -> str:
    """
    미리보기 표시용 최종 정제.
    process_full() 이후 결과에서 구조 마커만 추가로 제거.
    """
    return strip_structural(text).strip()


def extract_korean_text(text: str) -> str:
    """
    스캔 도서 특화 '마진 노이즈' 제거 (보수적 버전).

    [변경 이유] 이전 버전은 한글 8자 미만 줄을 통째로 삭제하고 첫 한글
    이전 내용을 모두 버려서, 영어·표·숫자·제목·짧은 문장을 잃어버렸다.
    현재 버전은 줄을 함부로 삭제하지 않고, 줄 앞쪽의 명백한 스캔 마진
    (페이지번호·기호로만 이뤄진 토큰이 2개 이상 연속)만 제거하며 본문은
    최대한 보존한다. 한글·영문·숫자가 하나도 없는 순수 기호 줄만 버린다.
    """
    lines = text.splitlines()
    cleaned: list[str] = []

    for line in lines:
        if not line.strip():
            cleaned.append('')
            continue

        proc = _strip_margin_noise(line)

        # 한글·영문·숫자가 하나도 없는 순수 기호 줄만 제거 (본문·데이터는 보존)
        if re.sub(r'[^\w가-힣]', '', proc) == '':
            continue

        # 문단 첫 줄 표시(rapidocr_engine._mark_paragraph_starts 의 공백 2칸)는 남긴다 — join_broken_lines 가 본다
        cleaned.append('  ' + proc if line.startswith('  ') else proc)

    return _normalize_whitespace('\n'.join(cleaned))


def _strip_margin_noise(line: str) -> str:
    """
    줄 앞쪽의 스캔 마진 노이즈(페이지번호·기호 토큰)만 제거하고 본문은 보존.

    - 선두에 '의미문자(한글/영문)가 없는' 토큰이 2개 이상 연속될 때만 제거.
      예: "00 10: *인간은..." → "인간은...",  "| 0 00000 진정으로..." → "진정으로..."
    - '2026년', '1,250만원'처럼 숫자가 한글에 붙은 토큰은 보존(노이즈 아님).
    - 'Google test ...'처럼 영문 단어로 시작하는 줄, 선두 숫자 1개로 시작하는
      줄(목록·데이터)도 보존.
    """
    tokens = line.split(' ')

    def _is_noise(tok: str) -> bool:
        t = tok.strip()
        if not t:
            return True
        # 한글·영문자·한자가 없는 토큰(숫자·기호만) = 마진 노이즈 후보
        # 한자도 글자다 — 빼면 한문 줄('夫未戰而廟算勝者, 得算多也.')이 통째로 지워진다(2026-10-18)
        return re.search(r'[가-힣A-Za-z一-鿿]', t) is None

    n = 0
    for tok in tokens:
        if _is_noise(tok):
            n += 1
        else:
            break

    # 숫자 사이를 기호로 이은 표기('6 · 25 전쟁'·'3 · 1 운동'·'2026. 10. 19.')는 본문 — 마진 잡음
    # ('00 10: *…'·'| 0 00000 …')은 숫자 사이에 잇는 기호가 없다
    if n >= 2 and re.fullmatch(r"\d{1,4}(?:\s*[·.:~\-+×=/]\s*\d{1,4})+\.?", " ".join(tokens[:n]).strip()):
        n = 0
    if n >= 2:                       # 선두 노이즈 토큰 2개 이상일 때만 제거
        tokens = tokens[n:]
    proc = ' '.join(tokens).strip()

    # 본문 첫 글자에 붙은 선두 잡음 기호 제거 (따옴표·괄호는 보존)
    proc = re.sub(r'^[\s*|~^#@、。]+', '', proc)
    # 줄 뒤 스캔 잡음 제거 (정상 구두점 . , ! ? : 는 보존)
    proc = re.sub(r'[\sᆞ~_|@#^*、。]+$', '', proc)
    # 줄 내부 가운뎃점(ᆞ) 클러스터 → 공백
    proc = re.sub(r'\s*ᆞ+\s*', ' ', proc)
    # 반복 공백 정리
    proc = re.sub(r'  +', ' ', proc).strip()

    return proc


def strip_structural(text: str) -> str:
    """구조 마커·표·구분선 제거 → 순수 텍스트만 반환."""
    lines  = text.splitlines()
    result = []
    for line in lines:
        if any(pat.search(line) for pat in _STRUCTURAL):
            continue
        result.append(line)

    # 연속 빈 줄 정리
    cleaned = re.sub(r'\n{3,}', '\n\n', '\n'.join(result))
    return cleaned.strip()


def remove_noise_lines(text: str) -> str:
    """의미 없는 짧은 줄(노이즈) 제거."""
    lines = text.splitlines()
    result = []
    for line in lines:
        # 빈 줄은 유지
        if not line.strip():
            result.append(line)
            continue
        # 의미 있는 문자(한글·영문·숫자)가 2자 이상이면 유지
        meaningful = re.sub(r'[^\w가-힣]', '', line)
        # '즉,'·'단,' 처럼 한 글자 말 + 문장부호인 줄은 내용이라 남긴다 (MyOCR 정제본 실측)
        if len(meaningful) >= 2 or re.fullmatch(r'[가-힣][,.:]', line.strip()):
            result.append(line)
    return '\n'.join(result)


_MD_HEADING_RE = re.compile(r'^#{1,6} \S')
# 그림 제목 줄 '◆ 그림 0.1 이익 계획 매트릭스'·'그림5.5경쟁력 인지 지도' — 글머리 ◆ 는 OCR 이 빠뜨리기도 한다.
# 줄 첫머리 본문 참조('그림 5.6의 구매 가능성은…'·'그림 5.4).')는 번호에 조사·괄호가 붙어 제외
_FIGURE_CAPTION_RE = re.compile(
    r'^[◆❖◇◈♦]?\s*그림\s*\d+\s*[.,]\s*\d+(?:\s+[^\s)]|(?![의에은는을를과와로도이가])[^\s\d).,])')


def join_broken_lines(text: str) -> str:
    """
    단락 내 줄바꿈 복원.
    OCR은 페이지의 물리적 줄을 그대로 추출하므로
    같은 단락에 속하는 줄들을 공백으로 이어 붙임.

    연결 조건:
      - 현재 줄이 문장 끝 구두점으로 종료되지 않음
      - 현재 줄 길이가 10자 이상 (짧은 제목/레이블 제외)
      - 다음 줄이 들여쓰기 없이 한글·영문으로 시작
      - 다음 줄이 새 단락 표시(빈 줄, 숫자. 로 시작)가 아님
      - 현재 물리적 줄이 오른쪽 끝까지 참(흔한 줄 길이의 75% 이상) — 짧은 제목·표 행 제외
      - 다음 줄이 항목 시작(- • ▶, 1. 2), ①, 가., 제1조 등)이 아님
    """
    lines = text.splitlines()
    lens = sorted(len(l.strip()) for l in lines if l.strip())
    full = lens[len(lens) // 2] * _FULL_LINE_RATIO if lens else 0
    letters = item_letters(lines)
    out: list[str] = []
    i = 0
    while i < len(lines):
        cur = lines[i]
        # 빈 줄 → 단락 구분, 그대로 추가
        if not cur.strip():
            out.append(cur)
            i += 1
            continue

        # 다음 줄과 연결 가능 여부 반복 판단
        while i + 1 < len(lines):
            nxt = lines[i + 1]
            if not nxt.strip():
                break  # 빈 줄 → 연결 중단
            if _MD_HEADING_RE.match(cur) or _MD_HEADING_RE.match(nxt):
                break  # 장·절 제목 줄(toc_headings)은 본문에 붙이지 않는다
            if _FIGURE_CAPTION_RE.match(nxt.strip()):
                break  # 그림 제목 줄 — 마침표 없이 끊긴 앞 문단에 붙었다('…욕심 그림0.1 이익 계획 매트릭스')
            # 지금까지 이은 줄이 아니라 '현재 물리적 줄'(lines[i])이 끝까지 찼는지를 본다
            last = not any(l.strip() for l in lines[i + 2:])
            if (len(lines[i].strip()) >= full and not is_item_start(nxt, letters)
                    and not (last and _PAGE_FOOTER_LINE.match(nxt.strip()))
                    and _should_join(cur, nxt)):
                cur = _join_wrapped(cur, nxt)
                i += 1
            else:
                break

        out.append(cur)
        i += 1

    return '\n'.join(out)


# ── 내부 함수 ──────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _wrap_nospace() -> frozenset[str]:
    """붙여 잇는 한글 쌍 표. 파일이 없으면 빈 표(예전처럼 늘 띄운다)."""
    try:
        return frozenset(_WRAP_TABLE.read_text(encoding="utf-8").split())
    except OSError:
        return frozenset()


def _join_wrapped(cur: str, nxt: str) -> str:
    """줄바꿈으로 갈라진 두 줄을 잇는다 — 낱말 중간에서 잘린 자리('운용되었|다')면 붙이고, 아니면 띄운다."""
    cur, nxt = cur.rstrip(), nxt.lstrip()
    if cur and nxt and (cur[-1] + nxt[0]) in _wrap_nospace():
        return cur + nxt
    return cur + ' ' + nxt


def _strip_page_number_lines(text: str) -> str:
    """쪽 맨 앞·맨 끝의 쪽 번호 줄을 지운다 — 그대로 두면 본문 줄에 붙는다('어떤 환경데 30')."""
    lines = text.splitlines()
    while lines and (not lines[0].strip() or _PAGE_NUM_LINE.match(lines[0])):
        lines.pop(0)
    while lines and (not lines[-1].strip() or _PAGE_NUM_LINE.match(lines[-1])):
        lines.pop()
    return '\n'.join(lines)

def _should_join(cur: str, nxt: str) -> bool:
    """두 줄을 이어 붙여야 하면 True."""
    cur_strip = cur.strip()
    nxt_strip = nxt.strip()

    if not cur_strip or not nxt_strip:
        return False
    # 현재 줄이 너무 짧으면 제목/레이블일 가능성 → 연결 안 함
    if len(cur_strip) < 10:
        return False
    # 현재 줄이 문장 끝 구두점으로 종료 → 연결 안 함
    if _SENT_END.search(cur_strip):
        return False
    # 다음 줄이 숫자+점으로 시작 (목록 항목) → 연결 안 함
    if re.match(r'^\d+[.．]\s', nxt_strip):
        return False
    # 다음 줄이 들여쓰기 있음 → 새 단락
    if nxt.startswith(('  ', '\t', '　')):
        return False
    # 다음 줄이 연속 텍스트로 시작
    return bool(_PARA_START.match(nxt_strip))


def _fix_spacing(text: str) -> str:
    """구두점 앞뒤 공백 정규화.
    예전엔 구두점 뒤 영문까지 띄워 'www.naver.com'·'C.의' 를 망가뜨렸고, 구두점 앞 공백 제거가
    줄바꿈까지 먹었다 → 한글 앞만 띄우고(영문 머리글자 뒤 마침표 제외), 공백·탭만 지운다 (MyOCR 이식)."""
    # 구두점 앞 공백 제거 (뒤가 공백·줄끝·한글일 때만 — '.NET'·' ...' 보존)
    text = re.sub(r'(?<=\S)[ \t]+([.,!?。！？、])(?=\s|$|[가-힣])', r'\1', text, flags=re.M)
    # 구두점 뒤 한글 앞 공백 확보 (이미 있으면 유지)
    text = re.sub(r'(?<![A-Za-z])([.。])([가-힣])|([,!?！？])([가-힣])',
                  lambda m: f"{m.group(1) or m.group(3)} {m.group(2) or m.group(4)}", text)
    # 괄호 앞뒤 공백
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    return text


def _insert_placeholders(text: str, regions: list[Region]) -> str:
    placeholders = [build_placeholder(r) for r in regions]
    if not placeholders:
        return text
    sep = '\n\n' + '─' * 40 + '\n'
    return text.rstrip() + sep + '\n\n'.join(placeholders) + '\n'


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.replace('\t', '    ')
    text = re.sub(r'(?<=\S) {2,}', ' ', text)
    return text


def _strip_line_ends(text: str) -> str:
    return '\n'.join(line.rstrip() for line in text.splitlines())
