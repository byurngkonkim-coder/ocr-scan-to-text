"""디지털 PDF 텍스트 레이어 직접 추출 — OCR·API 없이 내장 텍스트를 읽는다.

born-digital PDF(워드·편집 프로그램에서 만든 PDF)는 페이지 안에 '진짜 글자'가
들어 있어, 이미지 렌더링 → OCR → (정제) 과정 없이 그 텍스트 레이어를 바로 읽을 수
있다. 스캔 PDF(종이를 이미지로 찍은 것)는 텍스트 레이어가 없으므로 is_digital=False
로 판별되고, 호출부는 기존 OCR 경로로 자동 전환한다.

엔진 우선순위
─────────────
1) PyMuPDF(fitz)      — 최상의 읽기순서·어절 띄어쓰기 보존(다단 편집에 강함). 기본.
2) poppler pdftotext  — fitz 미설치 시 폴백(이 도구는 이미 poppler(pdf2image)에 의존).
둘 다 없으면 is_available()=False → 디지털 경로 비활성(무해, 기존 OCR 사용).

판별 휴리스틱(looks_digital)
────────────────────────────
- 페이지의 공백 제외 글자 수 ≥ MIN_PAGE_CHARS 이면 '텍스트 페이지'.
- (텍스트 페이지 / 전체) ≥ MIN_TEXT_PAGE_RATIO 이고 전체 글자 수 ≥ MIN_DOC_CHARS
  이면 문서를 '디지털'로 간주한다. 스캔 PDF는 텍스트가 거의 없어 자동으로 False.
- 스캐너가 넣은 OCR 텍스트 레이어(검색 가능 스캔본)가 있으면 그대로 활용(무료).
- 이미지가 대부분이고 텍스트가 소수 페이지뿐이면(비율 미달) 스캔으로 간주 → OCR.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

# 페이지가 '텍스트 페이지'로 인정받는 최소 공백 제외 글자 수.
# 쪽번호(예: '132')만 있는 페이지가 오탐되지 않도록 여유를 둔다.
MIN_PAGE_CHARS = 15
# 문서를 '디지털'로 판정하는 (텍스트 페이지 / 전체) 비율 하한.
MIN_TEXT_PAGE_RATIO = 0.5
# 문서를 '디지털'로 판정하는 전체 글자 수 하한(스트레이 문자 몇 개로 오탐 방지).
MIN_DOC_CHARS = 60

_WS_RE = re.compile(r"\s+")

# ── 내장 글자 품질 (2026-10-05) ────────────────────────────────────────────────
# '검색 가능 스캔본'(스캐너가 넣은 OCR 글자)은 글자 수만 보면 디지털로 판정되지만,
# 그 글자가 깨져 있으면('길- 푿는 그대에게… 워制 우리는 자신용') 정제로도 복구할 수 없다.
# 깨진 OCR의 두 신호: ① 흔히 쓰는 한글 2,350자(EUC-KR) 밖의 드문 음절('푿·쩟·삔'),
# ② 한글 획을 한자로 잘못 읽어 한글에 붙은 한자('워制·乂·卜').
# 실측(01_입력 22권): 정상 책은 쪽 점수 중앙값 0~0.3%, 깨진 책(길을묻는그대에게)은 2.2%.
# 점수 3% 이상인 쪽은 표본 전부가 실제로 깨진 쪽이었다(3.2% 쪽 포함).
GARBLED_PAGE_SCORE = 0.03   # 이 이상이면 그 쪽은 Vision으로 다시 읽는다
GARBLED_DOC_RATIO = 0.10    # 깨진 쪽이 이 비율 이상이면 문서 전체를 스캔본으로 본다
_GARBLE_MIN_HANGUL = 30     # 한글이 이보다 적은 쪽(영문·숫자 위주)은 판정하지 않는다
_HANGUL_HANJA_RE = re.compile(r"[가-힣][一-鿿]|[一-鿿][가-힣]")


def _rare_syllable(ch: str) -> bool:
    try:
        ch.encode("euc-kr")
        return False
    except UnicodeEncodeError:
        return True


def garble_score(text: str) -> float | None:
    """(드문 음절 수 + 한글·한자 붙음 수) / 한글 수. 한글이 적으면 None."""
    hangul = re.findall(r"[가-힣]", text or "")
    if len(hangul) < _GARBLE_MIN_HANGUL:
        return None
    bad = sum(map(_rare_syllable, hangul)) + len(_HANGUL_HANJA_RE.findall(text))
    return bad / len(hangul)


def is_garbled(text: str) -> bool:
    s = garble_score(text)
    return s is not None and s >= GARBLED_PAGE_SCORE


# 글자는 맞지만 잡음 기호·띄어쓰기 문제가 있는 쪽 — 이미지 재인식까지는 필요 없고
# 문장 정제(LLM)로 고쳐지는 유형('잘되는놈도 < 언젠가는자신의 I 본성'). 실측: 악당의명언
# 183/296쪽, 나머지 책 0~95쪽. 정제를 끈 채 처리하면 그대로 저장되므로 알리는 데 쓴다.
NOISY_SYMBOL_RATIO = 0.02
NOISY_NOSPACE_RATIO = 0.15  # 8음절 이상 붙은 한글 덩어리가 한글에서 차지하는 비율
_NOISE_RE = re.compile(r"[■_^|•♦£«»、¤¦◆▲▼●○□◇†‡]")


def _nospace_ratio(text: str) -> float | None:
    hangul = re.findall(r"[가-힣]", text or "")
    if len(hangul) < _GARBLE_MIN_HANGUL:
        return None
    return sum(len(r) for r in re.findall(r"[가-힣]{8,}", text)) / len(hangul)


def is_unspaced(text: str) -> bool:
    """한글 어절이 띄어쓰기 없이 길게 붙은 쪽인가."""
    r = _nospace_ratio(text)
    return r is not None and r >= NOISY_NOSPACE_RATIO


def is_noisy(text: str) -> bool:
    if _nospace_ratio(text) is None:
        return False
    noise = len(_NOISE_RE.findall(text)) / real_chars(text)
    return noise >= NOISY_SYMBOL_RATIO or is_unspaced(text)


# ── 띄어쓰기를 잃은 텍스트층 (2026-10-15) ──────────────────────────────────────
# 실측: SERI 「교토식 경영」 PDF(23쪽 중 22쪽)는 글자를 한 자씩 찍고 어절 사이를 간격으로만 띄워,
# 텍스트층에 공백 글자가 없다('스피드경영을중시하는호리바제작소'). 글자 위치는 정확해서 간격을 재면
# 글자 사이 ≈0, 어절 사이 ≈0.5×글자 크기로 뚜렷이 갈린다 → 간격이 넓은 곳에 공백을 넣으면 원문 그대로.
# 정상 디지털 PDF 4종(이나모리 회계학·Kyocera 논문·시간당채산·McKinsey)은 어절 평균 2.7~3.7음절로
# 감지 기준(8음절 이상 덩어리 15%)에 걸리지 않는다. 이 PDF는 poppler 렌더가 빈 화면이라 OCR 재판독도 못 한다.
_RESPACE_GAP = 0.25   # 글자 크기 대비 이 이상 벌어지면 어절 사이로 본다
_LETTER_SPACED = 0.6  # 띄운 줄에서 한 음절 어절이 이 비율 이상이면 자간 넓힌 제목으로 보고 띄우지 않는다


def respace_rawdict(raw: dict, keep_block=None) -> str:
    """page.get_text('rawdict') 결과를 글자 간격으로 공백을 넣어 다시 조립한다.
    keep_block(block) 이 False 인 블록은 뺀다(머리글·바닥글 필터)."""
    out = []
    for b in raw["blocks"]:
        if b.get("type") != 0 or (keep_block and not keep_block(b)):
            continue
        lines = []
        for ln in b["lines"]:
            s, prev = [], None
            for sp in ln["spans"]:
                for c in sp["chars"]:
                    if (prev is not None and not c["c"].isspace() and not prev["c"].isspace()
                            and c["bbox"][0] - prev["bbox"][2] > _RESPACE_GAP * sp["size"]):
                        s.append(" ")
                    s.append(c["c"])
                    prev = c
            line = "".join(s)
            # 자간을 넓힌 제목('지 도 교 수')은 글자마다 벌어져 있다 — 한 음절 어절이 대부분이면 띄우지 않는다
            words = line.split()
            if len(words) >= 3 and sum(bool(re.fullmatch(r"[가-힣]", w)) for w in words) / len(words) >= _LETTER_SPACED:
                line = "".join(c["c"] for sp in ln["spans"] for c in sp["chars"])
            lines.append(line)
        out.append("\n".join(lines))
    return "\n".join(out) + "\n"


def real_chars(text: str) -> int:
    """공백(스페이스·탭·개행)을 제외한 실제 글자 수."""
    if not text:
        return 0
    return len(_WS_RE.sub("", text))


# ── 글자 대응표가 깨진 글꼴 (2026-10-10) ───────────────────────────────────────
# 실측: DBR 잡지 PDF 의 표·제목·캡션 글꼴(DongA3G)이 한글 일부를 구르무키·말라얄람·아랍 문자 등으로,
# 영문을 밀린 글자('TVA'→'57A')로 내보냈다('전통੶ 회계'·'포인트\x10ഊఔ'). 깨진 곳이 쪽마다 몇 줄뿐이라
# 쪽 점수(위 garble_score)로는 안 잡히고, 쪽 전체를 다시 읽으면 멀쩡한 본문까지 OCR 오류를 떠안는다.
# → 그런 글자가 든 줄만 이미지로 다시 읽고, 원문과 글자 정렬해 깨진 구간만 바꾼다.
# 오탐 실측(01_입력 98권 33,682쪽): 걸린 쪽 40 — 38쪽은 같은 DBR 글꼴 문제, 2쪽은 기호 글꼴()·제어문자.
# 키릴 문자는 확장 영역(U+0460~052F, 'ҳ주의'=군주의)만 — 현대 러시아어(U+0400~045F)는 책에 나올 수 있다
_FOREIGN_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1fѠ-ࣿऀ-෿-]")
_ALNUM_RE = re.compile(r"[A-Za-z0-9]")
# (배율, 흰 여백 px) — 같은 줄도 이미지 조건에 따라 OCR 이 다르게 읽어, 둘 다 읽고 원문과 더 맞는 쪽을 쓴다
_REOCR_VARIANTS = ((3, 6), (2, 0))
_REOCR_MIN_AGREEMENT = 0.5    # 원문의 멀쩡한 글자 중 OCR 과 정렬되는 비율이 이보다 낮으면 OCR 을 믿지 않는다


def has_foreign_glyphs(text: str) -> bool:
    """한글 책에 나올 리 없는 문자(제어문자·인도계·아랍 문자 등)가 있으면 True — 깨진 글꼴의 흔적."""
    return bool(_FOREIGN_RE.search(text or ""))


def _trust_chars(s: str) -> str:
    """원문에서 믿을 만한 글자만 — 깨진 문자·영문·숫자(깨진 글꼴에선 밀려 있다)·공백 제외."""
    return _WS_RE.sub("", _FOREIGN_RE.sub("", _ALNUM_RE.sub("", s)))


def _agreement(orig: str, ocr: str) -> float | None:
    """원문의 믿을 만한 글자 중 OCR 과 정렬되는 비율. 그런 글자가 없으면 None."""
    clean = _trust_chars(orig)
    if not clean:
        return None
    import difflib
    sm = difflib.SequenceMatcher(None, orig, ocr, autojunk=False)
    return sum(len(_trust_chars(orig[i:i + n])) for i, _, n in sm.get_matching_blocks()) / len(clean)


def merge_reocr(orig: str, ocr: str) -> tuple[str, bool]:
    """원문 줄과 OCR 결과를 글자 정렬해 깨진 글자·영문·숫자 구간만 OCR 로 바꾼다. (결과, OCR 을 믿었나).
    OCR 을 믿지 못하면 원문에서 깨진 문자만 지운다."""
    import difflib
    glyphs = len(_WS_RE.sub("", orig))
    if len(_trust_chars(orig)) < 4:      # 맞춰 볼 글자가 거의 없는 짧은 줄 — 길이가 그럴듯하면 믿는다
        ok = bool(ocr) and (not glyphs or 0.5 <= len(_WS_RE.sub("", ocr)) / glyphs <= 2)
    else:
        ag = _agreement(orig, ocr)
        ok = ag is not None and ag >= _REOCR_MIN_AGREEMENT
    if not ok:
        return _FOREIGN_RE.sub("", orig), False
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, orig, ocr, autojunk=False).get_opcodes():
        a, b = orig[i1:i2], ocr[j1:j2]
        if op == "equal":
            out.append(a)
        elif _FOREIGN_RE.search(a) or _ALNUM_RE.search(a):
            out.append(b)                    # 깨진 구간 → OCR
        elif op == "insert":                 # OCR 에만 있는 글자 — 깨진 글자 옆일 때만 받는다
            out.append(b if b.strip() and _FOREIGN_RE.search(orig[max(0, i1 - 1):i1 + 1]) else "")
        else:
            out.append(a)                    # 멀쩡한 원문 유지 (OCR 이 빠뜨리거나 다르게 읽은 곳)
    return "".join(out), True


def repair_page_text(text: str, lines, ocr_line) -> tuple[str, int]:
    """쪽 텍스트에서 깨진 줄(lines: [(원문 줄, 위치)])을 ocr_line(위치, 원문) 결과로 고친다. (텍스트, 고친 줄 수).
    끝에 남은 깨진 문자는 지운다."""
    fixed = 0
    # 긴 줄부터 — 짧은 깨진 줄('전통੶ 회계')이 긴 줄('(전통੶ 회계 기준)')의 일부일 때 긴 줄이 먼저 바뀌어야 한다.
    # 앞뒤 공백은 떼고 맞춘다(줄 정보의 들여쓰기 공백이 쪽 텍스트와 다를 수 있다)
    for orig, rect in sorted(lines, key=lambda x: len(x[0].strip()), reverse=True):
        key = orig.strip()
        if not key:
            continue
        new, ok = merge_reocr(key, ocr_line(rect, key))
        if key in text:
            text = text.replace(key, new.strip(), 1)
            fixed += ok
    return _FOREIGN_RE.sub("", text), fixed


def _reocr_line(page, rect, orig: str) -> str:
    """줄 영역을 이미지로 다시 읽는다 — 이미 한 줄이라 글자 위치 찾기(검출) 없이 인식만(줄당 약 0.1초)."""
    import fitz
    import numpy as np
    from PIL import Image, ImageOps
    from core import rapidocr_engine
    engine = rapidocr_engine.get_engine("kor+eng")
    best = None
    for zoom, pad in _REOCR_VARIANTS:
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        if pad:
            img = ImageOps.expand(img, border=pad, fill="white")
        res = engine(np.array(img), use_det=False, use_cls=False, use_rec=True)
        txt = " ".join(str(t) for t in (getattr(res, "txts", None) or [])).strip()
        cand = (_agreement(orig, txt) or 0, len(txt), txt)
        best = cand if best is None or cand > best else best
    return best[2]


def _repair_garbled_font(page, text: str) -> tuple[str, int]:
    """쪽에 깨진 글꼴 흔적이 있으면 그 줄만 다시 읽어 고친다. RapidOCR 이 없으면 그대로 둔다."""
    if not has_foreign_glyphs(text):
        return text, 0
    from core import rapidocr_engine
    if not rapidocr_engine.is_available():
        return text, 0
    lines = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            orig = "".join(s["text"] for s in l["spans"])
            if has_foreign_glyphs(orig):
                lines.append((orig, l["bbox"]))
    return repair_page_text(text, lines, lambda rect, orig: _reocr_line(page, rect, orig))


# ── 백엔드 감지 ────────────────────────────────────────────────────────────────

def _has_fitz() -> bool:
    try:
        import fitz  # noqa: F401  (PyMuPDF)
        return True
    except Exception:
        return False


def _find_pdftotext() -> str | None:
    """poppler pdftotext 실행파일 경로. 프로그램 폴더 안·Windows 기본 설치 경로 우선, 없으면 PATH."""
    from core.pdf_splitter import poppler_dir
    d = poppler_dir()
    if d and (d / "pdftotext.exe").exists():
        return str(d / "pdftotext.exe")
    return shutil.which("pdftotext")


def backend() -> str | None:
    """사용 가능한 추출 백엔드: 'fitz' | 'pdftotext' | None."""
    if _has_fitz():
        return "fitz"
    if _find_pdftotext():
        return "pdftotext"
    return None


def is_available() -> bool:
    """디지털 PDF 직접 추출이 가능한 환경인지 여부."""
    return backend() is not None


# ── 페이지 텍스트 추출 ─────────────────────────────────────────────────────────

def extract_pages(pdf_path: str | Path, repaired: dict[int, int] | None = None) -> list[str]:
    """PDF의 페이지별 텍스트 레이어를 리스트로 추출한다(index 0 = 1페이지).

    repaired 를 넘기면 깨진 글꼴 줄을 이미지로 다시 읽어 고치고, 고친 수를 {쪽 번호: 줄 수}로 채운다
    (fitz 백엔드만). 넘기지 않으면 내장 글자 그대로 — 파일 정보 표시처럼 빨라야 하는 곳용.
    사용 가능한 백엔드가 없으면 RuntimeError.
    """
    b = backend()
    if b == "fitz":
        return _extract_fitz(pdf_path, repaired)
    if b == "pdftotext":
        return _extract_pdftotext(pdf_path)
    raise RuntimeError(
        "PDF 텍스트 레이어 추출 백엔드가 없습니다 (PyMuPDF/poppler 미설치)."
    )


_FITZ_LOCK = threading.Lock()   # PyMuPDF 는 스레드 안전이 아니다 — 여러 파일 동시 처리 시 한 번에 하나만


def _extract_fitz(pdf_path: str | Path, repaired: dict[int, int] | None = None) -> list[str]:
    with _FITZ_LOCK:
        pages = _extract_fitz_unlocked(pdf_path)
        if repaired is not None and any(has_foreign_glyphs(p) for p in pages):
            import fitz
            with fitz.open(str(pdf_path)) as doc:
                for i, p in enumerate(pages):
                    pages[i], n = _repair_garbled_font(doc[i], p)
                    if n:
                        repaired[i + 1] = n
        return pages


def _extract_fitz_unlocked(pdf_path: str | Path) -> list[str]:
    import fitz
    from core import config

    enable_filter = config.get_enable_margin_filter()
    top_ratio = config.get_margin_top_ratio()
    bottom_ratio = config.get_margin_bottom_ratio()

    out: list[str] = []
    with fitz.open(str(pdf_path)) as doc:
        for page in doc:
            if not enable_filter:
                # "text" 모드: 블록 읽기순서 유지 + 어절 공백 보존(다단 편집에 강함).
                text = page.get_text("text") or ""
                if is_unspaced(text):
                    text = _respaced_or(text, respace_rawdict(page.get_text("rawdict")))
                out.append(text)
                continue

            h = page.rect.height
            y_top_limit = h * top_ratio
            y_bottom_limit = h * (1.0 - bottom_ratio)

            blocks = page.get_text("blocks")
            body_blocks = []
            for b in blocks:
                if b[6] != 0:  # 0 이외는 텍스트 블록이 아님 (이미지 등)
                    continue
                # 블록의 중앙 Y좌표가 본문 영역 내에 있는지 판단
                y_center = (b[1] + b[3]) / 2.0
                if y_top_limit <= y_center <= y_bottom_limit:
                    body_blocks.append(b)

            # 블록들을 Y, X 좌표 순으로 정렬하여 텍스트 조립
            body_blocks.sort(key=lambda x: (x[1], x[0]))
            page_text = "\n".join(b[4] for b in body_blocks)
            if is_unspaced(page_text):
                raw = page.get_text("rawdict")
                raw["blocks"].sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
                in_body = lambda b: y_top_limit <= (b["bbox"][1] + b["bbox"][3]) / 2.0 <= y_bottom_limit
                page_text = _respaced_or(page_text, respace_rawdict(raw, in_body))
            out.append(page_text or "")
    return out


def _respaced_or(text: str, respaced: str) -> str:
    """간격으로 다시 조립한 글이 실제로 덜 붙어 있고 글자가 그대로일 때만 쓴다."""
    if real_chars(respaced) != real_chars(text):
        return text
    return respaced if (_nospace_ratio(respaced) or 0) < (_nospace_ratio(text) or 0) else text


def _extract_pdftotext(pdf_path: str | Path) -> list[str]:
    exe = _find_pdftotext()
    if not exe:
        raise RuntimeError("pdftotext 실행파일을 찾을 수 없습니다.")
    # -raw: 콘텐츠 스트림(읽기) 순서 유지 — 다단 편집의 순서 뒤섞임을 방지한다.
    proc = subprocess.run(
        [exe, "-enc", "UTF-8", "-raw", str(pdf_path), "-"],
        capture_output=True, timeout=120,
    )
    text = proc.stdout.decode("utf-8", errors="replace")
    # pdftotext는 페이지를 폼피드(\x0c)로 구분한다. 끝의 빈 조각은 제거.
    pages = text.split("\f")
    if pages and pages[-1].strip() == "":
        pages.pop()
    return pages


# ── 디지털 판별 ────────────────────────────────────────────────────────────────

def looks_digital(pages: list[str]) -> bool:
    """페이지 텍스트 리스트로 문서가 '디지털(텍스트 레이어 보유)'인지 판정."""
    total = len(pages)
    if total == 0:
        return False
    text_pages = sum(1 for p in pages if real_chars(p) >= MIN_PAGE_CHARS)
    total_real = sum(real_chars(p) for p in pages)
    return (
        text_pages >= 1
        and (text_pages / total) >= MIN_TEXT_PAGE_RATIO
        and total_real >= MIN_DOC_CHARS
    )


@dataclass
class DigitalPdf:
    """analyze() 결과."""
    pages: list[str]      # 페이지별 원문 텍스트(index 0 = 1페이지)
    total_pages: int
    text_pages: int       # 텍스트가 있는 페이지 수
    is_digital: bool
    garbled_pages: list[int] = field(default_factory=list)   # 내장 글자가 깨진 쪽(1부터)
    noisy_pages: list[int] = field(default_factory=list)     # 잡음·띄어쓰기 문제 쪽(깨진 쪽 제외)
    repaired_lines: dict[int, int] = field(default_factory=dict)  # 깨진 글꼴 줄을 다시 읽어 고친 {쪽: 줄 수}


def analyze(pdf_path: str | Path, repair: bool = False) -> DigitalPdf | None:
    """PDF를 열어 텍스트 레이어를 추출하고 디지털 여부까지 판정한다.

    repair=True: 깨진 글꼴 줄을 이미지로 다시 읽어 고친다(실제 추출용). 정보 표시용은 False(빠름).
    반환:
      DigitalPdf — 추출 성공(디지털 여부는 .is_digital 로 확인)
      None       — 추출 백엔드 없음 또는 추출 실패(→ 호출부는 기존 OCR로 진행)
    """
    if not is_available():
        return None
    repaired: dict[int, int] | None = {} if repair else None
    try:
        pages = extract_pages(pdf_path, repaired)
    except Exception:
        return None
    total = len(pages)
    if total == 0:
        return None
    text_pages = sum(1 for p in pages if real_chars(p) >= MIN_PAGE_CHARS)
    garbled = [i + 1 for i, p in enumerate(pages) if is_garbled(p)]
    # 깨진 쪽이 많으면 3% 미만 쪽도 깨져 있다(길을묻는그대에게: 중앙값 2.2%) → 문서 전체 OCR
    mostly_garbled = text_pages and len(garbled) / text_pages >= GARBLED_DOC_RATIO
    return DigitalPdf(
        pages=pages,
        total_pages=total,
        text_pages=text_pages,
        is_digital=looks_digital(pages) and not mostly_garbled,
        garbled_pages=garbled,
        noisy_pages=[i + 1 for i, p in enumerate(pages) if i + 1 not in garbled and is_noisy(p)],
        repaired_lines=repaired or {},
    )
