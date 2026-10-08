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


def is_noisy(text: str) -> bool:
    hangul = re.findall(r"[가-힣]", text or "")
    if len(hangul) < _GARBLE_MIN_HANGUL:
        return False
    nospace = sum(len(r) for r in re.findall(r"[가-힣]{8,}", text)) / len(hangul)
    noise = len(_NOISE_RE.findall(text)) / real_chars(text)
    return noise >= NOISY_SYMBOL_RATIO or nospace >= NOISY_NOSPACE_RATIO


def real_chars(text: str) -> int:
    """공백(스페이스·탭·개행)을 제외한 실제 글자 수."""
    if not text:
        return 0
    return len(_WS_RE.sub("", text))


# ── 백엔드 감지 ────────────────────────────────────────────────────────────────

def _has_fitz() -> bool:
    try:
        import fitz  # noqa: F401  (PyMuPDF)
        return True
    except Exception:
        return False


def _find_pdftotext() -> str | None:
    """poppler pdftotext 실행파일 경로. Windows 기본 설치 경로 우선, 없으면 PATH."""
    import platform
    if platform.system() == "Windows":
        cand = Path("C:/poppler/Library/bin/pdftotext.exe")
        if cand.exists():
            return str(cand)
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

def extract_pages(pdf_path: str | Path) -> list[str]:
    """PDF의 페이지별 텍스트 레이어를 리스트로 추출한다(index 0 = 1페이지).

    사용 가능한 백엔드가 없으면 RuntimeError.
    """
    b = backend()
    if b == "fitz":
        return _extract_fitz(pdf_path)
    if b == "pdftotext":
        return _extract_pdftotext(pdf_path)
    raise RuntimeError(
        "PDF 텍스트 레이어 추출 백엔드가 없습니다 (PyMuPDF/poppler 미설치)."
    )


_FITZ_LOCK = threading.Lock()   # PyMuPDF 는 스레드 안전이 아니다 — 여러 파일 동시 처리 시 한 번에 하나만


def _extract_fitz(pdf_path: str | Path) -> list[str]:
    with _FITZ_LOCK:
        return _extract_fitz_unlocked(pdf_path)


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
                out.append(page.get_text("text") or "")
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
            out.append(page_text or "")
    return out


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


def analyze(pdf_path: str | Path) -> DigitalPdf | None:
    """PDF를 열어 텍스트 레이어를 추출하고 디지털 여부까지 판정한다.

    반환:
      DigitalPdf — 추출 성공(디지털 여부는 .is_digital 로 확인)
      None       — 추출 백엔드 없음 또는 추출 실패(→ 호출부는 기존 OCR로 진행)
    """
    if not is_available():
        return None
    try:
        pages = extract_pages(pdf_path)
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
    )
