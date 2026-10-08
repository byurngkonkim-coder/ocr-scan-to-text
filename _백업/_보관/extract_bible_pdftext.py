# -*- coding: utf-8 -*-
"""킹제임스 흠정역 성경전서 PDF — 텍스트 레이어 결정론 추출 (재OCR 대체 정공법)

발견 사실 (2026-07-04 실측):
  - 이 PDF는 텍스트 레이어를 내장한다 (비전 OCR 불필요 → 글자 오류 0)
  - PyMuPDF 블록 추출이 2단 조판을 올바른 읽기 순서로 반환한다 (인터리빙 원천 차단)
  - 줄 끝 후행 공백 유무가 어절 경계를 인코딩한다:
      '숯 가운' + '데서 일하고'   → 후행 공백 없음 → 직접 결합 '숯 가운데서 일하고'
      '모양을 ' + '내며'          → 후행 공백 있음 → 공백 결합 '모양을 내며'
    → 어절 내 공백 문제의 무손실 해결 (규칙서 §14/§15 원문충실 원칙 부합)

제거하는 것은 인쇄 러닝헤드(페이지 번호 + '책명 장' 헤더 블록)뿐이다.
줄 분리는 절 번호(^N ), 장 표제(제 N 장), ¶ 단락 기호, 문장 종결 뒤 제목 블록에서만.
"""
import io
import os
import re
import sys

import fitz  # PyMuPDF

BASE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(BASE, "king_james_bible.pdf")
OUT = os.path.join(BASE, "king_james_bible_pdftext.txt")

VERSE_START_RE = re.compile(r"^(\d{1,3})\s")
CHAPTER_RE = re.compile(r"^제\s*\d+\s*[장편]\s*$")
PILCROW_RE = re.compile(r"^¶")
HEAD_PAGENUM_RE = re.compile(r"^\d{1,4}$")
HEAD_BOOK_RE = re.compile(r"^[가-힣]+\s\d{1,3}$")
TERMINAL_PUNCT_RE = re.compile(r"[.!?。'\"’”)\]]\s*$")


def is_running_head_block(text: str, y0: float) -> bool:
    """페이지 상단의 '쪽번호 + 책명 장' 러닝헤드 블록 판정 (본문 아님)."""
    if y0 > 40:
        return False
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines or len(lines) > 2:
        return False
    return all(HEAD_PAGENUM_RE.match(l) or HEAD_BOOK_RE.match(l) for l in lines)


def main():
    doc = fitz.open(PDF)
    out_lines = []          # 완성된 출력 줄들
    cur = ""                # 조립 중인 현재 줄

    def flush():
        nonlocal cur
        if cur.strip():
            out_lines.append(cur.rstrip())
        cur = ""

    # 후행 공백 규약(없음=어절 분할, 있음=어절 경계)은 좁은 2단 '본문' 블록에서만
    # 성립한다. 표제지·판권·책 소개 같은 넓은 블록은 장식 조판이라 규약이 없으므로
    # 공백 결합 + 블록 단위 단락으로 처리한다 (실측: 직접 결합 시 '부지런히비교하고'
    # 같은 어절 오결합 발생).
    NARROW_MAX_WIDTH = 200.0

    for pno, page in enumerate(doc):
        for x0, y0, x1, y1, text, bno, btype in page.get_text("blocks"):
            if btype != 0:
                continue
            if is_running_head_block(text, y0):
                continue
            # 표제지·판권(첫 2페이지)은 중앙 정렬 장식 조판이라 블록 폭이 좁아도
            # 본문 규약(후행 공백 = 어절 경계)이 성립하지 않는다 → 부속물 정책 강제
            narrow = (x1 - x0) <= NARROW_MAX_WIDTH and pno >= 2

            if not narrow:
                flush()                       # 부속물 블록 = 독립 단락
                para = " ".join(l.strip() for l in text.splitlines() if l.strip())
                if para:
                    out_lines.append(para)
                continue

            block_lines = text.splitlines()
            for li, line in enumerate(block_lines):
                if not line.strip():
                    continue
                starts_new = (
                    VERSE_START_RE.match(line)
                    or CHAPTER_RE.match(line.strip())
                    or PILCROW_RE.match(line.lstrip())
                    # 블록 첫 줄 + 직전 줄이 문장 종결로 끝남 = 제목/새 단락
                    or (li == 0 and cur and TERMINAL_PUNCT_RE.search(cur))
                )
                if starts_new:
                    flush()
                if not cur:
                    cur = line
                else:
                    # 후행 공백 있으면 그 공백이 어절 경계로 남고,
                    # 없으면 직접 이어붙어 줄에 걸쳐 분할된 어절이 복원된다.
                    cur = cur + line
                if CHAPTER_RE.match(line.strip()):
                    flush()                   # 장 표제는 단독 줄로
    flush()

    with io.open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"페이지 {doc.page_count} → 출력 줄 {len(out_lines):,}")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
