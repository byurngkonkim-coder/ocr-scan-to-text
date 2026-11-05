"""쪽 경계 표지 [p.N] — 정제본을 쪽별 원문과 대조해 원본 PDF 쪽을 되살린다
(intent docs/intents/2026-10-24-NotebookLM용-MD-구조개선.md, 작업 1)."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import page_marks
from core.types import OutputFormat

_P1 = "경영자는 조직의 성과를 책임지는 사람이다. 그 성과는 조직 바깥에서 고객이 판정한다."
_P2 = "그러므로 내부의 노력만으로는 충분하다고 말할 수 없다. 시장이 마지막 심판이기 때문이다."
_P3 = "지식 근로자는 생산 수단을 머릿속에 지니고 다닌다. 이것이 새로운 사회의 바탕이다."


def _strip(text: str) -> str:
    """표지 줄을 뺀 글 — 무손실 확인용(문장 뒤에 넣은 표지는 그 자리 띄어쓰기가 줄바꿈으로 남는다)."""
    return re.sub(r'\[p\.\d+\]\n', '', text)


def test_marks_each_page_in_order_without_losing_text():
    clean = f"{_P1}\n\n{_P2}\n\n{_P3}"
    out, st = page_marks.mark_pages(clean, [(5, _P1), (6, _P2), (7, _P3)])
    assert re.findall(r'^\[p\.(\d+)\]$', out, re.M) == ["5", "6", "7"]
    assert out.startswith("[p.5]\n" + _P1)
    assert _strip(out) == clean
    assert st == {"found": 3, "missing": []}


def test_page_turning_mid_sentence_puts_mark_after_sentence_end():
    """쪽이 문장 중간에서 바뀌면 그 문장이 끝난 뒤 줄에 — 문장을 끊지 않는다."""
    a = "경영자는 조직의 성과를 책임지는 사람이다. 그 성과는 조직 바깥에서"
    b = "고객이 판정한다. 그러므로 내부의 노력만으로는 충분하다고 말할 수 없다."
    clean = "경영자는 조직의 성과를 책임지는 사람이다. 그 성과는 조직 바깥에서 고객이 판정한다. " \
            "그러므로 내부의 노력만으로는 충분하다고 말할 수 없다."
    out, _ = page_marks.mark_pages(clean, [(1, a), (2, b)])
    assert "바깥에서 고객이 판정한다.\n[p.2]\n그러므로" in out
    assert _strip(out).replace("\n", " ") == clean


def test_removed_page_is_skipped():
    """정제가 지운 쪽(표지·광고)은 표지를 넣지 않고 못 찾은 쪽으로 보고한다."""
    ad = "이 출판사의 다른 책들을 만나 보세요 경영 시리즈 전 열두 권 절찬 판매 중입니다"
    clean = f"{_P1}\n\n{_P3}"
    out, st = page_marks.mark_pages(clean, [(1, _P1), (2, ad), (3, _P3)])
    assert re.findall(r'\[p\.(\d+)\]', out) == ["1", "3"]
    assert st["missing"] == [2]


def test_spacing_and_typo_fixes_still_match():
    """정제가 띄어쓰기를 넣고 글자 한두 개를 고쳐도 같은 쪽으로 찾는다."""
    raw2 = "그러므로내부의노력만으로는충분하다고말할수없다.시장이마지막심판이기때문이다."
    clean = f"{_P1}\n\n{_P2}"
    out, st = page_marks.mark_pages(clean, [(1, _P1), (2, raw2)])
    assert st["found"] == 2 and "\n[p.2]\n" + _P2 in out


def test_mark_goes_after_heading_line():
    """쪽이 장 제목으로 시작하면 표지는 제목 다음 줄 — 그 장 본문의 쪽이 된다."""
    clean = f"{_P1}\n\n# 제2장 지식 사회\n\n{_P3}"
    out, _ = page_marks.mark_pages(clean, [(1, _P1), (2, "제2장 지식 사회\n" + _P3)])
    assert "# 제2장 지식 사회\n\n[p.2]\n" + _P3 in out


def test_mark_not_inside_markdown_table():
    """쪽 시작이 표 안이면 표가 끝난 뒤에 — 표를 깨지 않는다."""
    table = "| 구분 | 낮음 | 높음 |\n| --- | --- | --- |\n| 가치 | 시장점유율이 낮은 사업부 | 성장률이 높은 사업부 |"
    clean = f"{_P1}\n\n{table}\n\n{_P3}"
    out, _ = page_marks.mark_pages(clean, [(1, _P1), (2, "시장점유율이 낮은 사업부 성장률이 높은 사업부\n" + _P3)])
    assert table in out and "\n[p.2]\n" in out.split(table)[1]


def test_structured_texts_adds_marks_to_md_and_txt():
    """txt+md 짝 — 두 파일 모두 같은 쪽 표지(사용자 결정 2026-10-24)."""
    import ocr_gradio
    clean = f"{_P1}\n\n{_P2}\n"
    pages = [(3, _P1), (4, _P2)]
    txt, md = ocr_gradio._structured_texts(clean, {"toc_entries": []}, OutputFormat.TXT, [], pages)
    assert "[p.3]" in txt and "[p.4]" in txt and txt == md
    # 쪽 정보가 없으면(txt 입력) 지금처럼 그대로
    assert ocr_gradio._structured_texts(clean, {"toc_entries": []}, OutputFormat.TXT, []) == (clean, clean)


# ── 문서 정보 머리말(intent 작업 4) ──────────────────────────────────────────────
def test_book_name_from_filename():
    """파일명 '책제목_저자' — 끝의 날짜·상태 표시는 뗀다. 칸이 하나면 저자 없음."""
    import ocr_gradio
    assert ocr_gradio._book_name("20261024_062842_NextSociety_피터드러커.pdf") == ("NextSociety", "피터드러커")
    assert ocr_gradio._book_name("1001가지기독교명언_론로즈_20240302.pdf") == ("1001가지기독교명언", "론로즈")
    assert ocr_gradio._book_name("1Q84_MurakamiHaruki_기울기 보정.pdf") == ("1Q84", "MurakamiHaruki")
    assert ocr_gradio._book_name("2천년의강의_사마천의생각경영법_김원중.pdf") == ("2천년의강의 사마천의생각경영법", "김원중")
    assert ocr_gradio._book_name("BCG의6가지성공전략_원문.pdf") == ("BCG의6가지성공전략", "")
    assert ocr_gradio._book_name("GameTheory.pdf") == ("GameTheory", "")


def test_book_name_author_first_when_front_text_says_so():
    """'저자_책제목' 파일명 — 앞쪽 글에 '마이클 포터 지음' 이 있으면 순서를 바꾼다(마이클포터_경쟁전략 실측)."""
    import ocr_gradio
    front = "경쟁전략\nCOMPETITIVE STRATEGY 마이클 포터 지음 조동성 옮김"
    assert ocr_gradio._book_name("20261106_044132_마이클포터_경쟁전략.pdf", front) == ("경쟁전략", "마이클포터")
    # 앞쪽 글이 파일명 순서와 맞으면 그대로
    assert ocr_gradio._book_name("NextSociety_피터드러커.pdf", "피터 드러커 지음") == ("NextSociety", "피터드러커")


def test_structured_texts_doc_header_md_and_txt():
    """맨 앞 문서 정보 — 책·저자(파일명 기준 추정)·원본 파일·쪽 수·처리일·장절 결과·쪽 표지 뜻. MD 는 인용 블록,
    TXT 는 같은 줄에서 기호만 뺀다. 본문(차례·쪽 표지)은 그 뒤에 그대로."""
    import ocr_gradio
    clean = f"{_P1}\n\n{_P2}\n"
    rep = {"toc_entries": [], "toc_note": "차례를 찾지 못해 장·절 정비 없이 정제"}
    info = {"name": "NextSociety_피터드러커.pdf", "pages": 374}
    txt, md = ocr_gradio._structured_texts(clean, rep, OutputFormat.TXT, [], [(3, _P1), (4, _P2)], info)
    head, body = md.split("\n\n", 1)
    assert head.startswith("> **문서 정보**") and all(l.startswith(">") for l in head.split("\n"))
    for s in ("NextSociety", "피터드러커", "파일명 기준", "NextSociety_피터드러커.pdf", "PDF 374쪽", "처리일",
              "차례를 찾지 못해", "[p.N]"):
        assert s in head
    assert body.startswith("[p.3]\n" + _P1)
    t_head, t_body = txt.split("\n\n", 1)
    assert ">" not in t_head and "**" not in t_head and "NextSociety_피터드러커.pdf" in t_head
    assert t_body == body
