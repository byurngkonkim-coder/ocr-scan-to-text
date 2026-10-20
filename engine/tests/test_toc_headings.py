"""차례 기준 장·절 정비(toc_headings) — 정제(iter_clean) 연결·줄 잇기 보호·MD/TXT 저장.

toc_headings.py 는 MyOCR(engine/toc_headings.py)와 바이트 그대로 같은 파일이다. 모듈 자체 단위 테스트는
MyOCR tests/test_book_layout.py 에 있고, 여기서는 01 정제 경로에 붙인 부분을 본다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import config, postprocessor, text_cleaner, toc_headings
from core.types import OutputFormat

_BODY = ("경영자는 조직의 성과를 책임지는 사람이며 그 성과는 조직 바깥에서 고객에 의해 판정되므로 "
         "내부의 노력만으로는 충분하다고 말할 수 없다.")
_TOC = ("차례\n·머리말/3\n제1장 새로운 경영 패러다임\n1.경영은 기업경영이다 7\n2.단 하나의 올바른 조직\n9\n"
        "제2장 전략\n1.감소하는 출산율 11\n2.소득분포 13\n·감사의말/15")


def _book():
    pages = ["표지", _TOC, "머리말\n" + _BODY, _BODY, "제1장 새로운 경영 패러다임\n" + _BODY,
             "1.경영은 기업경영이다\n" + _BODY, _BODY, "2.단 하나의 올바른 조직\n" + _BODY, _BODY,
             "1.감소하는 출산율\n" + _BODY, _BODY, "2.소득분포\n" + _BODY, _BODY, "감사의말\n" + _BODY]
    return pages + [_BODY] * 20


def _clean(pages, report):
    g = text_cleaner.iter_clean(pages, local=True, report_out=report)
    try:
        while True:
            next(g)
    except StopIteration as s:
        return s.value


def test_iter_clean_marks_toc_headings(monkeypatch):
    monkeypatch.setattr(config, "get_local_cleanup_engine", lambda: "rules")
    rep: dict = {}
    out = _clean(_book(), rep)
    assert [e["title"] for e in rep["toc_entries"]][:3] == ["머리말", "제1장 새로운 경영 패러다임", "1.경영은 기업경영이다"]
    # 본문에 '제2장 전략' 줄이 없다 — 그 장의 첫 절 앞에 넣는다(추정 1)
    assert rep["toc_note"].startswith("차례 8항목 중 7개를 본문 제목으로 정비(+못 찾은 장 제목 1개")
    lines = out.split("\n")
    for h in ("# 머리말", "# 제1장 새로운 경영 패러다임", "## 1.경영은 기업경영이다", "## 2.단 하나의 올바른 조직",
              "# 제2장 전략", "## 1.감소하는 출산율", "# 감사의말"):
        i = lines.index(h)                                   # 제목 줄이 본문에 붙지 않았다
        assert lines[i - 1] == "" and lines[i + 1] == ""     # 앞뒤 빈 줄
    assert "·머리말/3" not in out                            # 차례 쪽은 본문에서 빠진다(MD 맨 앞 목록으로)


def test_iter_clean_without_toc_unchanged(monkeypatch):
    monkeypatch.setattr(config, "get_local_cleanup_engine", lambda: "rules")
    rep: dict = {}
    out = _clean([_BODY] * 5, rep)
    assert "#" not in out and rep["toc_entries"] == [] and "차례를 찾지 못해" in rep["toc_note"]


def test_join_broken_lines_keeps_heading_line():
    long = "가" * 40
    assert postprocessor.join_broken_lines(f"# 제1장 시작하며 이야기를 풀어 간다\n{long}") == \
        f"# 제1장 시작하며 이야기를 풀어 간다\n{long}"


def test_pagebreak_before_heading_keeps_paragraph():
    out = text_cleaner.post_process_pagebreaks("끝나지 않은 문장의\n⟨PAGEBREAK⟩\n# 제2장 전략")
    assert out == "끝나지 않은 문장의\n\n# 제2장 전략"


def test_structured_text_md_and_txt():
    """저장 형식이 md 면 맨 앞 차례 목록 + '#' 제목, txt 면 같은 글에서 '#'·목록 기호만 뺀다."""
    import ocr_gradio
    entries = [{"level": 1, "title": "제1장 시작", "page": None}, {"level": 2, "title": "1.첫 절", "page": 3}]
    body = "# 제1장 시작\n\n## 1.첫 절\n\n본문 문장이다.\n"
    rep = {"toc_entries": entries, "toc_note": "차례 2항목 중 2개"}
    logs: list[str] = []
    md = ocr_gradio._structured_text(body, rep, OutputFormat.MD, logs)
    assert md.startswith("# 차례\n\n- 제1장 시작\n  - 1.첫 절\n\n# 제1장 시작")
    txt = ocr_gradio._structured_text(body, rep, OutputFormat.TXT, [])
    assert txt.startswith("차례\n\n제1장 시작\n  1.첫 절\n\n제1장 시작\n\n1.첫 절\n\n본문") and "#" not in txt
    assert logs == ["  📑 차례 2항목 중 2개"]
    # 차례가 없으면 정제본 그대로
    assert ocr_gradio._structured_text("본문\n", {"toc_entries": []}, OutputFormat.TXT, []) == "본문\n"


def test_toc_prepass_reads_raw_lines_before_page_join(monkeypatch):
    """쪽 교정의 줄 잇기가 차례 줄을 한 줄로 붙이기 전(raw_text)에서 차례를 읽는다(1분의사결정 실측)."""
    import ocr_gradio
    from core.types import OCREngine, PageResult
    monkeypatch.setattr(config, "get_local_cleanup_engine", lambda: "rules")
    raws = _book()
    raws[1] = ("차 례\n머리말.9\nI. 자신의 길을 찾는다\n1. 혼돈에서 탈출한다（금요일 아침） .15\n"
               "2. 신뢰할 수 있는 방법 .27\nII. 머리를 써라\n3. 정말로 필요한 것은? .37\n"
               "4. 여러 가지 선택 가능한 방법에 대해 안다\n（금요일 저녁）-53\n5.\n끝까지 생각한다 .68")
    raws[2] = "머리말\n" + _BODY
    raws[4] = "혼돈에서 탈출한다\n（금요일 아침）\n" + _BODY            # 장 표제지 없음 · 번호표 없는 절 제목
    raws[5] = "I . 자신의 길을 찾는다 17\n" + _BODY                     # 쪽 머리글(첫 등장) — 제목이 아니다
    raws[7] = "신뢰할 수 있는 방법\n" + _BODY
    pages = [PageResult(page_num=i + 1, raw_text=r, corrected_text=" ".join(r.split("\n")),
                        engine=OCREngine.RAPIDOCR, confidence=90.0) for i, r in enumerate(raws)]
    texts, toc = ocr_gradio._toc_prepass(pages)
    titles = [e["title"] for e in toc["entries"]]
    assert "4. 여러 가지 선택 가능한 방법에 대해 안다 (금요일 저녁)" in titles and "5. 끝까지 생각한다" in titles
    assert texts[1] == ""                                              # 차례 쪽은 본문에서 비운다
    assert texts[4].startswith("# I. 자신의 길을 찾는다") and "## 1. 혼돈에서 탈출한다" in texts[4]
    assert "#" not in texts[5]                                         # 머리글 줄에는 붙지 않았다
    assert texts[3] == pages[3].corrected_text                         # 제목 없는 쪽은 교정본 그대로
    # 정제 끝까지 제목이 남는다 — 잡음 기호 제거(clean_page)가 '#' 을 지우던 것(실측: 결과에 제목 0줄)
    rep: dict = {}
    g = text_cleaner.iter_clean(texts, local=True, report_out=rep, toc=toc)
    try:
        while True:
            next(g)
    except StopIteration as s:
        out = s.value
    heads = [l for l in out.split("\n") if l.startswith("#")]
    assert "# I. 자신의 길을 찾는다" in heads and "## 1. 혼돈에서 탈출한다(금요일 아침)" in heads
    assert rep["toc_note"] == toc_headings.report(toc["entries"], toc["stats"])


def test_chapter_page_heading_survives_fuzzy_header_rule(monkeypatch):
    """장 표제지 쪽의 '# I.자신의 길을 찾는다' 가 같은 글자의 쪽 머리글('I.자신의 길을 찾는다 17')로 지워지던 것."""
    monkeypatch.setattr(config, "get_local_cleanup_engine", lambda: "rules")
    pages = ["\n# I.자신의 길을 찾는다\n"] + [f"I.자신의 길을 찾는다 {17 + 2 * k}\n{_BODY}" for k in range(8)]
    toc = {"entries": [{"level": 1, "title": "I.자신의 길을 찾는다", "page": None}], "stats": {"found": 1}}
    g = text_cleaner.iter_clean(pages, local=True, report_out={}, toc=toc)
    try:
        while True:
            next(g)
    except StopIteration as s:
        out = s.value
    assert out.startswith("# I.자신의 길을 찾는다\n\n") and "찾는다 19" not in out


def test_chapter_title_page_survives_front_matter_removal(monkeypatch):
    """부속물 삭제를 켜면 제목 한 줄뿐인 장 표제지가 '표지·제목' 쪽으로 지워졌다(1분의사결정 9번째 쪽)."""
    monkeypatch.setattr(config, "get_local_cleanup_engine", lambda: "rules")
    pages = ["표지", "책 제목", "\n# I.자신의 길을 찾는다\n", "## 1.혼돈에서 탈출한다\n\n" + _BODY] + [_BODY] * 40
    toc = {"entries": [{"level": 1, "title": "I.자신의 길을 찾는다", "page": None},
                       {"level": 2, "title": "1.혼돈에서 탈출한다", "page": 15}], "stats": {"found": 2}}
    g = text_cleaner.iter_clean(pages, drop_front_matter=True, local=True, report_out={}, toc=toc)
    try:
        while True:
            next(g)
    except StopIteration as s:
        out = s.value
    assert "# I.자신의 길을 찾는다\n\n## 1.혼돈에서 탈출한다" in out


def test_keep_headings_applies_only_to_body():
    assert text_cleaner.keep_headings(str.upper, "# 제목 a\nbody\n\n## 절 b\nmore") == "# 제목 a\n\nBODY\n\n## 절 b\n\nMORE"
    assert text_cleaner.keep_headings(str.upper, "no heading") == "NO HEADING"


def test_shared_module_identical_to_myocr():
    """MyOCR 과 같은 파일인지 — 한쪽만 고치면 두 프로그램 결과가 갈린다."""
    import hashlib
    mine = Path(toc_headings.__file__).read_bytes()
    other = Path(__file__).resolve().parents[3] / "MyOCR" / "engine" / "toc_headings.py"
    if other.exists():
        assert hashlib.md5(mine.replace(b"\r\n", b"\n")).hexdigest() == \
            hashlib.md5(other.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
