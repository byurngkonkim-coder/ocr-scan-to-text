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
    # 제목도 본문과 같은 띄어쓰기(R-17) — 차례 목록과 본문 제목이 같은 글자(2026-10-24)
    assert [e["title"] for e in rep["toc_entries"]][:3] == ["머리말", "제1장 새로운 경영 패러다임", "1.경영은 기업 경영이다"]
    # 본문에 '제2장 전략' 줄이 없다 — 그 장의 첫 절 앞에 넣는다(추정 1)
    assert rep["toc_note"].startswith("차례 8항목 중 7개를 본문 제목으로 정비(+못 찾은 장 제목 1개")
    lines = out.split("\n")
    for h in ("# 머리말", "# 제1장 새로운 경영 패러다임", "## 1.경영은 기업 경영이다", "## 2.단 하나의 올바른 조직",
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
    main, md_pair = ocr_gradio._structured_texts(body, rep, OutputFormat.MD, logs)
    assert main == md_pair and main.startswith("# 차례\n\n- 제1장 시작\n  - 1.첫 절\n\n# 제1장 시작")
    # txt(+md 짝): 주 결과는 '#'·목록 기호 없는 글, .md 짝은 MD 서식
    txt, md = ocr_gradio._structured_texts(body, rep, OutputFormat.TXT, [])
    assert txt.startswith("차례\n\n제1장 시작\n  1.첫 절\n\n제1장 시작\n\n1.첫 절\n\n본문") and "#" not in txt
    assert md == md_pair
    assert logs == ["  📑 차례 2항목 중 2개"]
    # 차례가 없으면 정제본 그대로
    assert ocr_gradio._structured_texts("본문\n", {"toc_entries": []}, OutputFormat.TXT, []) == ("본문\n", "본문\n")


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


# ── 2026-10-24 차례 품질(intent NotebookLM용 MD 구조개선 작업 2) ──────────────────────
def test_label_only_toc_row_merges_with_title():
    """번호표만 따로 읽힌 장('제1장' / '다음 사회의 모습 28')은 한 항목 — 같은 단계(NextSociety 실측)."""
    toc = ("차례\n제1장\n다음 사회의 모습 28\n제2장\n새로운 인구 구조 41\n제3장\n새로운 노동력 63\n"
           "제4장\n제조업의 역설 85\n제5장\n최고경영자의 미래 101")
    entries, _ = toc_headings.extract_toc_text([toc] + [_BODY] * 20)
    assert [(e["level"], e["title"], e["page"]) for e in entries][:2] == \
        [(1, "제1장 다음 사회의 모습", 28), (1, "제2장 새로운 인구 구조", 41)]


def test_rematch_headings_after_clean():
    """줄 잇기 전 원문에서 못 찾은 차례 제목이 정제본에 따로 선 줄로 있으면 제목으로(DrivingSalesmanPowerSkill 실측).
    본문 문장 속에 같은 말이 있어도 제목으로 바꾸지 않는다."""
    entries = [{"level": 1, "title": "03 이제세일즈를해볼까", "page": 21},
               {"level": 1, "title": "04 첫실적을보장하는약속", "page": 30}]
    text = ("앞에서 말한 03 이제 세일즈를 해 볼까 라는 이야기는 길게 이어지는 본문 문장의 일부일 뿐이다.\n\n"
            "03 이제 세일즈를 해 볼까\n\n그날 나는 처음으로 고객을 만났다.\n\n"
            "# 04 첫실적을보장하는약속\n\n약속은 지켜야 한다.")
    stats = {"found": 1, "inferred": 0, "missing": ["03 이제세일즈를해볼까"]}
    out = text_cleaner.rematch_headings(text, entries, stats, respace=False)
    lines = out.split("\n")
    # 본문 줄 글자 그대로 '#' 만 붙이고, 차례 목록도 그 글자로 맞춘다(무손실)
    assert "# 03 이제 세일즈를 해 볼까" in lines and "# 04 첫실적을보장하는약속" in lines
    assert entries[0]["title"] == "03 이제 세일즈를 해 볼까"
    assert lines[0].startswith("앞에서 말한 03")
    assert out.replace("# 03 이제", "03 이제") == text
    assert stats == {"found": 2, "inferred": 0, "missing": []}


def test_rematch_headings_matches_front_of_merged_toc_row():
    """차례 OCR 이 두 항목·저자를 한 줄로 붙여 읽어도('다이렉트경영 마이클델', CEO의책꽃이) 본문에 따로 선
    앞부분 줄을 제목으로. 짧은 줄(6자 미만·항목의 40% 미만)은 아니다."""
    entries = [{"level": 1, "title": "다이렉트경영 마이클델", "page": 40},
               {"level": 1, "title": "나의 인생과 일 헨리 포드 그리고 자동차 왕국의 탄생 이야기", "page": 50}]
    text = "그림6.3 다이렉트 경영\n\n앞 이야기.\n\n다이렉트 경영\n\n델은 직접 판다.\n\n나의 인생\n\n포드는 차를 만들었다."
    st = {"found": 0, "inferred": 0, "missing": [e["title"] for e in entries]}
    out = text_cleaner.rematch_headings(text, entries, st, respace=False)
    assert "# 다이렉트 경영" in out.split("\n") and "# 나의 인생" not in out
    assert st["found"] == 1


_P = "그날 아침 그녀는 택시 안에서 음악을 들으며 오래도록 창밖을 바라보았다."


def test_infer_chapters_marks_numbered_chapter_lines():
    """차례가 없는 책 — 본문의 장 번호표 줄(CHAPTER N, 1Q84 실측)을 번호 순서대로 제목으로. 글자는 그대로."""
    text = "\n\n".join(["머리 이야기다.", "CHAPTER 1", _P, "CHAPTER 2", _P, "CHAPTER 3", _P])
    out, entries = text_cleaner.infer_chapters(text)
    assert [e["title"] for e in entries] == ["CHAPTER 1", "CHAPTER 2", "CHAPTER 3"]
    assert all(e["level"] == 1 for e in entries)
    assert "# CHAPTER 2" in out.split("\n") and out.replace("# CHAPTER", "CHAPTER") == text


def test_infer_chapters_skips_sentences_toc_remnant_and_repeats():
    """'제1장을 시작하며'(문장)·'제3장 삶을 단순화하라·56'(차례 잔재)·같은 번호 반복은 제목이 아니다
    (GameTheory·FocalPoint·IDEAHACKS49 실측)."""
    text = "\n\n".join(["제3장 삶을 단순화하라·56", "제1장 집중의 힘", "제1장을 시작하며", _P,
                        "제1장 집중의 힘", _P, "제2장 목표를 세워라", _P, "제3장 삶을 단순화하라", _P])
    out, entries = text_cleaner.infer_chapters(text)
    assert [e["title"] for e in entries] == ["제1장 집중의 힘", "제2장 목표를 세워라", "제3장 삶을 단순화하라"]
    lines = out.split("\n")
    assert lines[0] == "제3장 삶을 단순화하라·56" and lines[2] == "# 제1장 집중의 힘" and "제1장을 시작하며" in lines
    assert lines.count("# 제1장 집중의 힘") == 1


def test_infer_chapters_parts_and_too_few():
    """부가 있으면 부 1단계·장 2단계. 장 번호가 3개 미만이면 추정하지 않는다(일반 글의 우연한 '1장')."""
    text = "\n\n".join(["1부", "1장 정렬의 중요성", _P, "2장 경제적 가치", _P, "2부", "3장 내부 조직", _P])
    out, entries = text_cleaner.infer_chapters(text)
    assert [(e["level"], e["title"]) for e in entries] == \
        [(1, "1부"), (2, "1장 정렬의 중요성"), (2, "2장 경제적 가치"), (1, "2부"), (2, "3장 내부 조직")]
    few = "\n\n".join(["1장 처음", _P, "2장 다음", _P])
    assert text_cleaner.infer_chapters(few) == (few, [])


def test_infer_chapters_book_parts_gap_and_parts_only():
    """'BOOK 1 APRIL-JUNE' 은 부(1Q84), OCR 이 한 장을 못 읽어 번호가 하나 건너뛰어도 잇는다(IDEAHACKS49 2장),
    장 줄이 없어도 부가 3개 이상이면 부로 나눈다(Alignment 1부~4부)."""
    text = "\n\n".join(["BOOK 1 APRIL-JUNE", "1장_정보Hack", _P, "3장_정리 Hack", _P, "4장_오감Hack", _P,
                        "BOOK 2 JULY", "CHAPTER 1", _P])
    _, entries = text_cleaner.infer_chapters(text)
    assert [(e["level"], e["title"]) for e in entries] == \
        [(1, "BOOK 1 APRIL-JUNE"), (2, "1장_정보Hack"), (2, "3장_정리 Hack"), (2, "4장_오감Hack"),
         (1, "BOOK 2 JULY"), (2, "CHAPTER 1")]
    # 권 표시 없이 권마다 1장부터(성경) — 장을 3개 넘게 찾은 뒤 1장이면 다시 센다
    bible = "\n\n".join([f"제 {n} 장\n\n{_P}" for n in (1, 2, 3, 4, 1, 2, 3)])
    assert len(text_cleaner.infer_chapters(bible)[1]) == 7
    parts = "\n\n".join(["1부", _P, "2부", _P, "CHAPTER9", _P, "3부", _P, "4부", _P, "2장 주석의 소제목", _P])
    assert [e["title"] for e in text_cleaner.infer_chapters(parts)[1]] == ["1부", "2부", "3부", "4부"]


def test_iter_clean_infers_chapters_without_toc(monkeypatch):
    """차례를 못 찾은 책도 정제 결과에 장 제목·맨 앞 목록용 항목이 생기고, 보고는 '추정'이라고 밝힌다."""
    monkeypatch.setattr(config, "get_local_cleanup_engine", lambda: "rules")
    rep: dict = {}
    names = ["경영의 원칙", "고객의 발견", "혁신의 조건", "성과의 측정"]
    pages = [f"제{i}장 {n}\n" + _BODY for i, n in enumerate(names, 1)] + [_BODY] * 6
    out = _clean(pages, rep)
    assert [e["title"] for e in rep["toc_entries"]][:2] == ["제1장 경영의 원칙", "제2장 고객의 발견"]
    assert "# 제1장 경영의 원칙" in out.split("\n")
    assert "본문에서 장 제목 4개를 추정" in rep["toc_note"]


def test_rematch_headings_never_drops_or_duplicates_lines():
    """정제본 다시 찾기는 줄을 지우거나 이미 있는 제목을 또 넣지 않는다(NextSociety '피터 드러커' 줄 소실 회귀)."""
    entries = [{"level": 1, "title": "책머리에", "page": 9},
               {"level": 1, "title": "제4부 변화하는 세계 경제", "page": 290},
               {"level": 2, "title": "제1장 대규모 조직의 등장", "page": 297}]
    text = ("# 책머리에\n\n피터 드러커\n\n본문 문장이다.\n\n# 제4부 변화하는 세계 경제\n\n"
            "## 제1장 대규모 조직의 등장\n\n조직은 커졌다.")
    out = text_cleaner.rematch_headings(text, entries, {"found": 3, "inferred": 0, "missing": []}, respace=False)
    assert out == text


def test_rematch_headings_respaces_titles_and_list():
    """respace: 제목을 본문과 같은 띄어쓰기로 — 차례 목록(entries)과 본문 제목 줄이 같은 글자."""
    entries = [{"level": 1, "title": "03 이제세일즈를해볼까", "page": 21}]
    out = text_cleaner.rematch_headings("# 03 이제세일즈를해볼까\n\n본문이다.", entries,
                                        {"found": 1, "inferred": 0, "missing": []}, respace=True)
    assert entries[0]["title"] == "03 이제 세일즈를 해 볼까"
    assert "# 03 이제 세일즈를 해 볼까" in out.split("\n") and "이제세일즈" not in out
