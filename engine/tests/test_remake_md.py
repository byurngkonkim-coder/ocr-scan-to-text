"""기존 결과 재변환 — NotebookLM용 MD 다시 만들기(intent 2026-10-24 NotebookLM용 MD 구조개선, 작업 6).
본문은 기존 정제 결과 그대로, 구조(제목·쪽 표지·머리말·나누기)만 다시 입힌다. OCR·정제·API 를 부르지 않는다."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import remake_md, storage_manager, text_cleaner

_P1 = "경영자는 조직의 성과를 책임지는 사람이다. 그 성과는 조직 바깥에서 고객이 판정한다."
_P2 = "그러므로 내부의 노력만으로는 충분하다고 말할 수 없다. 시장이 마지막 심판이기 때문이다."
_P3 = "지식 근로자는 생산 수단을 머릿속에 지니고 다닌다. 이것이 새로운 사회의 바탕이다."
_MD = f"# 차례\n\n- 제1장 경영\n  - 1.성과\n- 제2장 지식\n\n# 제1장 경영\n\n## 1.성과\n\n{_P1}\n\n{_P2}\n\n# 제2장 지식\n\n{_P3}\n"
_RAW_MD = ("# 책_저자.pdf\n\n- 총 페이지: 12\n- OCR 처리: 3p\n\n---\n\n"
           f"## 페이지 5\n\n제1장 경영\n1.성과\n{_P1}\n\n---\n\n## 페이지 6\n\n> ⚠ **검토 필요**\n> 사유: 낮음\n\n{_P2}\n\n---\n\n"
           f"## 페이지 7\n\n제2장 지식\n{_P3}\n\n---\n")


def test_parse_result_md_and_txt():
    entries, body = remake_md.parse_result(_MD)
    assert [(e["level"], e["title"]) for e in entries] == [(1, "제1장 경영"), (2, "1.성과"), (1, "제2장 지식")]
    assert body.startswith("# 제1장 경영")
    txt = f"차례\n\n제1장 경영\n  1.성과\n제2장 지식\n\n제1장 경영\n\n1.성과\n\n{_P1}\n"
    entries, body = remake_md.parse_result(txt)
    assert [(e["level"], e["title"]) for e in entries] == [(1, "제1장 경영"), (2, "1.성과"), (1, "제2장 지식")]
    assert body.startswith("제1장 경영\n\n1.성과")


def test_parse_result_strips_previous_remake():
    """한 번 다시 만든 결과를 또 돌려도 같은 결과 — 머리말·쪽 표지를 떼고 시작한다."""
    again = "> **문서 정보** — 시험\n> - 책: 시험\n\n" + _MD.replace(f"{_P1}\n", f"[p.5]\n{_P1}\n")
    assert remake_md.parse_result(again) == remake_md.parse_result(_MD)
    txt_again = "문서 정보 — 시험\n책: 시험\n\n차례\n\n제1장 경영\n\n제1장 경영\n\n[p.5]\n" + _P1
    entries, body = remake_md.parse_result(txt_again)
    assert [e["title"] for e in entries] == ["제1장 경영"] and "[p." not in body


def test_read_raw_pages_md_and_txt(tmp_path):
    md = tmp_path / "a_원문.md"
    md.write_text(_RAW_MD, encoding="utf-8")
    pages, name, total = remake_md.read_raw_pages(md)
    assert [n for n, _ in pages] == [5, 6, 7] and name == "책_저자.pdf" and total == 12
    assert "검토 필요" not in pages[1][1] and _P2 in pages[1][1]
    txt = tmp_path / "b_원문.txt"
    sep = "─" * 60
    txt.write_text(f"⬜ 빈 페이지 건너뜀 안내\n{sep}\n전체 9페이지 중 1페이지가 빈 페이지로\n{sep}\n\n"
                   f"[페이지 2]\n{sep}\n{_P1}\n\n[페이지 3]  ⚠ [검토필요] 낮음\n{sep}\n{_P2}\n", encoding="utf-8")
    pages, name, total = remake_md.read_raw_pages(txt)
    assert [n for n, _ in pages] == [2, 3] and name == "" and total == 9 and pages[0][1].strip() == _P1


def _env(tmp_path, monkeypatch):
    res = tmp_path / "02_결과"
    res.mkdir()
    up = tmp_path / "uploads"
    up.mkdir()
    monkeypatch.setattr(remake_md, "BACKUP_ROOT", tmp_path / "_백업")
    monkeypatch.setattr(storage_manager, "raw_copy_path", lambda p: up / f"{Path(p).stem}_원문{Path(p).suffix}")
    return res, up


def test_remake_rebuilds_structure_without_touching_body(tmp_path, monkeypatch):
    res, up = _env(tmp_path, monkeypatch)
    md = res / "20261024_000000_책_저자.md"
    md.write_text(_MD, encoding="utf-8")
    (res / "20261024_000000_책_저자.txt").write_text("옛 txt", encoding="utf-8")
    (up / "20261024_000000_책_저자_원문.md").write_text(_RAW_MD, encoding="utf-8")
    # OCR·정제를 부르면 실패하게 — 재변환은 기존 본문만 쓴다
    monkeypatch.setattr(text_cleaner, "iter_clean", lambda *a, **k: (_ for _ in ()).throw(AssertionError("정제 호출")))
    row = remake_md.remake(md)
    out = md.read_text(encoding="utf-8")
    assert out.startswith("> **문서 정보**") and "책_저자.pdf · PDF 12쪽" in out
    assert re.findall(r'^\[p\.(\d+)\]$', out, re.M) == ["5", "6", "7"]
    assert "# 제1장 경영" in out.split("\n") and "## 1.성과" in out.split("\n")
    txt = (res / "20261024_000000_책_저자.txt").read_text(encoding="utf-8")
    assert txt.startswith("문서 정보 — ") and "[p.6]" in txt and "#" not in txt
    assert row["marks"] == 3 and row["raw"] and row["status"] == "완료"
    backups = list((tmp_path / "_백업").rglob("*"))
    assert {p.name for p in backups if p.is_file()} == {md.name, "20261024_000000_책_저자.txt"}
    # 본문 글자는 그대로(제목 표시·쪽 표지·머리말만 더해짐)
    body = re.sub(r'^\[p\.\d+\]\n', '', out.split("\n\n", 1)[1], flags=re.M)
    assert re.sub(r'\s', '', body) == re.sub(r'\s', '', _MD)
    # 두 번 돌려도 같다
    remake_md.remake(md)
    assert md.read_text(encoding="utf-8") == out


def test_remake_without_raw_copy_still_structures(tmp_path, monkeypatch):
    """원문 사본이 없으면 쪽 표지만 빼고 — 차례 없는 책은 장 추정, 머리말엔 쪽 수 없음."""
    res, _ = _env(tmp_path, monkeypatch)
    t = res / "20261024_000000_긴책.txt"
    t.write_text("\n\n".join([f"제{i}장 이야기 {n}\n\n{_P1}" for i, n in enumerate("가나다라", 1)]), encoding="utf-8")
    row = remake_md.remake(t)
    md = t.with_suffix(".md").read_text(encoding="utf-8")
    assert "# 제1장 이야기 가" in md.split("\n") and "[p." not in md and "PDF" not in md.split("\n\n")[0]
    assert row["raw"] is False and row["marks"] == 0 and "추정" in row["toc"]


def test_dashboard_remake_run(tmp_path, monkeypatch):
    """② [교정] 탭 'NotebookLM용 MD 다시 만들기' — 목록은 정제 결과만, 실행 결과에 쪽 표지·원본 보관 위치."""
    import ocr_gradio
    res, up = _env(tmp_path, monkeypatch)
    monkeypatch.setattr(ocr_gradio, "RESULTS_DIR", res)
    (res / "20261024_000000_책_저자.md").write_text(_MD, encoding="utf-8")
    (up / "20261024_000000_책_저자_원문.md").write_text(_RAW_MD, encoding="utf-8")
    (res / "20261024_000001_쪽단위.md").write_text("# a.pdf\n\n## 페이지 1\n\n본문\n", encoding="utf-8")
    assert ocr_gradio._remake_choices() == [("20261024_000000_책_저자.md", "20261024_000000_책_저자.md")]
    assert "먼저 고르세요" in list(ocr_gradio._remake_run([]))[-1]
    last = list(ocr_gradio._remake_run(["20261024_000000_책_저자.md"]))[-1]
    assert "완료" in last and "쪽 표지 3개" in last and "원본 보관" in last


def test_list_targets_skips_partial_and_unrefined(tmp_path, monkeypatch):
    res, up = _env(tmp_path, monkeypatch)
    (res / "20261024_000000_가.md").write_text(_MD, encoding="utf-8")
    (res / "20261024_000000_가.txt").write_text("x", encoding="utf-8")
    (res / "20261024_000001_나.txt").write_text(_P1, encoding="utf-8")
    (res / "다_정제중.txt").write_text(_P1, encoding="utf-8")
    (res / "20261024_000002_라.md").write_text("# 라.pdf\n\n- 총 페이지: 3\n\n---\n\n## 페이지 1\n\n본문\n", encoding="utf-8")
    (up / "20261024_000000_가_원문.md").write_text(_RAW_MD, encoding="utf-8")
    rows = {r["path"].name: r for r in remake_md.list_targets(res)}
    assert set(rows) == {"20261024_000000_가.md", "20261024_000001_나.txt", "20261024_000002_라.md"}
    assert rows["20261024_000000_가.md"]["raw"] and not rows["20261024_000001_나.txt"]["raw"]
    assert rows["20261024_000002_라.md"]["skip"] and not rows["20261024_000000_가.md"]["skip"]


def test_rebuild_with_checked_toc_retitles_old_ocr_headings():
    # 옛 결과의 제목 줄은 OCR 차례 글자 — 확인된 차례(책정보 파일)로 다시 맞춘다(1001가지기독교명언)
    old = "# 차례\n\n- 감사의삶\n- 죄\n\n# 감사의삶\n\n감사하는 삶은 더 좋다.\n\n죄\n\n죄는 무섭다.\n"
    toc = [{"level": 1, "title": "감사의 삶", "page": 16}, {"level": 1, "title": "죄", "page": 243}]
    entries, body, note = remake_md.rebuild(old, toc)
    assert "# 감사의 삶" in body.split("\n") and "# 죄" in body.split("\n")
    assert "# 감사의삶" not in body
