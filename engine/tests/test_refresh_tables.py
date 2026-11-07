"""재처리 결과의 옛 표를 지금 엔진 표로 바꾸기(tools/refresh_tables, 2x2Matrix 2026-11-07)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import refresh_tables as rt   # noqa: E402

OLD_A = "| 제품개발 | 다각화 |\n| --- | --- |\n| 시장침투 | 시장개발 |"
NEW_A = "| 제품 \\ 시장 | 기존 | 신규 |\n| --- | --- | --- |\n| 신규 | 제품개발 | 다각화 |\n| 기존 | 시장침투 | 시장개발 |"
OLD_B = "| ISTJ 집중력의 깊이 사실에 근거 | ISFJ 집중력의 깊이 온정과배려 |\n| --- | --- |\n| ISTP 적응력 | ISFP 적응력 |"
NEW_B = "| 집중력의깊이 사실에근거 | 온정과배려 |\n| --- | --- |\n| 적응력 | 적응력 |"


def _run(tmp_path, monkeypatch, md: str, new_pages: dict) -> tuple[str, dict]:
    monkeypatch.setattr(rt, "OUT_DIR", tmp_path)
    monkeypatch.setattr(rt, "CACHE_DIR", tmp_path / "_표")
    (tmp_path / "_표").mkdir()
    (tmp_path / "_표" / "책.json").write_text(json.dumps(new_pages, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "책.md").write_text(md, encoding="utf-8")
    rep = rt.splice("책")
    return (tmp_path / "책.md").read_text(encoding="utf-8"), rep


def test_matrix_replaced_and_leaked_axis_line_removed(tmp_path, monkeypatch):
    md = f"본문.\n\n그림1.2 제품-시장 매트릭스\n\n{OLD_A}\n기존 신규 시장\n\n다음 본문."
    out, rep = _run(tmp_path, monkeypatch, md, {"12": [NEW_A]})
    assert rep["replaced"] == 1 and rep["leak_lines"] == 1
    assert NEW_A in out and "기존 신규 시장" not in out and "다음 본문." in out


def test_plain_table_with_less_text_kept(tmp_path, monkeypatch):
    # 일반 격자 표는 새 표가 옛 글자를 거의 다 담을 때만 — 'ISTJ' 등이 빠지면 옛 표를 둔다
    md = f"도표 8.2 16가지 MBTI 유형\n\n{OLD_B}\n\n본문."
    out, rep = _run(tmp_path, monkeypatch, md, {"397": [NEW_B]})
    assert rep["replaced"] == 0 and OLD_B in out


def test_order_kept_with_same_named_tables():
    # 같은 이름 표가 두 번(1.2·6.21) — 순서를 지켜 짝짓는다
    olds = ["제품개발다각화", "원가우위차별화", "제품개발다각화"]
    news = ["제품시장기존신규제품개발다각화", "원가우위차별화원가틈새", "제품시장기존신규제품개발다각화"]
    assert rt._align(olds, news) == [(0, 0), (1, 1), (2, 2)]
