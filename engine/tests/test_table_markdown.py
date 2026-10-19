"""스캔 표 → Markdown 표 복원과 본문 제자리 배치 (SI 영업 및 수행에 대한 운영방안, 2026-10-19)."""
from types import SimpleNamespace

from core import figure_detector as fd
from core import rapidocr_engine, writer
from core.types import OutputFormat

COLS = [(0, 100), (100, 200), (200, 300)]


def _line(cy, *cells):
    """cells: (열 번호, 글자) — 글자 높이 20 상자를 그 열 가운데에 둔다."""
    return [cy, [(t, (COLS[j][0] + 10, cy - 10, 80, 20)) for j, t in cells]]


def test_tight_full_rows_stay_separate():
    # 줄 간격이 글자 높이의 1.3배여도 칸을 다 채운 줄은 각자 행이다(p5 인벤토리 15행이 2행으로 뭉쳤다)
    lines = [_line(30 + 26 * i, (0, f"사업{i}"), (1, "3월"), (2, "8,000")) for i in range(4)]
    rows = fd._group_rows(lines, COLS, [], unit=20)
    assert len(rows) == 4


def test_continuation_line_joins_its_row_unless_ruled():
    lines = [_line(30, (0, "1"), (1, "가득"), (2, "수행 중인 상태")),
             _line(55, (2, "내부품의 결재 후 투입")),
             _line(100, (0, "2"), (1, "가동"), (2, "제안 중"))]
    rows = fd._group_rows(lines, COLS, [], unit=20)
    assert [len(r) for r in rows] == [4, 3]
    # 사이에 가로선이 있으면 붙이지 않는다
    rows = fd._group_rows(lines, COLS, [(43, 0, 300)], unit=20)
    assert len(rows) == 3


def test_merged_label_on_rule_goes_to_next_row():
    # 세로 병합 칸 이름(「목표 및」)이 행 경계선에 걸치면 다음 행 첫 칸으로
    lines = [_line(30, (0, "역"), (1, "수주목표"), (2, "252억")),
             _line(56, (0, "목표및")),
             _line(84, (1, "도전목표"), (2, "378억"))]
    rows = fd._group_rows(lines, COLS, [(55, 100, 300)], unit=20)
    assert len(rows) == 2
    assert "목표및" in fd._fill_cells(rows[1], COLS)[0]


def test_lone_column_folds_left():
    grid = [["프로젝트명", "", "영업대표"], ["농어촌", "", "가나다"], ["", "합계", "253,600"]]
    fd._fold_lone_columns(grid)
    assert grid == [["프로젝트명", "영업대표"], ["농어촌", "가나다"], ["합계", "253,600"]]


def test_insert_anchors_by_height():
    boxes = [[[0, y], [10, y], [10, y + 10], [0, y + 10]] for y in (100, 500, 900)]
    out = rapidocr_engine._insert_anchors(["위", "가운데", "아래"], boxes, 1000,
                                          [(0.3, "⟨TABLE:1-1⟩"), (0.95, "⟨TABLE:1-2⟩")])
    assert out == ["위", "", "⟨TABLE:1-1⟩", "", "가운데", "아래", "", "⟨TABLE:1-2⟩", ""]


def _page(num, text, figures):
    return SimpleNamespace(page_num=num, corrected_text=text, figures=figures)


def test_place_tables_in_body_and_keeps_lost_ones():
    md = "| a | b |\n| --- | --- |\n| 1 | 2 |"
    pages = [_page(2, "", [f"[표 1]\n{md}", "[그림 1: 이미지 감지됨]"]),
             _page(5, "", [f"[표 2]\n{md}"])]
    text = "앞 문단.\n\n⟨TABLE:2-1⟩\n\n뒤 문단."
    out = writer.place_tables(text, pages)
    assert out.startswith("앞 문단.\n\n| a | b |")
    assert "⟨TABLE" not in out
    assert out.endswith("[표 2 (p.5)]\n" + md)   # 표지가 사라진 표도 글자를 잃지 않는다



def test_write_txt_puts_table_in_body_not_in_nontext_list(tmp_path):
    md = "| a | b |\n| --- | --- |\n| 1 | 2 |"
    page = SimpleNamespace(page_num=1, corrected_text="본문\n\n⟨TABLE:1-1⟩", figures=[f"[표 1]\n{md}"],
                           is_blank=False, needs_review=False)
    res = SimpleNamespace(pages=[page], total_pages=1, source_path=tmp_path / "x.pdf", output_path=None)
    out = writer.write(res, tmp_path, OutputFormat.TXT).read_text(encoding="utf-8")
    assert f"본문\n\n{md}" in out and "비텍스트 영역" not in out