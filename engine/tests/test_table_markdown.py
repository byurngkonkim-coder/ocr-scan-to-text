"""표 → Markdown 변환 품질.

배경(2026-10-17 측정): 글자 상자를 '세로로 겹치면 같은 행'으로 묶고 왼쪽부터 이어 붙이던 방식은
병합 머리글('2025년' 이 두 칸)을 한 행으로 섞고, 여러 줄 칸('업무 효율화와 / 비용 절감')을 두 칸으로
가르며 줄 순서까지 뒤집었다(정답 9장 중 7장 정확, 열 수가 행마다 다른 표 12장 중 4장).
MyOCR 과 같은 표 구조 모델(RapidTable SLANETPLUS) + 열 기준 칸 상자 보정으로 바꿨다.
"""
from pathlib import Path

from PIL import Image

from core import figure_detector

_DOCS = Path(__file__).parent / "sample_docs"


def _grid(name):
    md = figure_detector._extract_table_markdown(Image.open(_DOCS / name).convert("RGB"), "kor+eng")
    rows = [l for l in md.splitlines() if not l.startswith("| ---")]
    return [[c.strip() for c in l.strip().strip("|").split("|")] for l in rows]


def test_simple_table_exact():
    assert _grid("table_simple.png") == [["이름", "부서", "직급", "연봉"],
                                         ["김철수", "영업부", "과장", "5500"],
                                         ["이영희", "개발팀", "대리", "4800"]]


def test_merged_header_keeps_rows_and_columns():
    g = _grid("table_merged_header.png")
    assert g == [["구분", "2025년", "", "2026년", ""],
                 ["", "상반기", "하반기", "상반기", "하반기"],
                 ["매출", "120", "135", "142", "150"],
                 ["비용", "80", "88", "91", "95"]], g


def test_multiline_cell_is_one_cell_in_order():
    g = _grid("table_multiline_cell.png")
    assert ["목적", "업무 효율화와 비용 절감"] in g, g
    assert ["기간", "2026년 1월부터 12월까지"] in g, g
    assert len({len(r) for r in g}) == 1, g   # 모든 행의 열 수가 같아야 Markdown 표가 깨지지 않는다


def test_pipe_in_cell_escaped():
    assert figure_detector._grid_to_markdown([["A|B", "C"], ["1", "2"]]).splitlines()[0] == "| A\\|B | C |"


def test_body_fragment_grid_is_not_a_table():
    # 예측기계 135쪽 실측: 본문 조각이 빈 칸으로 채운 격자('| 다시만나도… |  |')로 나와 가짜 표가 됐다.
    # 글자 있는 칸이 2개 이상인 행이 절반 미만이면 표가 아니다
    fake = "| 다시만나도상대방의얼 모르는것도앞서설명했 |  |\n| --- | --- |\n|  |  |\n| 그러나 |  |\n|  | 말이다 |"
    canvas = "| 예측 | 판단 |  | 행동 |\n| --- | --- | --- | --- |\n|  |  |  |  |\n| 입력 |  | 트레이닝 | 피드백 |\n|  |  |  |  |"
    assert not figure_detector._looks_like_table(fake)
    assert figure_detector._looks_like_table(canvas)   # 185쪽 빈 캔버스: 빈 행이 있어도 표
    assert figure_detector._looks_like_table("| 이름 | 부서 |\n| --- | --- |\n| 김철수 | 영업부 |")
