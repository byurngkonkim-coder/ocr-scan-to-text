"""2×2 매트릭스 그림 → 축 이름을 머리행·머리열에 둔 Markdown 표 (2x2Matrix 알렉스로위, 2026-10-21).

배경: 바깥 상자 + 십자선 4분면 그림을 표 구조 모델이 읽으면 한 칸 안 여러 줄이 행으로 쪼개졌고
(그림1.5·6.4), 상자 밖 축 글자는 본문으로 새어 세로쓰기가 「R영올」 같은 잡음이 됐다.
정답: 원본 PDF 150dpi 렌더를 눈으로 옮겨 적은 10장(docs/intents/2026-10-21-2x2매트릭스_표인식개선.md §2).
합격선: 축 일치 8장 이상 · 모양 10장 모두 3×3 · 칸 글자 일치(글자 F1) 0.9 이상.
"""
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from core import figure_detector as fd
from core import postprocessor

_DOCS = Path(__file__).parent / "sample_docs"

# 쪽: (모서리 「세로축 \ 가로축」, [가로축 왼쪽·오른쪽 끝], [세로축 위·아래 끝], [왼위, 오위, 왼아래, 오아래])
TRUTH = {
    19: ("사업투자\\수익성추구", ["아니오", "예"], ["예", "아니오"],
         ["이익보다성장추구", "최적화", "경영실패", "현금유입"]),
    48: ("중요성\\긴급성", ["높다", "낮다"], ["높다", "낮다"],
         ["위기마감일급박한문제", "계획예방인간관계", "많은메일끼어드는일급한일", "사소한일바쁜업무시간낭비요소"]),
    129: ("가격\\품질", ["낮다", "높다"], ["높다", "낮다"],
          ["시장점유율하락영역경쟁자2", "경쟁자1", "", "FTXS시장점유율상승영역"]),
    130: ("차별화된가치\\원가절감", ["낮다", "높다"], ["사업차별화", "차별화없음"],
          ["IBMNCR", "FTXS", "", ""]),
    131: ("제조업체\\제품", ["독창적", "모방"], ["소매상에잘알려짐", "소매상에안알려짐"],
          ["성공가능성높음NCR후지츠", "영업노력에따라다름IBM", "영업노력에따라다름", "성공가능성희박"]),
    139: ("제품\\고객", ["기존", "신규"], ["신규", "기존"],
          ["새로운부품과솔루션니즈정의", "조심스런접근", "마케팅,인지도강화", "신규고객에게마케팅,시장확대"]),
    162: ("패러다임충격\\제품의혁신성", ["낮다", "높다"], ["높다", "낮다"],
          ["시제품순수과학기술마니아", "초기시장캐즘선각자", "보수주의자수명주기의끝주류시장", "실용주의자볼링레인토네이도"]),
    223: ("경쟁의범위\\경쟁우위", ["낮은원가", "차별화"], ["넓은시장", "좁은시장"],
          ["포드1910현대1990", "GM1920폭스바겐비틀1960~1980", "현대1980~2003", "도요타와혼다1990볼보,코르벳,로버지속적임"]),
    245: ("주의력\\선택", ["파고드는", "자발적"], ["의식의전면", "의식의이면"],
          ["고객경쟁사", "혁신대인관계문제", "팀워크물류리더십", "가족"]),
    287: ("규율의문화\\기업가윤리", ["낮다", "높다"], ["높다", "낮다"],
          ["계층조직", "위대한조직크로커", "A&P관료조직", "창업조직"]),
}


def _nospace(s: str) -> str:
    return "".join(s.split())


@pytest.fixture(scope="module")
def grids():
    out = {}
    for p in TRUTH:
        tables = [r for r in fd.detect(Image.open(_DOCS / f"matrix_p{p}.png").convert("RGB")).regions
                  if r.kind == "table"]
        rows = [l for l in tables[0].table_md.splitlines() if not l.startswith("| ---")] if tables else []
        out[p] = [[c.strip() for c in l.strip()[1:-1].split(" | ")] for l in rows]
    return out


def test_every_matrix_is_3x3(grids):
    assert {p: [len(r) for r in g] for p, g in grids.items()} == {p: [3, 3, 3] for p in TRUTH}


def test_axis_labels_in_header(grids):
    ok = [p for p, (corner, cols, rows, _c) in TRUTH.items()
          if grids[p] and _nospace(grids[p][0][0]) == corner
          and [_nospace(c) for c in grids[p][0][1:]] == cols
          and [_nospace(grids[p][1][0]), _nospace(grids[p][2][0])] == rows]
    assert len(ok) >= 8, ok


def test_cell_text_f1(grids):
    tp = got_n = want_n = 0
    for p, (_a, _b, _c, cells) in TRUTH.items():
        got = Counter(_nospace("".join(grids[p][1][1:] + grids[p][2][1:])))
        want = Counter("".join(cells))
        tp += sum((got & want).values())
        got_n += sum(got.values())
        want_n += sum(want.values())
    f1 = 2 * tp / (got_n + want_n)
    assert f1 >= 0.9, f1


def test_axis_text_masked_from_body():
    """축 글자 띠까지 표 영역 — 본문 판독(마스킹 이미지)에 축 글자가 남지 않는다."""
    img = Image.open(_DOCS / "matrix_p129.png").convert("RGB")
    det = fd.detect(img)
    left = fd._rough_ocr(det.masked_image, "kor+eng")[0]
    assert "품질" not in left and "가격" not in left, left


def test_grid_table_is_not_matrix():
    """일반 격자 표(안쪽 선 여러 개)는 매트릭스 경로를 타지 않는다."""
    img = Image.open(_DOCS / "table_simple.png").convert("RGB")
    gray = fd._to_gray_cv(img)
    assert fd._matrix_markdown(img, gray, (0, 0, img.width, img.height), "kor+eng") is None


@pytest.mark.parametrize("caption", ["◆ 그림 0.1 이익 계획 매트릭스", "그림5.5경쟁력인지 지도"])
def test_figure_caption_not_joined_to_paragraph(caption):
    body = "그러나 두려움, 욕심 이런 것들이 판단을 흐리게 만들어 결국 사업을 위태롭게 하는 경우가 많고 많다"
    out = postprocessor.join_broken_lines(f"{body}\n{body}\n{caption}").splitlines()
    assert out[-1] == caption


def test_figure_reference_line_still_joined():
    body = "구매 가능성은 사이먼 마자로의 연구에서 영감을 받았는데 고객이 구매를 결정하는 두 가지 기준은"
    out = postprocessor.join_broken_lines(f"{body}\n{body}\n그림 5.6의 구매 가능성은 제품과 명성이다.")
    assert len(out.splitlines()) == 1
