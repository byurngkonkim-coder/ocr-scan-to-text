"""디지털 PDF(텍스트 레이어) 내장 규칙 정제 — 2026-10-19 DBR 기사 일괄 처리 실측.

'회사전략에 맞는 리더십을 발굴하라 - DBR.PDF': 내장 글자는 단락마다 빈 줄('  ')로 나뉘고 쪽마다
'2페이지' 꼬리말이 첫 줄에 있다. 정제 결과에 '2페이지'가 남고, 단락 사이 빈 줄이 모두 사라졌다(C-08 위반).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import config, noise_rules, text_cleaner


def test_clean_page_keeps_paragraph_blank_line():
    page = "첫 단락 문장입니다. \n  \n\n둘째 단락 문장입니다."
    assert noise_rules.clean_page(page) == "첫 단락 문장입니다.\n\n둘째 단락 문장입니다."


def test_page_label_line_removed_at_page_edge_only():
    pages = ["2페이지\n본문 문장입니다.", "본문은 3페이지 참조.\n4페이지", "1968\n연도 줄은 남긴다."]
    assert text_cleaner.strip_running_headers(pages) == [
        "본문 문장입니다.", "본문은 3페이지 참조.", "1968\n연도 줄은 남긴다."]


def test_topic_word_at_paragraph_start_is_not_header():
    # '더 원대한 포부를 가진 리더 - DBR': 단락 첫머리에 자주 오는 주제어가 깨진 머리글로 지워졌다
    body = ["가계 대출이 늘었다", "기업 금융을 키웠다", "정부 지원을 받지 않았다", "시장 점유율이 올랐다",
            "고객 신뢰가 깊어졌다", "직원 몰입도가 높았다", "주주 배당을 지켰다", "지점 수를 줄였다"]
    pages = [f"스탠다드차타드은행은 {a}.\n{b}.\n둘째 단락에서도 스탠다드차타드은행은 {a}.\n"
             f"셋째 단락에서도 스탠다드차타드은행은 {b}.\n{c}.\n{a}."
             for a, b, c in zip(body, body[1:] + body[:1], body[2:] + body[:2])]
    out, _ = noise_rules.strip_fuzzy_headers(pages)
    assert out == pages


def _clean(pages):
    g = text_cleaner.iter_clean(pages, drop_front_matter=True, local=True)
    try:
        while True:
            next(g)
    except StopIteration as s:
        return s.value


def test_digital_pages_rules_cleanup_paragraphs_and_page_labels(monkeypatch):
    monkeypatch.setattr(config, "get_local_cleanup_engine", lambda: "rules")
    pages = [
        "회사전략에 맞는 리더십을 발굴하라\n박광서\n율도테크의 홍길동 씨는 최근 부장으로 승진하면서 팀장 보직을 맡았다."
        " 자리에 걸맞은 성과를 내야 한다.\n  \n홍 팀장은 일명 ‘사오정’ 세대다. 임원으로 승진하지 못하면 커리어를"
        " 정리해야 할지도 모른다.",
        "2페이지\n홍길동 팀장이 당황해하는 기색을 보이자 허 상무는 생각의 출발점부터 제대로 잡아야 한다고 말했다.\n"
        "  \n그에 따르면 회사가 어떤 전략적 방향을 추구하는지에 따라서 요구하는 리더십 또한 달라진다.",
    ]
    out = _clean(pages)
    assert "페이지" not in out
    paras = out.strip().split("\n\n")
    assert paras[-1].startswith("그에 따르면")
    assert any(p.startswith("홍 팀장은") for p in paras)
    assert any(p.startswith("홍길동 팀장이") for p in paras)
