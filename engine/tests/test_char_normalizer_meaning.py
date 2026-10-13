# Regression: ISSUE-002 — L1 기호 치환이 목록 기호 '(c)'를 '©'로, 'No, '를 'No. '로 바꿔
#   스캔본 본문의 뜻을 바꾸던 문제
# Found by /qa on 2026-10-13
# Report: .gstack/qa-reports/run-20261013T070910Z/qa-report-ocr-dashboard-2026-10-13.md
"""L1 기호 치환은 저작권 표기·번호 표기에만 적용하고 본문 뜻은 바꾸지 않는다."""
import pytest

from core import char_normalizer


@pytest.mark.parametrize("raw, expected", [
    # 목록 기호 (a)(b)(c) — 저작권 기호가 아니다
    ("(b) 타인의 권리\n(c) 법령을 위반하지 않을 것", "(b) 타인의 권리\n(c) 법령을 위반하지 않을 것"),
    ("다음 (a), (b), (c) 중 하나", "다음 (a), (b), (c) 중 하나"),
    # 영어 감탄·대답의 쉼표
    ("No, 그는 동의하지 않았다.", "No, 그는 동의하지 않았다."),
    # 기존 계약 유지 — 저작권 연도·Copyright 표기, 번호 표기
    ("(c) 2026 저작권자", "© 2026 저작권자"),
    ("Copyright (C) 홍길동", "Copyright © 홍길동"),
    ("No, 5 항목", "No. 5 항목"),
])
def test_symbol_map_keeps_meaning(raw, expected):
    assert char_normalizer.normalize_all(raw) == expected
