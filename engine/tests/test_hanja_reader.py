"""한자 읽기(core.hanja_reader) — 한글 v5 모델이 비워 둔 한자 자리를 중국어 모델로 채운다.

Regression (2026-10-18): 한글 모델 사전에 한자가 0자라 '출사표()'·'무경칠서' 처럼 한자가 빠졌다.
틀린 한자보다 빈칸이 낫다 — 표본 시험에서 틀렸던 사례(孔父嘉→孔父, 一曰→一日)는 넣지 않아야 한다.
그림 조각은 원본 PDF 150dpi 쪽에서 세로 500px 를 잘랐다(짧게 자르면 검출 결과가 달라진다).
"""
import re
from pathlib import Path

import pytest
from PIL import Image

from core import config, hanja_reader, noise_rules, postprocessor, rapidocr_engine

HANJA = Path(__file__).parent / "sample_docs" / "hanja"
needs_ocr = pytest.mark.skipif(not rapidocr_engine.is_available(), reason="RapidOCR 미설치")


def _read(name, on=True, monkeypatch=None):
    monkeypatch.setattr(config, "get_hanja_reading", lambda: on)
    return rapidocr_engine.run(Image.open(HANJA / name), refine=False).text


def _hanja(text):
    return re.findall(r"[一-鿿]+", text)


def _hangul(text):
    return re.sub(r"[^가-힣]", "", text)


@needs_ocr
@pytest.mark.parametrize("name, want", [
    ("paren_jangwon.png", ["縮圖"]),                     # 빈 괄호 '축도()' → '축도(縮圖)'
    # 괄호 없이 붙은 한자 '무경칠서武經七書' · 줄 끝 '계計'(2026-10-18 줄 끝 틈도 읽음)
    ("inline_sonja.png", ["計", "武經七書"]),
    # 一曰 은 日 과 헷갈려 넣지 않는다. 늘어난 글자는 원본과 대조해 모두 일치(2026-10-18 틈 바짝 다시 읽기)
    ("confusable_sonja.png", ["故", "經", "校", "計", "情", "道", "天", "地", "法",
                              "令民與上同意也", "與", "也"]),
    ("partial_gongja.png", ["大司馬"]),                  # 孔父嘉 를 孔父 로 덜 읽음 → 넣지 않는다
    ("plain_ctrl.png", []),                              # 한자 없는 책
])
def test_fills_only_sure_hanja(name, want, monkeypatch):
    on = _read(name, True, monkeypatch)
    assert _hanja(on) == want
    # 한글은 한 글자도 바뀌지 않는다
    assert _hangul(on) == _hangul(_read(name, False, monkeypatch))


@needs_ocr
def test_hanja_inserted_in_place(monkeypatch):
    assert "축도(縮圖)가" in _read("paren_jangwon.png", True, monkeypatch)


def test_merge_puts_hanja_at_gap_and_drops_junk():
    # 한글 모델이 '一' 을 '-' 로 읽은 자리 — 한자를 넣으면 '-' 는 지운다
    k = [(c, 0.99, (x, 0, x + 20, 30)) for c, x in [("법", 0), (")", 20), (":", 40), ("-", 70), ("은", 120)]]
    assert hanja_reader._merge("법): -은", k, [("一曰", 80, 100)]) == "법): 一曰은"


def test_merge_keeps_text_when_chars_do_not_match():
    k = [("가", 0.9, (0, 0, 20, 30))]
    assert hanja_reader._merge("나다", k, [("漢", 30, 30)]) is None


def test_fullwidth_comma_becomes_korean_comma():
    assert hanja_reader._norm("智，信，仁") == "智, 信, 仁"


def test_ks_rejects_simplified():
    assert hanja_reader._ks("勢") and not hanja_reader._ks("势")


@pytest.mark.parametrize("line, keep", [
    ('다섯째는 法이다"라고 해석된다.', "法이다"),
    ("그 첫째는 道이고, 둘째는", "道이고"),
    ("는 較(교)와동의어로서 비교의 의미", "較(교)와"),
    ("은멀고가까움 險易은험함과평탄함", "險易은"),        # OCR 이 뒤 낱말을 붙여 읽음
])
def test_hanja_word_survives_noise_rules(line, keep):
    # 정제(noise_rules)가 한글 어절에 붙은 한자를 잡음으로 떼던 것 — 실제 경로에서 法·道·時制·險易 소실(2026-10-18)
    assert keep in noise_rules._clean_line(line)


@pytest.mark.parametrize("line, want", [
    ("一오스틴은 말했다", "오스틴은 말했다"),
    ("臣麗因주기도문은 좋다", "주기도문은 좋다"),
    ("乂가족이 왔다", "가족이 왔다"),
])
def test_junk_hanja_still_removed(line, want):
    assert noise_rules._clean_line(line) == want


def test_hanja_only_line_survives_margin_noise_filter():
    text = "앞 문장이다.\n夫未戰而廟算勝者, 得算多也.\n뒤 문장이다."
    assert "夫未戰而廟算勝者" in postprocessor.process_full(text, engine="rapidocr")
