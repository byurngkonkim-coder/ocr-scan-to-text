"""Gemini 응답 종료 상태 가드 (RECITATION 소실 사고 회귀 방지).

실측 사고(2026-07-07): 저작권 출판물(메시지성경 신약) 스캔에서 Gemini가
finish_reason=RECITATION 으로 예외 없이 빈 텍스트를 반환 → 신뢰도 95%로
수용되어 842p 중 167p가 통째로 소실. STOP 이외의 종료는 수용하지 않고
예외를 올려 Tesseract 폴백(비전)·원문 보존 강등(정제)이 작동해야 한다.
"""
from types import SimpleNamespace

from core import gemini_vision


def _resp(finish=None, candidates=True):
    if not candidates:
        return SimpleNamespace(candidates=[])
    return SimpleNamespace(candidates=[SimpleNamespace(finish_reason=finish)])


class TestResponseOk:
    def test_stop_is_accepted(self):
        ok, fr = gemini_vision.response_ok(_resp("FinishReason.STOP"))
        assert ok and fr == "STOP"

    def test_recitation_is_rejected(self):
        ok, fr = gemini_vision.response_ok(_resp("FinishReason.RECITATION"))
        assert not ok and fr == "RECITATION"

    def test_safety_and_max_tokens_rejected(self):
        assert not gemini_vision.response_ok(_resp("FinishReason.SAFETY"))[0]
        assert not gemini_vision.response_ok(_resp("FinishReason.MAX_TOKENS"))[0]

    def test_no_candidates_rejected(self):
        # 프롬프트 차단(prompt_feedback) 시 후보 자체가 없음 — 비정상
        ok, fr = gemini_vision.response_ok(_resp(candidates=False))
        assert not ok and fr == "NO_CANDIDATES"

    def test_unknown_shape_is_tolerated(self):
        # SDK 구조가 바뀌어 finish_reason 판단이 불가하면 차단하지 않는다(안전측:
        # 오탐으로 정상 페이지를 전부 폴백시키는 것보다 통과가 낫다)
        ok, fr = gemini_vision.response_ok(_resp(finish=None))
        assert ok and fr == "UNKNOWN"

    def test_plain_string_finish_reason(self):
        # enum 이 아니라 문자열로 오는 경우도 처리
        ok, fr = gemini_vision.response_ok(_resp("RECITATION"))
        assert not ok and fr == "RECITATION"


# Regression: ISSUE-A — STOP(정상 종료)인데 본문이 빈 응답이 신뢰도 95%로 수용되어
# 쪽이 경고 없이 사라지던 결함. Found by /qa on 2026-10-03
# Report: .gstack/qa-reports/qa-report-ocr-2026-10-03.md
class TestEmptyTextRejected:
    def _fake_client(self, text):
        resp = SimpleNamespace(
            text=text, usage_metadata=None,
            candidates=[SimpleNamespace(finish_reason="FinishReason.STOP")])
        models = SimpleNamespace(generate_content=lambda **kw: resp)
        return SimpleNamespace(models=models)

    def _run(self, monkeypatch, text):
        from PIL import Image
        monkeypatch.setattr(gemini_vision, "_get_client", lambda: self._fake_client(text))
        return gemini_vision.run(Image.new("RGB", (40, 40), "white"))

    def test_empty_text_raises_for_fallback(self, monkeypatch):
        import pytest
        with pytest.raises(Exception):
            self._run(monkeypatch, "   \n")

    def test_normal_text_accepted(self, monkeypatch):
        assert self._run(monkeypatch, "본문입니다.").text == "본문입니다."

    # 2026-10-05 드러커 471쪽: 이미지 대신 '디지털 전환' 보고서를 지어내 쪽 내용이 사라짐
    def test_fabricated_report_raises_for_fallback(self, monkeypatch):
        import pytest
        fake = ("**모든 성공적인 프로젝트를 수행할 수 있는가**\n\n"
                "• 프로젝트 목표\n• 프로젝트 범위\n• 프로젝트 예산\n")
        with pytest.raises(Exception):
            self._run(monkeypatch, fake)

    def test_real_bullet_list_accepted(self, monkeypatch):
        page = "경영자의 과업은 다음과 같다.\n• 마케팅\n• 혁신\n• 생산성\n"
        assert self._run(monkeypatch, page).text == page
