"""Anthropic 응답 종료 상태 가드 (조용한 절단·거부 회귀 방지).

`test_gemini_response_guard.py` 의 Anthropic 경로 대응물이다.

실측 경위(2026-08-02): 2026-07-07 RECITATION 소실 사고를 조치할 때 원인이었던
Gemini 경로(`gemini_vision.response_ok`)에만 가드를 넣었고, **같은 구조의
Anthropic 경로 3곳이 약 1개월간 무방비**로 남아 있었다. Anthropic 은
stop_reason=max_tokens(토큰 상한 절단)·refusal(거부) 에서 예외 없이 잘린
텍스트를 정상처럼 반환하므로 동일한 조용한 소실이 가능하다.

교훈(EX-G02): 사고를 고칠 때 원인이 된 제공자만 고치면 다른 경로가 남는다.
가드뿐 아니라 **테스트도 제공자별로** 있어야 한다 — 이 파일이 그 완결이다.
"""
from types import SimpleNamespace

import pytest

from core import claude_vision, llm_corrector


def _resp(stop_reason=None):
    """stop_reason 만 가진 최소 응답 목."""
    return SimpleNamespace(stop_reason=stop_reason)


class TestClaudeVisionResponseOk:
    """core.claude_vision.response_ok — 비전 OCR 경로."""

    def test_end_turn_is_accepted(self):
        ok, sr = claude_vision.response_ok(_resp("end_turn"))
        assert ok and sr == "end_turn"

    def test_stop_sequence_is_accepted(self):
        ok, sr = claude_vision.response_ok(_resp("stop_sequence"))
        assert ok and sr == "stop_sequence"

    def test_max_tokens_is_rejected(self):
        # 토큰 상한 절단 — 텍스트가 잘렸는데 예외가 없다. 소실의 직접 원인.
        ok, sr = claude_vision.response_ok(_resp("max_tokens"))
        assert not ok and sr == "max_tokens"

    def test_refusal_is_rejected(self):
        # 모델 거부 — Gemini 의 RECITATION 에 해당하는 Anthropic 측 사유
        ok, sr = claude_vision.response_ok(_resp("refusal"))
        assert not ok and sr == "refusal"

    def test_pause_turn_is_rejected(self):
        ok, _ = claude_vision.response_ok(_resp("pause_turn"))
        assert not ok

    def test_unknown_shape_is_tolerated(self):
        # SDK 구조가 바뀌어 stop_reason 판단이 불가하면 차단하지 않는다
        # (안전측: 오탐으로 정상 페이지를 전부 실패시키는 것보다 통과가 낫다)
        ok, sr = claude_vision.response_ok(_resp(None))
        assert ok and sr == "UNKNOWN"

    def test_missing_attribute_is_tolerated(self):
        # stop_reason 속성 자체가 없는 응답도 통과 (동일한 안전측 판단)
        ok, sr = claude_vision.response_ok(SimpleNamespace())
        assert ok and sr == "UNKNOWN"


class TestLLMCorrectorCheckStop:
    """core.llm_corrector._check_stop — 텍스트 정제 경로."""

    def test_end_turn_passes_silently(self):
        assert llm_corrector._check_stop(_resp("end_turn")) is None

    def test_stop_sequence_passes_silently(self):
        assert llm_corrector._check_stop(_resp("stop_sequence")) is None

    def test_max_tokens_raises(self):
        with pytest.raises(RuntimeError, match="max_tokens"):
            llm_corrector._check_stop(_resp("max_tokens"))

    def test_refusal_raises(self):
        with pytest.raises(RuntimeError, match="refusal"):
            llm_corrector._check_stop(_resp("refusal"))

    def test_unknown_shape_is_tolerated(self):
        # 호출부는 예외 시 원본 청크를 유지한다. 판단 불가면 통과시켜
        # 정상 정제까지 원본으로 되돌리는 과잉 폴백을 막는다.
        assert llm_corrector._check_stop(_resp(None)) is None


class TestGuardParity:
    """두 경로가 같은 판정을 내리는지 — 한쪽만 방어되는 상황의 재발 방지."""

    @pytest.mark.parametrize("sr", ["end_turn", "stop_sequence"])
    def test_accepted_reasons_match(self, sr):
        assert claude_vision.response_ok(_resp(sr))[0] is True
        assert llm_corrector._check_stop(_resp(sr)) is None

    @pytest.mark.parametrize("sr", ["max_tokens", "refusal", "pause_turn"])
    def test_rejected_reasons_match(self, sr):
        assert claude_vision.response_ok(_resp(sr))[0] is False
        with pytest.raises(RuntimeError):
            llm_corrector._check_stop(_resp(sr))


class TestLlmCorrectorLengthGuard:
    """core.llm_corrector.correct — 정상 종료여도 분량이 틀리면 원본 유지 (2026-09-27)."""

    @pytest.fixture(autouse=True)
    def _key(self, monkeypatch):
        monkeypatch.setattr(llm_corrector, "get_api_key", lambda: "test-key")
        llm_corrector.reset_call_count()

    def test_dropped_sentences_keep_original(self, monkeypatch):
        # 문장을 빼먹은 응답(분량 50%) — end_turn 이라 종료 사유 가드로는 못 잡는다
        src = "첫째 문장입니다. 둘째 문장입니다.\n"
        monkeypatch.setattr(llm_corrector, "_call_api", lambda c, k: "첫째 문장입니다.")
        assert llm_corrector.correct(src) == src

    def test_added_commentary_keeps_original(self, monkeypatch):
        src = "교정할 문장입니다.\n"
        monkeypatch.setattr(llm_corrector, "_call_api",
                            lambda c, k: "교정할 문장입니다. (설명: 오타가 없어 그대로 두었습니다)")
        assert llm_corrector.correct(src) == src

    def test_correction_accepted_and_chunk_newlines_kept(self, monkeypatch):
        # 청크 두 개 — 모델이 끝 줄바꿈을 지워도 청크 경계 줄이 붙지 않아야 한다
        monkeypatch.setattr(llm_corrector, "_CHUNK_CHARS", 10)
        monkeypatch.setattr(llm_corrector, "_call_api", lambda c, k: c.strip().replace("햇삻", "햇살"))
        out = llm_corrector.correct("창문의 햇삻이\n들어온다\n")
        assert out == "창문의 햇살이\n들어온다\n"


def _fake_client(resp, captured=None):
    def create(**kw):
        return resp
    return SimpleNamespace(messages=SimpleNamespace(create=create))


def test_claude_vision_empty_text_is_rejected(monkeypatch):
    # end_turn 인데 글자가 없는 응답을 신뢰도 95로 받아들여 쪽이 비던 것 — 예외로 내서 로컬 결과·검토 필요로
    from PIL import Image
    resp = SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(text="  \n")],
                           usage=SimpleNamespace(input_tokens=1, output_tokens=1))
    monkeypatch.setattr(claude_vision, "_get_client", lambda: _fake_client(resp))
    with pytest.raises(RuntimeError):
        claude_vision.run(Image.new("RGB", (10, 10)))


def test_anthropic_clients_have_timeout(monkeypatch):
    # 시간 제한이 없으면 SDK 기본(10분 × 재시도)으로 한 쪽에 최대 약 30분 멈췄다
    anthropic = pytest.importorskip("anthropic")
    seen = []
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: seen.append(kw) or object())
    monkeypatch.setattr(claude_vision, "_client", None)
    monkeypatch.setattr("core.config.get_api_key", lambda: "test-key")
    claude_vision._get_client()
    monkeypatch.setattr(llm_corrector, "_client", None)
    llm_corrector._get_client("test-key")
    assert len(seen) == 2 and all(0 < kw.get("timeout", 0) <= 180 for kw in seen)
