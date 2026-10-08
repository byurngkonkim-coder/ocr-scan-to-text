"""Ollama 경로 종료 사유 가드 (G01·G03) — gemini/anthropic 가드 테스트의 대응물"""
import io
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import config, ollama_vision, vision
from core.types import OCREngine


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_urlopen(payload):
    return lambda req, timeout=None: _Resp(json.dumps(payload).encode())


@pytest.mark.parametrize("dr,ok", [("stop", True), (None, True), ("length", False), ("load", False)])
def test_response_ok(dr, ok):
    assert ollama_vision.response_ok({"done_reason": dr})[0] is ok


def test_length_is_rejected(monkeypatch):
    monkeypatch.setattr(ollama_vision.urllib.request, "urlopen",
                        _fake_urlopen({"response": "잘린 텍스트", "done_reason": "length"}))
    with pytest.raises(RuntimeError, match="done_reason=length"):
        ollama_vision.run(Image.new("RGB", (40, 40), "white"))


def test_empty_is_rejected(monkeypatch):
    monkeypatch.setattr(ollama_vision.urllib.request, "urlopen",
                        _fake_urlopen({"response": "  ", "done_reason": "stop"}))
    with pytest.raises(RuntimeError, match="빈 응답"):
        ollama_vision.run_text("본문")


def test_ok_returns_ollama_result(monkeypatch):
    monkeypatch.setattr(ollama_vision.urllib.request, "urlopen",
                        _fake_urlopen({"response": "정상 본문", "done_reason": "stop"}))
    r = ollama_vision.run(Image.new("RGB", (40, 40), "white"))
    assert r.text == "정상 본문" and r.engine == OCREngine.OLLAMA


def test_facade_routes_to_ollama(monkeypatch):
    monkeypatch.setattr(config, "get_vision_provider", lambda: "ollama")
    assert vision.provider_name() == "ollama"
    assert vision._impl() is ollama_vision
    vision.set_session_key("무시될 키")  # 로컬 엔진은 키를 환경변수에 쓰지 않는다
