"""hanspell(설치 시) 경로가 문장 조각으로 나눴다 이을 때 공백·문단 줄바꿈을 잃던 것(감사 잠복 1건)."""
import sys
import types

from core import spellcheck


def test_split_and_join_keeps_whitespace(monkeypatch):
    src = "첫 문장이다. 둘째 문장이다!\n\n새 문단이다? 끝."
    assert "".join(spellcheck._split_chunks(src, 10)) == src
    sc = types.SimpleNamespace(check=lambda chunk: types.SimpleNamespace(checked=chunk.strip()))
    monkeypatch.setitem(sys.modules, "hanspell", types.SimpleNamespace(spell_checker=sc))
    monkeypatch.setitem(sys.modules, "hanspell.spell_checker", sc)
    monkeypatch.setattr(spellcheck, "_check_hanspell", lambda: True)
    monkeypatch.setattr(spellcheck.time, "sleep", lambda s: None)
    assert spellcheck.correct_safe(src) == src
