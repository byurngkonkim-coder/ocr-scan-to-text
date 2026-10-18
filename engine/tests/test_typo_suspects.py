"""의심 목록 — Kiwi 오타 모드 후보를 본문은 바꾸지 않고 목록으로만(intent 2026-10-17).

실측(『지혜의 기술』 1권): 오타 모드가 바꾼 42곳 중 맞게 고친 것은 약 10곳 → 자동 적용 금지, 후보로만.
"""
import pytest

from core import spacing, typo_suspects, user_typos

pytestmark = pytest.mark.skipif(not spacing.is_available(), reason="kiwipiepy 미설치")


def test_finds_suspects_with_count_and_context():
    text = "때는즉시지해나침반에의지하라.\n걷모습만 보고 속지 마라. 걷모습을 꾸민다"
    rows = typo_suspects.find(text, user_dict={})
    pairs = {(r["wrong"], r["suggestion"]): r for r in rows}
    assert ("걷모습", "겉모습") in pairs
    assert ("지해", "지혜") in pairs
    assert pairs[("걷모습", "겉모습")]["count"] == 2
    assert "걷모습만 보고" in pairs[("걷모습", "겉모습")]["context"]
    assert rows[0]["wrong"] == "걷모습"          # 많이 나온 것부터


def test_registered_words_are_not_listed():
    rows = typo_suspects.find("걷모습만 보고", user_dict={"걷모습": "겉모습"})
    assert all(r["wrong"] != "걷모습" for r in rows)


def test_correct_text_gives_no_suspects():
    assert typo_suspects.find("그는 회사에 갔다. 우리는 그 일을 끝냈다.", user_dict={}) == []


def test_save_writes_tsv_under_results_subfolder(tmp_path):
    result = tmp_path / "책A.txt"
    result.write_text("걷모습만 보고", encoding="utf-8")
    out, n = typo_suspects.save(result, user_dict={})
    assert out == tmp_path / typo_suspects.SUBDIR / "책A.tsv"
    assert n >= 1
    assert result.read_text(encoding="utf-8") == "걷모습만 보고"     # 본문은 그대로
    rows = typo_suspects.read(out)
    assert rows[0]["wrong"] == "걷모습" and rows[0]["suggestion"] == "겉모습"


def test_suspects_use_own_kiwi_not_spacing_one():
    # 같은 Kiwi 로 space() 뒤 tokenize(typos=) 를 부르면 프로세스가 죽는다(kiwipiepy 0.24 실측)
    assert typo_suspects._get() is not spacing._get()
