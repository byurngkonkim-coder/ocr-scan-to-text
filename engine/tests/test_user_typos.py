"""사용자 오타 사전 — 의심 목록에서 사람이 확인해 등록한 항목만 글자를 바꾼다(intent 2026-10-17)."""
from core import user_typos


def test_registered_pairs_are_applied(tmp_path):
    p = tmp_path / "user_typos.tsv"
    added, rejected = user_typos.add([("흘륨한", "훌륭한"), ("넉녁한", "넉넉한")], p)
    assert added == 2 and rejected == []
    out, n = user_typos.apply("참 흘륨한 사람이다. 넉녁한 마음과 흘륨한 뜻", p)
    assert out == "참 훌륭한 사람이다. 넉넉한 마음과 훌륭한 뜻"
    assert n == 3


def test_apply_twice_gives_same_result(tmp_path):
    p = tmp_path / "user_typos.tsv"
    user_typos.add([("걷모습", "겉모습")], p)
    once, _ = user_typos.apply("걷모습만 보고", p)
    assert user_typos.apply(once, p) == (once, 0)


def test_rejects_unsafe_pairs(tmp_path):
    p = tmp_path / "user_typos.tsv"
    user_typos.add([("걷모습", "겉모습")], p)
    added, rejected = user_typos.add([
        ("이", "의"),            # 1자 — 다른 낱말 속에서 바뀐다
        ("같다", "같다"),        # 바뀌는 것 없음
        ("공부", "공부를"),      # 고칠 말이 틀린 말을 품음 → 돌릴 때마다 다시 바뀜
        ("겉모", "곁모"),        # 기존 고칠 말('겉모습') 속에 들어 있음 → 연쇄
        ("", "빈칸"),
    ], p)
    assert added == 0
    assert len(rejected) == 5
    assert user_typos.load(p) == {"걷모습": "겉모습"}


def test_longest_match_wins_and_no_chaining(tmp_path):
    p = tmp_path / "user_typos.tsv"
    user_typos.add([("지해", "지혜"), ("지해를", "지혜를")], p)
    assert user_typos.apply("지해를 얻고 지해로", p) == ("지혜를 얻고 지혜로", 2)


def test_count_and_missing_file(tmp_path):
    p = tmp_path / "없음.tsv"
    assert user_typos.load(p) == {}
    assert user_typos.apply("그대로", p) == ("그대로", 0)
    assert user_typos.count("흘륨 흘륨 훌륭", "흘륨") == 2


def test_file_is_plain_tsv(tmp_path):
    p = tmp_path / "user_typos.tsv"
    user_typos.add([("바껄", "바꿀")], p)
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if not ln.startswith("#")]
    assert lines == ["바껄\t바꿀"]
