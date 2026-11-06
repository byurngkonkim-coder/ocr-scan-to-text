"""OCR 혼동 글자 후보 — Kiwi TypoTransformer + 점수 차(intent 2026-11-06-로컬오탈자-후처리교정).

표본은 『악당의 명언』 OCR 결과에서 실제로 나온 오인식. 본문은 바꾸지 않고 후보만 낸다.
"""
import pytest

from core import ocr_typos, spacing, typo_suspects

pytestmark = pytest.mark.skipif(not spacing.is_available(), reason="kiwipiepy 미설치")


def _pairs(*lines):
    return {(w, s) for w, s, _ in ocr_typos.candidates(list(lines))}


@pytest.mark.parametrize("line, pair", [
    ("구매자에게 구메하고 소유하는 기쁨", ("구메", "구매")),
    ("곳 심에 한게가 있다", ("한게", "한계")),
    ("마케팅은 하눌에 돈 뿌리기", ("하눌", "하늘")),
    ("두려움과 훙분이 교차한다", ("훙분", "흥분")),
    ("이 사용하는 SNS에 마옴에 드는 명언", ("마옴", "마음")),
])
def test_finds_ocr_shape_confusions(line, pair):
    assert pair in _pairs(line)


def test_one_syllable_fix_is_widened_to_two_letters():
    # 사용자 사전은 2자 이상만 받는다(user_typos) — '올→을' 은 앞 글자를 붙여 '힘올→힘을'
    assert ("힘올", "힘을") in _pairs("조직도 힘올 얻는다")


def test_candidate_differs_in_exactly_one_letter_and_is_hangul_only():
    # 실측 잡음: 용언 원형 '혼들린→흔들리'(두 글자 다름), 기호를 붙인 'n혔'·',스'
    for w, s, _ in ocr_typos.candidates(["모두 같은 이야기를 하면 혼들린다.",
                                         "Competitio 스식n혔 경쟁", "사라지지 않을 것이다. ,스 헤라클레스도"]):
        assert sum(x != y for x, y in zip(w, s)) == 1
        assert w.isalpha() and all("가" <= c <= "힣" for c in w)


@pytest.mark.parametrize("line", ["특허 기간이 지나버리는 경우","궁금한 것을 참는 사람",
                                  "그는 회사에 갔다. 우리는 그 일을 끝냈다."])
def test_correct_text_gives_no_candidates(line):
    assert _pairs(line) == set()


def test_candidates_carry_score_margin_above_minimum():
    rows = ocr_typos.candidates(["구매자에게 구메하고 소유하는 기쁨"])
    assert rows and all(m >= ocr_typos.MARGIN_MIN for _, _, m in rows)


def test_suspect_list_includes_ocr_candidates_without_changing_text(tmp_path):
    # 훙분·힘올은 Kiwi 기본 오타 모드가 못 잡는다 — OCR 후보가 합쳐져야 목록에 나온다
    result = tmp_path / "책B.txt"
    body = "두려움과 훙분이 교차한다\n조직도 힘올 얻는다"
    result.write_text(body, encoding="utf-8")
    out, _ = typo_suspects.save(result, user_dict={})
    assert result.read_text(encoding="utf-8") == body          # 본문은 그대로
    pairs = {(r["wrong"], r["suggestion"]) for r in typo_suspects.read(out)}
    assert {("훙분", "흥분"), ("힘올", "힘을")} <= pairs


def test_ocr_candidates_use_own_kiwi():
    # 기본 오타 모드와 TypoTransformer 를 한 Kiwi 에 번갈아 주면 프로세스가 죽었다(kiwipiepy 0.24, 12회 중 4회)
    assert ocr_typos._get() is not typo_suspects._get()
    assert ocr_typos._get() is not spacing._get()


def test_registered_ocr_typo_is_not_listed():
    rows = typo_suspects.find("두려움과 훙분이 교차한다", user_dict={"훙분": "흥분"})
    assert all(r["wrong"] != "훙분" for r in rows)
