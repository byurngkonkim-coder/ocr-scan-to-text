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
        assert w.strip() == w and all("가" <= c <= "힣" or c == " " for c in w)


@pytest.mark.parametrize("line, pair", [   # ㅏ/ㅣ — 텍스트 레이어 PDF 3권(2026-11-06) 실측
    ("상실 앞에서, 어찌 분노히지 않을 수 있으랴.", ("노히", "노하")),
    ("계속 모르고 살이가는 자가 진정한 행운아", ("살이", "살아")),
    ("유일한 방법은 남들과 디르게 하는 것이다", ("디르", "다르")),
])
def test_finds_thin_stroke_a_i_confusions(line, pair):
    assert pair in _pairs(line)


def test_wrong_word_appears_only_where_the_error_is():
    # 사용자 사전은 모든 책에 글자 그대로 치환한다 — 2자 조각 '았을'(받았을)·'리고'(그리고)를 내면
    # 등록 순간 바른 낱말이 망가진다. 틀린 말은 책 안에서 오류 자리에만 나와야 한다
    lines = ["저는 질문을 받았을 때 답을 모르면 모른다고 합니다.",
             "그 사람을 행복하게 해 줄 수 았을까?"]
    text = "\n".join(lines)
    rows = ocr_typos.candidates(lines)
    assert any("있을" in s for _, s, _ in rows)
    for w, _, _ in rows:
        assert text.count(w) == 1 and "받았을" not in w


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


# ── 2단계: 조사 계열 자동 반영(사용자 결정 2026-11-06) ──
def test_auto_fix_changes_only_josa_confusions():
    src = "성공의 비밀올 알 수 있다\n구매자에게 구메하고 소유하는 기쁨\n보이는 것듈 해결하리란 믿음"
    new, n = ocr_typos.auto_fix(src)
    assert "비밀을" in new and "것들" in new
    assert "구메하고" in new          # 조사 계열이 아닌 짝은 목록만 — 본문은 그대로
    assert n == 2 and len(new) == len(src) and new.count("\n") == src.count("\n")


@pytest.mark.parametrize("line", ["내일 친구가 올 거야.", "올해는 비가 많이 올 것 같다.", "올바른 길을 가라.",
                                  "봄이 온 줄 몰랐다.", "온 힘을 다해 노력해 보아도", "훨씬 나올 수 있다"])
def test_auto_fix_keeps_correct_ol_on(line):
    assert ocr_typos.auto_fix(line) == (line, 0)


@pytest.mark.parametrize("line", ["사소한 잘못올을 저지르면 안 된다",       # 을을 이 된다
                                  "이 회사는 c power올 주창했다",           # 을/를 못 가림
                                  "제공권 —— 올 확보하는 것이 중요하다"])
def test_auto_fix_skips_doubled_or_non_hangul_left(line):
    assert ocr_typos.auto_fix(line) == (line, 0)


@pytest.mark.parametrize("src, want", [("결국 대학을 졸업히고 취직했다.", "결국 대학을 졸업하고 취직했다."),
                                       ("남들과 디르게 생각해야 한다.", "남들과 다르게 생각해야 한다.")])
def test_auto_fix_hi_di(src, want):   # ㅏ/ㅣ 히→하·디→다(사용자 결정 2026-11-06)
    assert ocr_typos.auto_fix(src) == (want, 1)


@pytest.mark.parametrize("line", ["특히 조용히 천천히 걸어라.", "분명히 확실히 말했다.", "히말라야에 오르다.",
                                  "디지털 시대의 디자인과 오디오, 라디오를 듣는다.", "미디어와 아이디어가 중요하다.",
                                  "어디로 가는지 모른다."])
def test_auto_fix_keeps_correct_hi_di(line):
    assert ocr_typos.auto_fix(line) == (line, 0)


def test_auto_fix_looks_through_closing_quotes():
    assert ocr_typos.auto_fix("“노동신문”올 “인민일보”와 비교했다")[0].startswith("“노동신문”을")


def test_auto_fix_is_idempotent():
    once, _ = ocr_typos.auto_fix("온 힘올 다해 노력해 보아도 마옴대로 되지 않는다")
    assert ocr_typos.auto_fix(once) == (once, 0)
