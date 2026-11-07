"""재처리 도구 점검(tools/reprocess_book.quality) — 규칙이 제목 순서를 뒤엉키게 하거나 본문을 지우면 불합격."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import reprocess_book as rb   # noqa: E402

_E = [{"level": 1, "title": "PART I. 이론", "page": 1}, {"level": 2, "title": "1. 외부 환경 분석", "page": 2},
      {"level": 1, "title": "PART II. 사례", "page": 3}, {"level": 2, "title": "1. 외부 환경 분석", "page": 4}]
_BODY = "본문 글자가 여기 있다. " * 20


def _doc(*heads):
    return "\n\n".join(h + "\n\n" + _BODY for h in heads)


def test_good_result_passes():
    src = _doc("# PART I. 이론", "## 1. 외부 환경 분석", "# PART II. 사례", "## 1. 외부 환경 분석")
    q = rb.quality(_E, src, src)
    assert q["ok"] and q["found"] == 4 and q["order_breaks"] == 0


def test_same_title_twice_out_of_order_fails():
    # 12년차기획맨 실제 사고 꼴 — PART II 의 절 제목이 PART I 앞에 섰다
    out = _doc("## 1. 외부 환경 분석", "# PART I. 이론", "# PART II. 사례", "## 1. 외부 환경 분석")
    q = rb.quality(_E, out, out)
    assert not q["ok"] and q["order_breaks"] >= 1


def test_body_deleted_fails():
    src = _doc("# PART I. 이론", "## 1. 외부 환경 분석", "# PART II. 사례", "## 1. 외부 환경 분석")
    out = "# PART I. 이론\n\n## 1. 외부 환경 분석\n\n# PART II. 사례\n\n## 1. 외부 환경 분석\n\n" + _BODY
    q = rb.quality(_E, src, out)
    assert not q["ok"] and q["loss"] > 0.03


def test_english_body_deleted_fails():
    # JesusCEO: 영문 책은 한글이 0 이라 '0 → 0' 으로 통과했다 — 영문자가 많으면 영문자로 센다
    ents = [{"level": 1, "title": "He Said Thank You", "page": 1}, {"level": 1, "title": "He Owned It", "page": 2}]
    body = "Each of us must answer to someone in the end. " * 20
    src = f"# He Said Thank You\n\n{body}\n\n# He Owned It\n\n{body}"
    out = f"# He Said Thank You\n\n# He Owned It\n\n{body}"
    q = rb.quality(ents, src, out)
    assert q["script"] == "영문" and not q["ok"] and q["loss"] > 0.03


def test_korean_body_lost_in_book_with_more_english_fails():
    # 변화의기술: 쪽마다 영문 인용이 있어 영문자(22,585)가 한글(17,077)보다 많다 — 영문자만 재면 한글 문단이
    # 지워져도 몰랐다. 두 글자 모두 재고 더 많이 준 쪽으로 판정한다
    ents = [{"level": 1, "title": "나는 나를 사랑합니다", "page": 1}, {"level": 1, "title": "생각이 바뀌면", "page": 2}]
    eng = "It is not lack of ability or opportunity that holds you back. " * 30
    kor = "당신의 생각 속에서 스스로 설정한 한계를 제외하고는 당신이 성취할 수 있는 것에는 한계가 없다. " * 3
    src = f"# 나는 나를 사랑합니다\n\n{eng}\n\n{kor}\n\n# 생각이 바뀌면\n\n{eng}\n\n{kor}"
    out = f"# 나는 나를 사랑합니다\n\n{eng}\n\n{kor}\n\n# 생각이 바뀌면\n\n{eng}"
    q = rb.quality(ents, src, out)
    assert not q["ok"] and q["loss"] > 0.03


def test_titles_bunched_far_from_their_pages_fail():
    # Alianment정렬 실제 사고(2026-11-07) — 장 제목 10개가 모두 맨 뒤 주석 속 '2장 …_역사적 고찰' 줄에 섰다.
    # 순서는 맞아서 순서 검사로는 못 잡았다. 차례 쪽번호와 본문 속 자리가 크게 어긋나면 불합격.
    ents = [{"level": 1, "title": f"{k}장 제목{k}", "page": k * 40} for k in range(1, 10)]
    chap = "\n\n".join(_BODY for _ in range(20))      # 장마다 문단 20개
    body = "\n\n".join(chap for _ in range(9))
    good = "\n\n".join(f"# {k}장 제목{k}\n\n{chap}" for k in range(1, 10))
    bad = body + "\n\n" + "\n\n".join(f"# {k}장 제목{k}" for k in range(1, 10))
    assert rb.quality(ents, good, good)["ok"] and rb.quality(ents, good, good)["misplaced"] == 0
    q = rb.quality(ents, good, bad)
    assert not q["ok"] and q["misplaced"] >= 5


def test_book_info_title_ending_in_number_kept_with_dash(tmp_path):
    # '세일즈 입문 매뉴얼 30' — 끝 숫자가 쪽번호로 잘렸다(DrivingSalesmanPowerSkill). 끝에 ' -' 를 붙이면 쪽번호 없음
    p = tmp_path / "책.txt"
    p.write_text("책: 시리즈\n\n차례:\n1권 세일즈 입문 매뉴얼 30 -\n  01 첫 장 12\n", encoding="utf-8")
    toc = rb.read_book_info(p)["toc"]
    assert toc[0] == {"level": 1, "title": "1권 세일즈 입문 매뉴얼 30", "page": None}
    assert toc[1] == {"level": 2, "title": "01 첫 장", "page": 12}


def test_book_info_backref_followed_by_digit_rejected(tmp_path):
    # '\142.' 가 묶음 1 + '42.' 가 아니라 8진수 'b' 로 바뀌었다(IDEAHACKS49) — 읽을 때 막는다
    import pytest
    p = tmp_path / "책.txt"
    p.write_text("책: 시리즈\n바꾸기: (a)(b) => \\142. 제목\\n\\n\\2\n\n차례:\n머리말 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="g<1>"):
        rb.read_book_info(p)
    p.write_text("책: 시리즈\n바꾸기: (a)(b) => \\g<1>42. 제목\\n\\n\\2\n\n차례:\n머리말 1\n", encoding="utf-8")
    assert rb.read_book_info(p)["subs"] == [("(a)(b)", "\\g<1>42. 제목\\n\\n\\2")]


def test_book_info_empty_replacement_allowed(tmp_path):
    # '정규식 => ' (지우기) — 줄 끝 빈칸이 잘려 ' => ' 를 못 찾고 조용히 무시됐다(MBAInaBox 로고 잡음)
    p = tmp_path / "책.txt"
    p.write_text("책: 시리즈\n바꾸기: ^BOX  => \n\n차례:\n머리말 1\n", encoding="utf-8")
    assert rb.read_book_info(p)["subs"] == [("^BOX ", "")]


def test_ratchet_fails_when_found_titles_drop(tmp_path):
    # 21세기지식경영 실제 회귀 48 → 44 를 사람이 표를 대조해서야 찾았다
    base = tmp_path / "base.json"
    assert rb.ratchet("책", {"found": 48, "total": 48}, base) == ""
    assert rb.ratchet("책", {"found": 44, "total": 48}, base) == "제목 48 → 44 회귀"
    assert rb.ratchet("책", {"found": 48, "total": 48}, base) == ""


def test_status_finds_unfinished_and_next(tmp_path, monkeypatch):
    # 다시 시작 지점 — 결과는 만들었지만 기록 문서에 회차가 없는 책이 '하던 책', 그다음이 아직 안 만든 가장 오래된 결과
    res, out = tmp_path / "02_결과", tmp_path / "02_결과" / "_재처리"
    out.mkdir(parents=True)
    for s in ("20260101_a", "20260102_b", "20260103_c"):
        (res / f"{s}.md").write_text("x", encoding="utf-8")
    for s in ("20260101_a", "20260102_b"):
        (out / f"{s}.md").write_text("x", encoding="utf-8")
    log = tmp_path / "기록.md"
    log.write_text("### 1회 — 20260101_a (2026-11-07)\n", encoding="utf-8")
    monkeypatch.setattr(rb, "RESULTS", res)
    monkeypatch.setattr(rb, "OUT_DIR", out)
    st = rb.status(log)
    assert st == {"done": 1, "last": "20260101_a", "unfinished": ["20260102_b"], "next": "20260103_c",
                  "remaining": 2}


def test_status_reads_stem_with_space(tmp_path, monkeypatch):
    # 파일 이름에 빈칸이 든 책('1Q84_MurakamiHaruki_기울기 보정')을 기록해도 '하던 책' 으로 남지 않는다
    res, out = tmp_path / "02_결과", tmp_path / "02_결과" / "_재처리"
    out.mkdir(parents=True)
    (res / "20260101_a b.md").write_text("x", encoding="utf-8")
    (out / "20260101_a b.md").write_text("x", encoding="utf-8")
    log = tmp_path / "기록.md"
    log.write_text("### 1회 — 20260101_a b (2026-11-07)\n", encoding="utf-8")
    monkeypatch.setattr(rb, "RESULTS", res)
    monkeypatch.setattr(rb, "OUT_DIR", out)
    st = rb.status(log)
    assert st["last"] == "20260101_a b" and st["unfinished"] == []
