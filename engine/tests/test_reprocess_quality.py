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


def test_ratchet_fails_when_found_titles_drop(tmp_path):
    # 21세기지식경영 실제 회귀 48 → 44 를 사람이 표를 대조해서야 찾았다
    base = tmp_path / "base.json"
    assert rb.ratchet("책", {"found": 48, "total": 48}, base) == ""
    assert rb.ratchet("책", {"found": 44, "total": 48}, base) == "제목 48 → 44 회귀"
    assert rb.ratchet("책", {"found": 48, "total": 48}, base) == ""
