"""기존 결과 재교정 — 백업 → 오타 사전(기본+사용자) → R-17 띄어쓰기 → 덮어쓰기 → 의심 목록(intent 2026-10-17)."""
import pytest

from core import reclean, spacing, typo_suspects, user_typos


@pytest.fixture
def env(tmp_path, monkeypatch):
    results = tmp_path / "02_결과"
    results.mkdir()
    monkeypatch.setattr(user_typos, "PATH", tmp_path / "user_typos.tsv")
    monkeypatch.setattr(reclean, "BACKUP_ROOT", tmp_path / "_백업")
    monkeypatch.setattr(typo_suspects, "save", lambda p: (p.parent / "_의심목록" / f"{p.stem}.tsv", 0))
    return results


def test_applies_dictionaries_and_backs_up(env, tmp_path):
    user_typos.add([("흘륨한", "훌륭한")], user_typos.PATH)
    f = env / "책A.txt"
    f.write_text("흘륨한 교육을 받았다.\n둘째 줄", encoding="utf-8")
    [r] = reclean.reclean_files([f])
    assert f.read_text(encoding="utf-8") == "훌륭한 교육을 받았다.\n둘째 줄"
    assert r["changed"] and r["typos"] == 1
    assert r["backup"].read_text(encoding="utf-8") == "흘륨한 교육을 받았다.\n둘째 줄"
    assert r["backup"].parent.parent == tmp_path / "_백업"


def test_unchanged_file_is_not_rewritten_or_backed_up(env):
    f = env / "책B.txt"
    f.write_text("바른 문장이다.", encoding="utf-8")
    [r] = reclean.reclean_files([f])
    assert not r["changed"] and r["backup"] is None
    assert not reclean.BACKUP_ROOT.exists()


def test_running_twice_gives_same_text(env):
    user_typos.add([("걷모습", "겉모습")], user_typos.PATH)
    f = env / "책C.txt"
    f.write_text("걷모습만 보고 인류에게하나밖에없는 원칙", encoding="utf-8")
    reclean.reclean_files([f])
    once = f.read_text(encoding="utf-8")
    [r] = reclean.reclean_files([f])
    assert f.read_text(encoding="utf-8") == once and not r["changed"]


@pytest.mark.skipif(not spacing.is_available(), reason="kiwipiepy 미설치")
def test_glued_words_spaced_without_losing_letters(env):
    f = env / "책D.txt"
    src = "인류에게하나밖에없는 진정한 원칙은 정의다."
    f.write_text(src, encoding="utf-8")
    [r] = reclean.reclean_files([f])
    out = f.read_text(encoding="utf-8")
    assert r["spaces"] > 0
    assert out.replace(" ", "") == src.replace(" ", "")


def test_skips_partial_and_non_txt(env):
    (env / "책E_정제중.txt").write_text("흘륨한", encoding="utf-8")
    (env / "책F.md").write_text("흘륨한", encoding="utf-8")
    rows = reclean.reclean_files([env / "책E_정제중.txt", env / "책F.md"])
    assert [r["skipped"] for r in rows] == ["중간 저장분", "txt 아님"]


def test_suspect_table_register_flow(env, monkeypatch):
    # 화면 5가지 질문: 목록 없음 안내 · 등록하면 표에서 빠짐 · 체크 안 하면 안내
    ocr_gradio = pytest.importorskip("ocr_gradio")
    monkeypatch.setattr(ocr_gradio, "RESULTS_DIR", env)
    rows, note = ocr_gradio._suspect_rows(None)
    assert rows == [] and "아직 없습니다" in note
    d = env / typo_suspects.SUBDIR
    d.mkdir()
    (d / "책A.tsv").write_text("틀린 말\t제안\t횟수\t문맥\n흘륨\t흙륨\t6\t흘륨한 교육\n지해\t지혜\t4\t지해를\n",
                              encoding="utf-8")
    rows, _ = ocr_gradio._suspect_rows("책A")
    assert [r[1] for r in rows] == ["흘륨", "지해"]
    _, note = ocr_gradio._suspect_register("책A", rows)
    assert "체크" in note
    rows[0][0], rows[0][2] = True, "훌륭"          # 제안이 틀려 고칠 말을 고쳐 등록
    rows, note = ocr_gradio._suspect_register("책A", rows)
    assert user_typos.load() == {"흘륨": "훌륭"}
    assert [r[1] for r in rows] == ["지해"] and "1개 등록" in note


def test_list_results_hides_partial_files(env):
    (env / "책A.txt").write_text("a", encoding="utf-8")
    (env / "책A_정제중.txt").write_text("a", encoding="utf-8")
    (env / "_의심목록").mkdir()
    assert [p.name for p in reclean.list_results(env)] == ["책A.txt"]
