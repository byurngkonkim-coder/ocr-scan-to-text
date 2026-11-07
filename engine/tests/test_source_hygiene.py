"""소스 파일이 깨지지 않았는지 — 백슬래시가 Bash heredoc·sed 를 거치며 한 번 더 풀려 생기는 사고를 막는다.

반복 사고(lessons 2026-10-06 한 세션 3회, 2026-11-07 세션 6회):
  - "\\n" 을 쓰려다 진짜 줄바꿈이 들어가 'unterminated string literal'
  - 정규식 r'\\s' 를 일반 문자열 '\\s' 로 써서 SyntaxWarning: invalid escape sequence (경고라 눈에 안 띈다)
  - 테스트 기대값 "| 제품 \\\\ 시장 |" 이 "| 제품 \\ 시장 |" 로 줄어 SyntaxWarning
경고를 오류로 바꿔 engine/ 의 모든 .py 를 컴파일한다. 고칠 곳: 그 줄을 raw 문자열(r'...')로 쓰거나 Edit/Write 도구로 다시 쓴다.
"""
import os
import warnings
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parent.parent
_OURS = ("core", "tools", "tests")      # 우리 코드만 — venv·python(내장 배포판)·외부 패키지는 보지 않는다
_SKIP = ("__pycache__", ".pytest_cache", "sample_docs")


def _sources() -> list[Path]:
    out = sorted(p for p in ENGINE.iterdir() if p.suffix == ".py")
    for sub in _OURS:
        for root, dirs, files in os.walk(ENGINE / sub):
            dirs[:] = [d for d in dirs if d not in _SKIP]
            out += sorted(Path(root) / f for f in files if f.endswith(".py"))
    return out


def compile_strict(source: str, name: str = "<src>") -> None:
    """SyntaxWarning(잘못된 이스케이프 등)까지 오류로 — compile() 이 경고만 내고 넘어가는 것을 막는다."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", SyntaxWarning)
        compile(source, name, "exec")


@pytest.mark.parametrize("path", _sources(), ids=lambda p: str(p.relative_to(ENGINE)))
def test_source_compiles_without_escape_warnings(path):
    try:
        compile_strict(path.read_text(encoding="utf-8"), str(path))
    except (SyntaxError, SyntaxWarning) as e:
        pytest.fail(f"{path.relative_to(ENGINE)}:{getattr(e, 'lineno', '?')} — {e}. "
                    "백슬래시가 든 줄은 raw 문자열(r'…')로, 고칠 때는 Bash heredoc 대신 Edit/Write 도구로.")


@pytest.mark.parametrize("bad", [
    'x = "| 제품 \\ 시장 |"',                  # 2026-11-07 test_refresh_tables 실제 사고
    "p = re.compile('\\s+')",                  # 정규식을 raw 로 안 씀
    's = "첫 줄\n둘째 줄"',                     # "\\n" 이 진짜 줄바꿈으로 풀림(2026-10-06)
])
def test_check_catches_past_breakages(bad):
    with pytest.raises((SyntaxError, SyntaxWarning)):
        compile_strict(bad)
