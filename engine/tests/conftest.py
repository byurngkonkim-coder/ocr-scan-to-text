"""tests 공통 — 04_처리엔진 을 import 경로에 올린다.

파일마다 sys.path.insert 를 두면 빠진 파일이 수집 순서에 따라 조용히 깨진다
(test_anthropic_response_guard.py 가 가장 먼저 수집되며 `No module named 'core'`).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
