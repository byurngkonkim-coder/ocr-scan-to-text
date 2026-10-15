"""설정 파일 관리 — API 키 및 사용자 옵션 로컬 저장"""
from __future__ import annotations
import json
import os
from pathlib import Path

# 설정 파일 위치: 도구 루트 디렉터리
_CONFIG_DIR = Path(__file__).parent.parent
_CONFIG_FILE = _CONFIG_DIR / "config.json"

_DEFAULTS: dict = {
    "anthropic_api_key": "",
    "google_api_key": "",
    "vision_provider": "gemini",          # "gemini" | "claude" | "ollama"
    # ── 로컬 Ollama (vision_provider="ollama") ─────────────────────────────────
    "ollama_url": "http://localhost:11434",
    "ollama_vision_model": "qwen3-vl:4b-instruct",   # 이미지 → 텍스트
    "ollama_text_model": "exaone3.5:7.8b",  # 텍스트 → 텍스트 (run_text)
    "ollama_max_edge": 1600,              # CPU 추론 — 장변을 줄여 이미지 토큰·지연 절감
    "ollama_timeout_s": 900,              # CPU 추론은 페이지당 수 분 걸린다
    "gemini_model": "gemini-2.5-flash",   # 모델 변경 시 코드 수정 없이 이 값만 교체
    # ── 성능 튜닝 (Gemini Vision) ──────────────────────────────────────────────
    "gemini_max_edge": 2560,              # 전송 전 이미지 장변 상한(px). 클수록 정확↑·속도↓
    "gemini_jpeg_quality": 85,            # 전송 이미지 JPEG 품질(1~95). PNG 대비 업로드 대폭↓
    "gemini_timeout_s": 45,               # 호출 타임아웃(초). 네트워크 멈춤(hang) 방지 핵심
    "gemini_disable_thinking": True,      # OCR엔 추론 불필요 → 끄면 지연·토큰↓ (2.5-flash 전용)
    "default_lang": "kor+eng",
    "default_dpi": 200,                   # Vision은 내부 축소 → 200이면 충분(속도·메모리·네트워크↓)
    "default_format": "txt",
    "confidence_threshold": 85.0,   # 하루명화 보정: 깨진 쪽 63~84, 본문 88+ (QA 2026-10-03)
    "auto_fallback": True,
    "vision_max_ratio": 0.1,        # 비용 우선: 문서당 Vision 폴백 쪽 상한(비율, 최소 3쪽)
    "local_cleanup_engine": "rules", # 비용 우선 정제: rules(내장 규칙) | ollama(exaone3.5, 보류)
    "auto_spellcheck": True,
    "save_figures": False,
    # ── 문서 정제(교정) 단계 ────────────────────────────────────────────────
    # OCR(충실 추출) 이후 문서 전체를 LLM(Gemini)으로 교정하여 쪽번호·러닝헤더·
    # 스캔줄 등 레이아웃 노이즈 제거 + 페이지 넘김 단어 복원 + 띄어쓰기·맞춤법.
    "enable_cleanup": False,              # 기본 끔 — 전 쪽 Gemini 재전송이라 비용의 94%(2026-10-15 실측). 필요할 때만 체크
    "cleanup_chunk_chars": 4000,          # 정제 1회 호출당 목표 문자 수(페이지 누적)
    "parallel_workers": 4,                # 한 파일 안에서 동시에 보내는 정제 구간·Vision 쪽 수(1=차례대로)
    "batch_workers": 2,                   # ④ 일괄 처리에서 동시에 처리하는 파일 수(1=한 건씩)
    "remove_front_matter": False,         # 표제지·판권·차례 등 앞뒤 부속물까지 제거(기본 보존)
    # ── 여백 마진 제거 필터 ──────────────────────────────────────────────────
    "enable_margin_filter": False,        # 머리글·바닥글·쪽번호 영역 물리 필터 활성화 여부
    "auto_margin": True,                  # 스캔 PDF: 본문 쪽에서 머리글·바닥글 여백을 재서 가리고 인식
    "margin_top_ratio": 0.06,             # 상단 여백 비율 (0.0 ~ 1.0)
    "margin_bottom_ratio": 0.06,          # 하단 여백 비율 (0.0 ~ 1.0)
}


# ── JSON 파일 안전장치 (config.json·history.json 공용) ─────────────────────────
# 깨진 JSON을 빈 값으로 읽은 뒤 다음 저장이 덮어써 내용이 통째로 사라지던 결함 방지.

def backup_corrupt(path: Path) -> None:
    """깨진 JSON 파일을 *.bak-시각 으로 옮겨 보존한다(다음 저장이 덮어쓰지 못하게)."""
    from datetime import datetime
    try:
        path.replace(path.with_name(f"{path.name}.bak-{datetime.now():%Y%m%d_%H%M%S}"))
    except OSError:
        pass


import time


def atomic_write_json(path: Path, data) -> None:
    """임시 파일에 쓴 뒤 교체 — 쓰는 도중 종료돼도 기존 파일이 깨지지 않는다."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.03 * (attempt + 1))


# ── 읽기 ──────────────────────────────────────────────────────────────────────

def load() -> dict:
    """설정 파일 로드. 없으면 기본값 반환."""
    if not _CONFIG_FILE.exists():
        return dict(_DEFAULTS)
    try:
        with open(_CONFIG_FILE, encoding="utf-8") as f:
            saved = json.load(f)
        return {**_DEFAULTS, **saved}
    except json.JSONDecodeError:
        backup_corrupt(_CONFIG_FILE)
        return dict(_DEFAULTS)
    except OSError:
        return dict(_DEFAULTS)


# 환경변수 우선 (키별로 허용하는 환경변수 이름 목록, 앞쪽이 우선)
ENV_KEY_NAMES = {
    "anthropic_api_key": ["ANTHROPIC_API_KEY"],
    # 유료(결제 계정) 키를 최우선 — 사용자 지정 2026-09-26, 새 키 '...2' 2026-10-13(이전 키는 401)
    "google_api_key":    ["Gemini_api_key(biiing acount)2", "Gemini_api_key(biiing acount)",
                          "GEMINI_API_KEY", "GOOGLE_API_KEY"],
}


def get(key: str, fallback=None):
    """단일 설정값 조회. 환경변수 > 설정 파일 > 기본값 순서."""
    for env_name in ENV_KEY_NAMES.get(key, []):
        env_val = os.environ.get(env_name, "")
        if env_val:
            return env_val
    return load().get(key, fallback)


def key_source(key: str) -> str | None:
    """API 키가 어디서 오는지만 알려 준다 — 환경변수 이름 | 'config.json' | None.
    키 값은 절대 반환하지 않는다(화면 표시용)."""
    for env_name in ENV_KEY_NAMES.get(key, []):
        if os.environ.get(env_name, ""):
            return env_name
    return "config.json" if load().get(key, "") else None


# ── 쓰기 ──────────────────────────────────────────────────────────────────────

def save(updates: dict) -> None:
    """설정 파일에 값 저장 (기존 설정 유지)."""
    current = load()
    current.update(updates)
    # API 키는 파일에 저장하되 콘솔에 노출 금지
    atomic_write_json(_CONFIG_FILE, current)


def set_api_key(key: str) -> None:
    """API 키 저장."""
    save({"anthropic_api_key": key.strip()})


def clear_api_key() -> None:
    """저장된 API 키 삭제."""
    save({"anthropic_api_key": ""})


# ── 조회 헬퍼 ─────────────────────────────────────────────────────────────────

def get_api_key() -> str:
    """환경변수 → 설정 파일 순서로 API 키 반환. 없으면 빈 문자열.

    붙여넣기 오류로 키가 중복 연결된 경우(같은 'sk-ant-...' 키가 여러 번
    이어붙은 값)는 무효 처리하여 빈 문자열을 반환한다.
    → Claude Vision 폴백이 401 인증 오류로 처리를 중단시키지 않고,
      'API 키 미설정' 상태로 우아하게 강등(검토 필요 표시)된다.
    """
    key = get("anthropic_api_key", "")
    if key and key.count("sk-ant") > 1:
        return ""
    return key


def has_api_key() -> bool:
    return bool(get_api_key())


# ── Google Gemini ─────────────────────────────────────────────────────────────

def get_google_api_key() -> str:
    """환경변수(GEMINI_API_KEY → GOOGLE_API_KEY) → 설정 파일 순서로 반환."""
    key = get("google_api_key", "")
    # 붙여넣기 오류로 키가 중복 연결된 경우 무효 처리 (Gemini 키는 'AIza'로 시작)
    if key and key.count("AIza") > 1:
        return ""
    return key


def has_google_api_key() -> bool:
    return bool(get_google_api_key())


def get_vision_provider() -> str:
    """Vision 폴백 엔진 선택: 'gemini' | 'claude' | 'ollama'. 기본값 gemini."""
    return str(load().get("vision_provider", "gemini")).strip().lower()


def get_gemini_model() -> str:
    """Gemini 모델 ID. config.json의 gemini_model로 코드 수정 없이 변경 가능."""
    return str(load().get("gemini_model", "gemini-2.5-flash")).strip()


# ── 성능 튜닝 게터 ────────────────────────────────────────────────────────────

def get_gemini_max_edge() -> int:
    """전송 전 이미지 장변 상한(px). 0/음수면 다운스케일 안 함."""
    try:
        return int(load().get("gemini_max_edge", 2560))
    except (TypeError, ValueError):
        return 2560


def get_gemini_jpeg_quality() -> int:
    """전송 이미지 JPEG 품질(1~95)."""
    try:
        q = int(load().get("gemini_jpeg_quality", 85))
    except (TypeError, ValueError):
        return 85
    return max(1, min(95, q))


def get_gemini_timeout_s() -> int:
    """Gemini 호출 타임아웃(초). 네트워크 멈춤 시 무한 대기 방지."""
    try:
        return max(5, int(load().get("gemini_timeout_s", 45)))
    except (TypeError, ValueError):
        return 45


def get_gemini_disable_thinking() -> bool:
    """OCR 시 thinking(추론) 비활성화 여부. 2.5-flash에서 지연·토큰 절감."""
    return bool(load().get("gemini_disable_thinking", True))


def get_default_dpi() -> int:
    """PDF 렌더링 DPI. 낮을수록 빠르고 가벼움(네트워크·메모리↓). Vision은 200이면 충분."""
    try:
        return max(72, int(load().get("default_dpi", 200)))
    except (TypeError, ValueError):
        return 200


# ── 문서 정제(교정) 단계 게터 ─────────────────────────────────────────────────

def get_enable_cleanup() -> bool:
    """OCR 후 문서 전체 LLM 정제(교정) 단계 실행 여부. 기본 False(비용)."""
    return bool(load().get("enable_cleanup", False))


def get_confidence_threshold() -> float:
    """RapidOCR 우선 모드에서 이 신뢰도 미만이면 Vision 폴백. 기본 85."""
    try:
        return float(load().get("confidence_threshold", 85.0))
    except (TypeError, ValueError):
        return 85.0


def get_local_cleanup_engine() -> str:
    """비용 우선 모드 문장 정제 엔진: 'rules'(내장 정제규칙, 기본) | 'ollama'(exaone3.5 — 마지막 수단, 느림)."""
    return "ollama" if load().get("local_cleanup_engine") == "ollama" else "rules"


def get_vision_max_ratio() -> float:
    """비용 우선 모드에서 한 문서가 Vision 폴백에 쓸 수 있는 쪽 비율 상한. 기본 0.1(최소 3쪽).
    실측(2026-10-15): B2B영업전략 301쪽 중 131쪽(43%)이 폴백 — 보완이 아니라 주 경로가 됐다."""
    try:
        return float(load().get("vision_max_ratio", 0.1))
    except (TypeError, ValueError):
        return 0.1


def get_parallel_workers() -> int:
    """한 파일 안의 동시 호출 수(정제 구간·Vision 쪽). 1~8, 기본 4."""
    try:
        return min(8, max(1, int(load().get("parallel_workers", 4))))
    except (TypeError, ValueError):
        return 4


def get_batch_workers() -> int:
    """④ 일괄 처리 동시 파일 수. 1~4, 기본 2. 동시 Gemini 호출 = 이 값 × parallel_workers."""
    try:
        return min(4, max(1, int(load().get("batch_workers", 2))))
    except (TypeError, ValueError):
        return 2


def get_cleanup_chunk_chars() -> int:
    """정제 1회 호출당 목표 문자 수(페이지 단위 누적). 최소 1000."""
    try:
        return max(1000, int(load().get("cleanup_chunk_chars", 4000)))
    except (TypeError, ValueError):
        return 4000


def get_remove_front_matter() -> bool:
    """표제지·판권·차례 등 앞뒤 부속물까지 제거할지 여부. 기본 False(보존)."""
    return bool(load().get("remove_front_matter", False))


# ── 여백 마진 제거 필터 게터 ──────────────────────────────────────────────────

def get_enable_margin_filter() -> bool:
    """머리글·바닥글·쪽번호 물리 필터 활성화 여부. 기본 False."""
    return bool(load().get("enable_margin_filter", False))


def get_auto_margin() -> bool:
    """스캔 PDF 여백 자동 측정·가림(core.margin_detector). 기본 True."""
    return bool(load().get("auto_margin", True))


def get_margin_top_ratio() -> float:
    """상단 여백 비율 (0.0 ~ 1.0). 기본 0.06."""
    try:
        val = float(load().get("margin_top_ratio", 0.06))
        return max(0.0, min(1.0, val))
    except (TypeError, ValueError):
        return 0.06


def get_margin_bottom_ratio() -> float:
    """하단 여백 비율 (0.0 ~ 1.0). 기본 0.06."""
    try:
        val = float(load().get("margin_bottom_ratio", 0.06))
        return max(0.0, min(1.0, val))
    except (TypeError, ValueError):
        return 0.06


def set_google_api_key(key: str) -> None:
    save({"google_api_key": key.strip()})


def config_path() -> Path:
    return _CONFIG_FILE


def show() -> dict:
    """사람이 읽을 수 있는 설정 딕셔너리 (API 키는 마스킹)."""
    cfg = load()
    key = cfg.get("anthropic_api_key", "")
    cfg["anthropic_api_key"] = (
        key[:8] + "..." + key[-4:] if len(key) > 12 else ("설정됨" if key else "미설정")
    )
    return cfg


# ── 로컬 Ollama ───────────────────────────────────────────────────────────────

def get_ollama_url() -> str:
    return str(load().get("ollama_url", "http://localhost:11434")).strip()


def get_ollama_vision_model() -> str:
    return str(load().get("ollama_vision_model", "qwen3-vl:4b-instruct")).strip()


def get_ollama_text_model() -> str:
    return str(load().get("ollama_text_model", "exaone3.5:7.8b")).strip()


def get_ollama_max_edge() -> int:
    return int(load().get("ollama_max_edge", 1600))


def get_ollama_timeout_s() -> int:
    return int(load().get("ollama_timeout_s", 900))
