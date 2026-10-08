"""API 사용량 추적 — 실제 토큰 기반 비용 추정 및 로컬 저장"""
from __future__ import annotations
import json
import threading
from datetime import datetime
from pathlib import Path

# ── 모델별 가격 (USD / 1M 토큰) ────────────────────────────────────────────
_PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-haiku-4-5":          {"input": 0.80, "output": 4.00},
    "claude-haiku":              {"input": 0.80, "output": 4.00},
    "claude-sonnet-4-6":         {"input": 3.00, "output": 15.00},
    "claude-opus-4-8":           {"input": 15.00, "output": 75.00},
    # Google Gemini (대략치 · 요율 변동 가능)
    "gemini-2.5-flash":          {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite":     {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash":          {"input": 0.10, "output": 0.40},
    "gemini-1.5-flash":          {"input": 0.075, "output": 0.30},
}
_DEFAULT_PRICE = {"input": 0.80, "output": 4.00}


def _active_labels() -> dict:
    """현재 vision_provider 에 맞는 표시 라벨·요율·콘솔 링크."""
    try:
        from core import config
        provider = config.get_vision_provider()
        model = config.get_gemini_model() if provider != "claude" else "claude-haiku-4-5-20251001"
    except Exception:
        provider, model = "gemini", "gemini-2.5-flash"
    if provider == "claude":
        return {
            "vision": "Claude Vision (OCR)",
            "llm":    "L3 문맥 보정 (Claude Haiku)",
            "price":  _PRICING.get(model, _DEFAULT_PRICE),
            "console": "Anthropic 콘솔",
            "url":    "https://console.anthropic.com/settings/billing",
        }
    return {
        "vision": "Google Gemini OCR",
        "llm":    "문장 정제·교정 (Gemini)",
        "price":  _PRICING.get(model, {"input": 0.30, "output": 2.50}),
        "console": "Google AI Studio",
        "url":    "https://aistudio.google.com/",
    }

# 통계 저장 파일
_STATS_FILE = Path(__file__).parent.parent / "usage_stats.json"

# 기본 통계 구조
_EMPTY: dict = {
    "vision": {"calls": 0, "input_tokens": 0, "output_tokens": 0},
    "llm":    {"calls": 0, "input_tokens": 0, "output_tokens": 0},
    "history": [],   # 최근 20건 기록
}


# ── 읽기 / 쓰기 ────────────────────────────────────────────────────────────

def _load() -> dict:
    if not _STATS_FILE.exists():
        return {k: (v.copy() if isinstance(v, dict) else list(v))
                for k, v in _EMPTY.items()}
    try:
        return json.loads(_STATS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {k: (v.copy() if isinstance(v, dict) else list(v))
                for k, v in _EMPTY.items()}


def _save(stats: dict) -> None:
    # 임시 파일에 쓰고 바꿔 끼운다 — 쓰는 도중 다른 쪽이 읽으면 깨진 JSON → _load 가 빈
    # 통계로 보고 → 다음 저장이 누적치를 지워 버린다.
    from core.config import atomic_write_json
    atomic_write_json(_STATS_FILE, stats)


# 병렬 처리(구간·쪽 동시 호출)에서 읽기-더하기-쓰기가 겹치면 기록이 서로 덮어쓴다
_lock = threading.Lock()


# ── 기록 API ───────────────────────────────────────────────────────────────

def record_vision(input_tokens: int, output_tokens: int,
                  model: str = "claude-haiku-4-5-20251001") -> None:
    """Claude Vision 호출 1회 기록."""
    with _lock:
        stats = _load()
        stats["vision"]["calls"]         += 1
        stats["vision"]["input_tokens"]  += input_tokens
        stats["vision"]["output_tokens"] += output_tokens
        _append_history(stats, "vision", model, input_tokens, output_tokens)
        _save(stats)


def record_llm(input_tokens: int, output_tokens: int,
               model: str = "claude-haiku-4-5-20251001") -> None:
    """L3 LLM 보정 호출 1회 기록."""
    with _lock:
        stats = _load()
        stats["llm"]["calls"]         += 1
        stats["llm"]["input_tokens"]  += input_tokens
        stats["llm"]["output_tokens"] += output_tokens
        _append_history(stats, "llm", model, input_tokens, output_tokens)
        _save(stats)


def reset() -> None:
    """사용 통계 초기화."""
    with _lock:
        _save({k: (v.copy() if isinstance(v, dict) else list(v))
               for k, v in _EMPTY.items()})


# ── 조회 API ───────────────────────────────────────────────────────────────

def get_stats() -> dict:
    """통계 딕셔너리 반환 (calls, tokens, cost_usd, cost_krw 포함)."""
    s = _load()
    result = {}
    total_usd = 0.0
    p = _active_labels()["price"]   # 현재 provider 요율(대략치)

    for kind in ("vision", "llm"):
        d    = s.get(kind, _EMPTY[kind])
        inp  = d.get("input_tokens", 0)
        out  = d.get("output_tokens", 0)
        cost = (inp * p["input"] + out * p["output"]) / 1_000_000
        total_usd += cost
        result[kind] = {
            "calls":         d.get("calls", 0),
            "input_tokens":  inp,
            "output_tokens": out,
            "cost_usd":      round(cost, 6),
            "cost_krw":      round(cost * 1_380, 1),  # 환율 1,380원 기준
        }

    result["total"] = {
        "cost_usd": round(total_usd, 6),
        "cost_krw": round(total_usd * 1_380, 1),
    }
    result["history"] = s.get("history", [])[-20:]
    return result


def format_summary() -> str:
    """사람이 읽기 쉬운 사용량 요약 문자열."""
    s   = get_stats()
    v   = s["vision"]
    l   = s["llm"]
    t   = s["total"]
    lab = _active_labels()
    lines = [
        "【 이 앱의 API 사용량 (로컬 추정) 】",
        "",
        f"  {lab['vision']}",
        f"    호출 횟수   : {v['calls']:,}회",
        f"    입력 토큰   : {v['input_tokens']:,}",
        f"    출력 토큰   : {v['output_tokens']:,}",
        f"    추정 비용   : ${v['cost_usd']:.4f}  (약 {v['cost_krw']:,.0f}원)",
        "",
        f"  {lab['llm']}",
        f"    호출 횟수   : {l['calls']:,}회",
        f"    입력 토큰   : {l['input_tokens']:,}",
        f"    출력 토큰   : {l['output_tokens']:,}",
        f"    추정 비용   : ${l['cost_usd']:.4f}  (약 {l['cost_krw']:,.0f}원)",
        "",
        "  ─────────────────────────────────",
        f"  누적 합계     : ${t['cost_usd']:.4f}  (약 {t['cost_krw']:,.0f}원)",
        "",
        f"  ※ 추정치입니다(요율 변동 가능). 실제 사용량·요금은 {lab['console']}에서 확인하세요.",
        f"    {lab['url']}",
    ]
    return "\n".join(lines)


# ── 내부 ──────────────────────────────────────────────────────────────────

def _cost_usd(inp: int, out: int,
              model: str = "claude-haiku-4-5-20251001") -> float:
    p = _PRICING.get(model, _DEFAULT_PRICE)
    return (inp * p["input"] + out * p["output"]) / 1_000_000


def _append_history(stats: dict, kind: str, model: str,
                    inp: int, out: int) -> None:
    entry = {
        "time":  datetime.now().strftime("%Y-%m-%d %H:%M"),
        "type":  kind,
        "model": model,
        "in":    inp,
        "out":   out,
        "cost":  round(_cost_usd(inp, out, model), 6),
    }
    if "history" not in stats:
        stats["history"] = []
    stats["history"].append(entry)
    # 최근 100건만 유지
    if len(stats["history"]) > 100:
        stats["history"] = stats["history"][-100:]
