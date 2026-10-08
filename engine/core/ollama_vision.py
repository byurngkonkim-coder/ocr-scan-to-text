"""Ollama(로컬) Vision 폴백 — 인터넷·API 키 없이 저품질 스캔 재인식

gemini_vision / claude_vision 과 동일한 인터페이스(run / run_text / is_available / _client)를
제공한다. 엔진 선택은 core.vision 파사드가 config.json 의 vision_provider 로 결정한다.

    "vision_provider": "ollama"
    "ollama_vision_model": "qwen3-vl:4b-instruct"   # 이미지 → 텍스트
    "ollama_text_model":   "exaone3.5:7.8b" # 텍스트 → 텍스트 (run_text)

HTTP 호출은 표준 라이브러리(urllib)만 사용한다 — 추가 패키지 없음.
로컬 CPU 추론은 느리므로(페이지당 수 분) 타임아웃을 넉넉히 둔다.
"""
import base64
import json
import urllib.request

from PIL import Image

from core import config
from core.gemini_vision import _prepare_jpeg_bytes
from core.types import OCREngine, OCRResult

OLLAMA_VISION_PROMPT = "이 이미지의 모든 텍스트를 읽는 순서대로 그대로 추출하세요. 설명 없이 텍스트만 출력."
# 한 쪽 ≈ 1,500~2,000토큰. 반복 루프 시 이 상한에서 끊겨 length 예외 → Tesseract 폴백 (CPU 약 9분 상한)
VISION_NUM_PREDICT = 3072

_client = None  # 파사드 reset_client() 호환용 — HTTP 무상태라 캐시할 클라이언트가 없다


# ── LLM 종료 사유 가드 (G01·G03) ───────────────────────────────────────────
# Ollama는 num_predict·num_ctx 상한에 걸리면 done_reason=length 로 잘린 텍스트를
# 예외 없이 정상 반환한다 (실측 2026-09-25: 120줄 청크가 원문의 10%만 출력).
def response_ok(d: dict) -> tuple[bool, str]:
    """(정상 종료 여부, done_reason). stop/판단불가(UNKNOWN)만 수용."""
    dr = str(d.get("done_reason") or "UNKNOWN")
    return dr in ("stop", "UNKNOWN"), dr


def _base_url() -> str:
    return config.get_ollama_url().rstrip("/")


def _generate(model: str, prompt: str, images: list[str] | None = None, num_ctx: int = 8192,
              num_predict: int = -1) -> str:
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        # thinking 모델은 CPU에서 추론 토큰만 수천 개 소모 (실측 qwen3-vl:4b 한 쪽 900초 초과).
        # 단, qwen3-vl:4b 태그는 이 값을 무시하므로 기본 모델은 -instruct 태그를 쓴다.
        "think": False,
        "options": {"temperature": 0.0, "num_predict": num_predict, "num_ctx": num_ctx},
    }
    if images:
        body["images"] = images
    req = urllib.request.Request(
        f"{_base_url()}/api/generate",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=config.get_ollama_timeout_s()) as r:
        d = json.load(r)
    ok, dr = response_ok(d)
    text = d.get("response") or ""
    if not ok:
        raise RuntimeError(f"Ollama 비정상 종료 done_reason={dr} (텍스트 {len(text)}자) — 페이지 소실 방지 차단")
    if not text.strip():
        raise RuntimeError(f"Ollama 빈 응답 (model={model}) — 페이지 소실 방지 차단")
    return text


def _vision_prompt(model: str, lang_hint: str) -> str:
    # GLM-OCR 은 OCR 전용 모델이라 자유 지시문 대신 고정 작업 프롬프트를 쓴다 (모델 카드 규약)
    if "glm-ocr" in model:
        return "Text Recognition:"
    # Gemini용 VISION_PROMPT(줄바꿈 보존·[그림] 표기)는 4B 로컬 모델을 반복 루프에 빠뜨린다
    # (실측 2026-09-26: 900초 타임아웃 vs 이 짧은 지시문 290초 정상 종료)
    return OLLAMA_VISION_PROMPT


def run(image: Image.Image, lang_hint: str = "kor+eng") -> OCRResult:
    """로컬 Ollama 비전 모델로 이미지에서 텍스트 추출.

    비정상 종료·빈 응답·연결 실패는 예외 — 호출부(_process_page)가 Tesseract 폴백으로 전환한다.
    """
    model = config.get_ollama_vision_model()
    jpeg = _prepare_jpeg_bytes(image, max_edge=config.get_ollama_max_edge())
    text = _generate(model, _vision_prompt(model, lang_hint), images=[base64.b64encode(jpeg).decode("ascii")],
                     num_predict=VISION_NUM_PREDICT)
    return OCRResult(
        text=text,
        confidence=90.0,
        engine=OCREngine.OLLAMA,
        has_jamo_split=False,
    )


def run_text(prompt: str) -> str:
    """텍스트 → 텍스트 로컬 호출 (정제 단계용). 문맥창은 입력+출력을 덮도록 글자 수×2로 잡는다."""
    num_ctx = min(32768, max(4096, -(-len(prompt) * 2 // 1024) * 1024))
    return _generate(config.get_ollama_text_model(), prompt, num_ctx=num_ctx)


def is_available() -> bool:
    """Ollama 서버가 응답하고 비전 모델이 설치돼 있으면 True."""
    try:
        with urllib.request.urlopen(f"{_base_url()}/api/tags", timeout=2) as r:
            names = {m.get("name", "") for m in json.load(r).get("models", [])}
    except Exception:  # noqa: BLE001 — 서버 미기동·주소 오류는 '사용 불가'로 처리
        return False
    model = config.get_ollama_vision_model()
    return model in names or f"{model}:latest" in names
