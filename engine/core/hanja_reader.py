"""한자 읽기 — 한글 모델이 비워 둔 자리를 중국어 모델로 다시 읽어 끼워 넣는다.

한글 v5 인식 모델의 글자 사전(11,945자)에는 한자가 하나도 없어, 한자 자리가 비거나('출사표()')
아무 표시 없이 사라졌다('일반적으로 의 뜻'). 같이 깔린 중국어 PP-OCRv6 인식 모델은 한자 15,565자를
알고 한글은 모른다(한글은 빈칸으로 낸다). 한글 모델 결과의 글자 상자로 한자가 있을 자리를 찾고,
그 자리만 중국어 모델로 읽는다.

틀린 한자보다 빈칸이 낫다(intent 2026-10-18) — 아래를 모두 지킬 때만 넣는다.
  1. 좌우 여유를 달리해 두 번 읽은 결과가 같다          (自勉→自免·人心如面→心如面 차단)
  2. 글자별 최저 신뢰도 >= MIN_CONF                    (司空→司三·司馬遷→司馬透 차단)
  3. KS X 1001 한자(cp949)만 — 간체자는 버린다          (勢→势 차단)
  4. 日·曰 은 넣지 않는다 — 서로 헷갈린다               (一曰→一日 차단)
  5. 한글 모델이 확신한 한글 글자 자리의 한자는 버린다  (추천사→外 차단)
  6. 자리에 남은 잉크 덩어리가 읽은 한자보다 많으면 덜 읽은 것 (孔父嘉→孔父 차단)
표본 4권 79쪽 실측(2026-10-18): 넣은 56곳 틀림 0, 한글 바뀐 곳 0, 한자 없는 책 15쪽 넣은 곳 0.

넓히기(2026-10-18 오후) — 괄호 없는 한자·한문 원문 줄을 놓치던 원인 세 가지:
  · 좌우 여유가 옆 한글까지 잘라 작은 한자를 흐리게 읽음 → 틈만 바짝 다시 읽기(_read_gap 2차)
  · 줄 끝 한자는 한글 모델이 글자 상자를 아예 안 냄 → 줄 앞뒤 끝도 틈으로 본다
  · 한문 줄을 '....' 로 읽어 점수가 높으면 줄 판독을 안 탐 → 한글·영숫자 없는 줄도 줄 판독
  · 한자 사이에 지어낸 한글 한 글자('五事七허論') → _sandwiched
고전 4권 32쪽: 한자 76→234자, 눈으로 대조한 쪽에서 틀린 한자 0. 한자 없는 책 8쪽 넣은 곳 0.
"""
from __future__ import annotations

import logging
import re
import threading

import numpy as np

logger = logging.getLogger(__name__)

TEXT_SCORE = 0.5      # RapidOCR 기본 줄 점수 문턱 — 이보다 낮은 줄은 예전처럼 버린다
SURE_HANGUL = 0.8     # 한글 모델이 이 이상 확신한 한글 글자는 한자로 바꾸지 않는다
MIN_CONF = 0.8
CONFUSABLE = set("日曰")
BRACKETS = set("()[]<>《》〈〉「」『』'\"‘’“”")
HPUNCT = set(",，、·.．。:;!?")
LINE_CONF = 0.5       # 한문 줄: 세 번 읽어 한자가 똑같으면 이 신뢰도까지 받는다(같은 줄 실측 낮은 글자도 원본과 일치)
# 중국어 판독에 '(' 와 ')' 가 함께 나오면 괄호가 한자 범위를 정해 주므로 잉크 판정(6)을 건너뛴다.
# 같은 표본 79쪽: 장원_제갈량 빈 괄호 24→18 이 24→11 로, 늘어난 7곳 모두 원본과 일치(2026-10-18)
PAREN_SKIP_INK = True

_CHI = None
_CHI_LOCK = threading.Lock()


def _is_hanja(c: str) -> bool:
    return "一" <= c <= "鿿" or "㐀" <= c <= "䶿"


def _is_hangul(c: str) -> bool:
    return "가" <= c <= "힣"


def _norm(s: str) -> str:
    """중국어 모델의 전각 쉼표 '，'·마침표 '．' → ', '·'. ' (한국어 본문 표기)."""
    s = re.sub(r"\s*[．。]\s*", ". ", s)
    return re.sub(r"\s*[,，]\s*", ", ", s).strip()


def _ks(c: str) -> bool:
    try:
        c.encode("cp949")
        return True
    except UnicodeEncodeError:
        return False


def _ext(box) -> tuple[float, float, float, float]:
    b = np.asarray(box)
    return float(b[:, 0].min()), float(b[:, 1].min()), float(b[:, 0].max()), float(b[:, 1].max())


def _chi_engine():
    """중국어 인식 엔진(지연 로드) — 늘 같은 인자로 부른다(RapidOCR 은 호출 인자를 설정에 남긴다)."""
    global _CHI
    with _CHI_LOCK:
        if _CHI is None:
            from rapidocr import RapidOCR
            _CHI = RapidOCR(params={"Global.use_cls": False, "Global.log_level": "error"})
    return _CHI


def _rec(img: np.ndarray, x0, y0, x1, y1) -> list[tuple[str, float, float]]:
    """잘라 낸 구간을 중국어 모델로 읽어 (글자, 신뢰도, 쪽 x좌표) 목록."""
    H, W = img.shape[:2]
    X0, Y0, X1, Y1 = max(0, int(x0)), max(0, int(y0)), min(W, int(x1)), min(H, int(y1))
    if X1 - X0 < 5 or Y1 - Y0 < 5:
        return []
    c = _chi_engine()(img[Y0:Y1, X0:X1], use_det=False, use_cls=False,
                      return_word_box=True, return_single_char_box=True)
    if not c.txts or not c.word_results:
        return []
    w = c.word_results[0]
    L = w.line_txt_len or 1
    out, i = [], 0
    for word, cols in zip(w.words, w.word_cols):
        for ch, col in zip(word, cols):
            out.append((ch, float(w.confs[i]) if i < len(w.confs) else 0.0, X0 + (col + 0.5) / L * (X1 - X0)))
            i += 1
    return out


def _incomplete(img, sure, g0, g1, y0, y1, xa, xb, n, h) -> bool:
    """한자 앞뒤에 글자만 한 잉크가 남았거나 덩어리가 읽은 한자보다 많으면 덜 읽은 것.

    川·小 처럼 갈라지는 글자도 덩어리가 늘어 버려진다 — 넣지 않는 쪽이 안전하다.
    번진 인쇄('孔父嘉' 의 嘉)도 잉크로 세려고 밝기 문턱을 180 으로 둔다.
    """
    L = max([b for a, b in sure if b <= g0 + 2], default=g0 - h)
    R = min([a for a, b in sure if a >= g1 - 2], default=g1 + h)
    w = max((xb - xa) / (n - 1) if n > 1 else 0.8 * h, 0.4 * h)
    X0, X1 = int(max(0, L + 2)), int(min(img.shape[1], R - 2))
    if X1 <= X0:
        return False
    ink = (img[int(y0):int(y1), X0:X1].mean(axis=2) < 180).any(axis=0)
    runs, st = [], None
    for i, v in enumerate(list(ink) + [False]):
        if v and st is None:
            st = i
        elif not v and st is not None:
            runs.append((X0 + st, X0 + i))
            st = None
    glyph = [(a, b) for a, b in runs if b - a >= 0.3 * w]   # 괄호·따옴표·쉼표처럼 가는 획은 뺀다
    if not glyph:
        return False
    return len(glyph) > n or min(a for a, _ in glyph) < xa - 0.9 * w or max(b for _, b in glyph) > xb + 0.9 * w


def _ink_rows(img, x0, x1, y0, y1, pad=3) -> tuple[float, float]:
    """구간 [x0,x1] 안 잉크가 있는 위아래 범위 — 본문보다 작게 찍힌 한자를 바짝 잘라 읽는다."""
    H, W = img.shape[:2]
    Y0, Y1 = max(0, int(y0)), min(H, int(y1))
    sub = img[Y0:Y1, max(0, int(x0)):min(W, int(x1))]
    rows = np.where((sub.mean(axis=2) < 180).any(axis=1))[0] if sub.size else []
    if not len(rows):
        return y0, y1
    return Y0 + rows[0] - pad, Y0 + rows[-1] + 1 + pad


def _read_gap(img, kchars, g0, g1, y0, y1, h, paren: bool):
    """틈 [g0,g1] 의 한자 → (넣을 글자, 첫 한자 x, 끝 한자 x) 또는 None.

    먼저 좌우 여유를 넓게(h·0.5h) 두고 읽고, 안 되면 틈만 바짝(0.1h·0.3h, 잉크 높이) 다시 읽는다.
    넓은 여유는 옆 한글까지 잘라 넣어 작은 한자를 흐리게 읽었다('론詭道論이다' 의 詭 0.51).
    """
    sure = [(a, b) for c, s, (a, _, b, _) in kchars if _is_hangul(c) and s >= SURE_HANGUL]
    ty0, ty1 = _ink_rows(img, g0, g1, y0 - 4, y1 + 4)
    for m1, m2, r0, r1 in ((h, 0.5 * h, y0 - 4, y1 + 4), (0.1 * h, 0.3 * h, ty0, ty1)):
        got = _try_gap(img, sure, g0, g1, y0, y1, h, paren, m1, m2, r0, r1)
        if got:
            return got
    return None


def _try_gap(img, sure, g0, g1, y0, y1, h, paren, m1, m2, r0, r1):
    def take(m):
        seq = _rec(img, g0 - m, r0, g1 + m, r1)
        hj = [j for j, (ch, _, x) in enumerate(seq) if _is_hanja(ch)
              and not any(a + 0.1 * (b - a) <= x <= b - 0.1 * (b - a) for a, b in sure)]
        if not hj:
            return [], seq
        # 첫 한자~끝 한자 사이의 쉼표·가운뎃점은 살린다('智, 信, 仁')
        return [t for t in seq[hj[0]:hj[-1] + 1] if _is_hanja(t[0]) or t[0] in HPUNCT], seq

    a, seq_a = take(m1)
    b, _ = take(m2)
    sa, sb = "".join(t[0] for t in a), "".join(t[0] for t in b)
    if not sa or sa != sb:
        return None
    hj = [t for t in a if _is_hanja(t[0])]
    if not all(_ks(t[0]) for t in hj) or CONFUSABLE & set(sa):
        return None
    if min(min(t[1] for t in hj), min(t[1] for t in b if _is_hanja(t[0]))) < MIN_CONF:
        return None
    xa, xb = hj[0][2], hj[-1][2]
    if paren and PAREN_SKIP_INK:
        s = "".join(t[0] for t in seq_a)
        inner = s.find("(") < s.find(sa) and s.find(")", s.find(sa)) > 0
        if inner:
            return _norm(sa), xa, xb
    if _incomplete(img, sure, g0, g1, y0, y1, xa, xb, len(hj), h):
        return None
    return _norm(sa), xa, xb


def _merge(txt: str, kchars, adds, drop=()) -> str | None:
    """한글 줄에 한자를 x 위치대로 끼워 넣고, 한자 자리에 한글 모델이 낸 잡음 글자('-'·'+')는 지운다.

    drop: 한자 자리에 한글 모델이 지어낸 한글 글자 번호(_sandwiched 가 찾음) — 함께 지운다.
    """
    if "".join(c for c in txt if not c.isspace()) != "".join(c for c, _, _ in kchars):
        return None
    cw = float(np.median([b[2] - b[0] for _, _, b in kchars]))
    drop, ins = set(drop), []
    for s, xa, xb in adds:
        for k, (c, _, (a, _, b, _)) in enumerate(kchars):
            if xa - 0.5 * cw <= (a + b) / 2 <= xb + 0.5 * cw and not _is_hangul(c) and not c.isdigit() and c not in BRACKETS:
                drop.add(k)
        mids = [((a + b) / 2, k) for k, (_, _, (a, _, b, _)) in enumerate(kchars) if k not in drop]
        prev = max([k for m, k in mids if m < xa], default=-1)
        gap_b = (xa - 0.5 * cw) - kchars[prev][2][2] if prev >= 0 else 0.0
        nxt = min([k for m, k in mids if m > xb], default=len(kchars))
        gap_a = kchars[nxt][2][0] - (xb + 0.5 * cw) if nxt < len(kchars) else 0.0
        ins.append((prev, s, gap_b, gap_a))
    pos = [i for i, c in enumerate(txt) if not c.isspace()]
    kept = [k for k in range(len(kchars)) if k not in drop]
    out = []
    for j, k in enumerate([-1] + kept):
        if k >= 0:
            out.append(txt[pos[k]])
        nk = kept[j] if j < len(kept) else None
        # 이 글자와 다음 남은 글자 사이 원문에 공백이 있었나(지운 잡음 글자 자리 포함)
        a = pos[k] + 1 if k >= 0 else 0
        b = pos[nk] if nk is not None else len(txt)
        ws = " " if k >= 0 and nk is not None and any(c.isspace() for c in txt[a:b]) else ""
        here = [(s, gb, ga) for p, s, gb, ga in ins if p == k]
        if not here:
            out.append(ws)
            continue
        for s, gb, ga in here:
            if ws:   # 원문 공백은 한 번만 — 한자는 더 가까운 쪽 글자에 붙인다(같으면 뒤 글자: '法이다'·'與는')
                out.append(" " + s if ga <= gb else s + " ")
            else:
                out.append((" " if k >= 0 and gb > 0.35 * cw else "") + s
                           + (" " if nk is not None and ga > 0.35 * cw else ""))
    return re.sub(r" {2,}", " ", "".join(out)).strip()


def _sandwiched(img, kchars, adds, spans, y0, y1, h, cw):
    """한자 두 덩이 사이에 낀 한글 한 글자 = 한글 모델이 한자를 한글로 지어낸 것('五事七허論' 의 허 = 計).

    그 글자를 빼고 두 틈을 하나로 다시 읽어, 앞 덩이 + 한자 한 자 + 뒤 덩이로 확실히 읽히면 바꾼다.
    → (adds, spans, 지울 글자 번호들)
    """
    drop = set()
    for k, (c, _, (a, _, b, _)) in enumerate(kchars):
        if not _is_hangul(c):
            continue
        L = [j for j, (_, _, xb) in enumerate(adds) if -0.5 * cw <= a - xb <= 1.5 * cw]
        R = [j for j, (_, xa, _) in enumerate(adds) if -0.5 * cw <= xa - b <= 1.5 * cw]
        if len(L) != 1 or len(R) != 1 or L == R:
            continue
        l, r = L[0], R[0]
        got = _read_gap(img, kchars[:k] + kchars[k + 1:], spans[l][0], spans[r][1], y0, y1, h, False)
        if not got:
            continue
        left, right = adds[l][0].replace(" ", ""), adds[r][0].replace(" ", "")
        s = got[0].replace(" ", "")
        mid = s[len(left):len(s) - len(right)]
        if s.startswith(left) and s.endswith(right) and len(mid) == 1 and _is_hanja(mid):
            adds = [x for j, x in enumerate(adds) if j not in (l, r)] + [got]
            spans = [x for j, x in enumerate(spans) if j not in (l, r)] + [(spans[l][0], spans[r][1])]
            drop.add(k)
    return adds, spans, drop


def _read_line(img, box, h) -> str | None:
    """한글 모델이 버리는 줄을 통째로 읽어 한문 줄이면 돌려준다."""
    x0, y0, x1, y1 = box
    a = _rec(img, x0 - 4, y0 - 4, x1 + 4, y1 + 4)
    b = _rec(img, x0 - 0.5 * h, y0 - 4, x1 + 0.5 * h, y1 + 4)
    sa = "".join(t[0] for t in a)
    chars = [c for c in sa if not c.isspace()]
    hj = [t for t in a if _is_hanja(t[0])]

    def han(seq):   # 마침표 모양(. ．)은 읽을 때마다 달라 한자만 맞춰 본다
        return "".join(t[0] for t in seq if _is_hanja(t[0]))
    if not hj or len(hj) < 0.6 * len(chars) or han(a) != han(b):
        return None
    if any(not _is_hanja(c) and c not in HPUNCT and c not in BRACKETS for c in chars):
        return None
    if not all(_ks(t[0]) for t in hj) or CONFUSABLE & set(sa):
        return None
    low = min(min(t[1] for t in hj), min(t[1] for t in b if _is_hanja(t[0])))
    if low < MIN_CONF:
        # 세 번째로 잉크 높이에 바짝 맞춰 읽어 한자가 또 같으면 낮은 신뢰도(LINE_CONF 이상)도 받는다
        ty0, ty1 = _ink_rows(img, x0, x1, y0 - 4, y1 + 4)
        if low < LINE_CONF or han(_rec(img, x0 - 4, ty0, x1 + 4, ty1)) != han(a):
            return None
    return _norm(sa)


def fill(img: np.ndarray, result) -> tuple[list[str], list, list[float]]:
    """RapidOCR 결과(text_score=0, 글자 상자 포함) → (줄 글자, 줄 상자, 줄 점수).

    점수 < TEXT_SCORE 줄은 예전처럼 버리되, 한문 줄로 확실히 읽히면 남긴다(점수는 0 — 쪽 신뢰도에 넣지 않는다).
    """
    lines, boxes, scores = [], [], []
    words = getattr(result, "word_results", None) or ()
    for i, txt in enumerate(result.txts or ()):
        txt, score = str(txt).strip(), float(result.scores[i])
        x0, y0, x1, y1 = _ext(result.boxes[i])
        h = y1 - y0
        try:
            # 한글·영문·숫자가 없는 줄('....'·'.)')은 한문 줄을 한글 모델이 점으로 읽은 것일 수 있다
            no_text = not any(_is_hangul(c) or (c.isascii() and c.isalnum()) for c in txt)
            if score < TEXT_SCORE or no_text:
                line = _read_line(img, (x0, y0, x1, y1), h)
                if line:
                    lines.append(line); boxes.append(result.boxes[i]); scores.append(0.0 if no_text else score)
                elif score >= TEXT_SCORE and txt:
                    lines.append(txt); boxes.append(result.boxes[i]); scores.append(score)
                continue
            kchars = [(c, float(s), _ext(b)) for c, s, b in words[i]] if i < len(words) else []
            adds, spans, drop = [], [], set()
            if len(kchars) >= 2:
                cw = float(np.median([b[2] - b[0] for _, _, b in kchars]))
                # 글자 사이 틈 + 줄 앞뒤 끝(한글 모델이 줄 끝 한자를 아예 안 낸다: '않는다.(孫武一書…')
                gaps = [(x0, kchars[0][2][0], False)]
                gaps += [(kchars[k][2][2], kchars[k + 1][2][0], kchars[k][0] == "(" and kchars[k + 1][0] == ")")
                         for k in range(len(kchars) - 1)]
                gaps.append((kchars[-1][2][2], x1, False))
                for g0, g1, paren in gaps:
                    if g1 - g0 > 0.9 * cw or paren:
                        got = _read_gap(img, kchars, g0, g1, y0, y1, h, paren)
                        if got and not any(abs(got[1] - a[1]) < 1 for a in adds):
                            adds.append(got); spans.append((g0, g1))
                if len(adds) >= 2:
                    adds, spans, drop = _sandwiched(img, kchars, adds, spans, y0, y1, h, cw)
            if adds:
                txt = _merge(txt, kchars, adds, drop) or txt
        except Exception as e:  # noqa: BLE001 — 한자 읽기가 실패해도 한글 결과는 지킨다
            logger.debug(f"한자 읽기 생략: {e}")
        if txt:
            lines.append(txt); boxes.append(result.boxes[i]); scores.append(score)
    return lines, boxes, scores
