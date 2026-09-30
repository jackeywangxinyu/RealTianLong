"""
[INPUT]: 依赖标准库 http.client / ssl / urllib / json / contextlib / math / os / threading / time / pathlib，core/ids 的 digest；
         llm_from_env() 里才导入 language/hedge 的 HedgedLLM（hedge 依赖本模块，延迟导入免得循环）
[OUTPUT]: 对外提供 LLMClient 协议（generate + stream）、LLMUnavailable、CallStat（每次调用的首字与总耗时）、
          GeminiClient（REST：长连接、思考档位逐级降档、SSE 流式且认结束原因、型号不可用时退回别名、忙时快速重试一次）、
          CachedLLM（磁盘缓存，开发/评测用；原子写入，坏条目算没命中）、
          ScriptedLLM（离线脚本回放：可模拟延迟与流式，测试与评测管道用）、
          llm_from_env()（叙述模型，默认包一层首字对冲）/ fast_llm_from_env()（解释用的快模型）、parse_json()
[POS]: language 的模型接入层；密钥只从环境变量读取，只放在请求头里，异常信息里抹掉。速度是主持体验的一半：显式设置思考档位
       （主持不需要深思）、每个线程复用一条 HTTPS 长连接、叙述走 streamGenerateContent 边生成边交付、主模型迟迟不出字就请快模型替补，
       每次调用记下首字与总耗时。慢的失败不重发：超时绝不重发（请求已送达，重发只会让等待与计费翻倍），只有闲置长连接被对端关掉才重连一次。
       GeminiClient 对外只抛 LLMUnavailable——传输中断、半截正文、坏编码、流内错误、SAFETY/RECITATION/MAX_TOKENS 截断都算；
       所有调用方都必须在 LLMUnavailable 时回退到确定性模板——模型是锦上添花，不是承重墙
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import base64
import contextlib
import http.client
import json
import logging
import math
import os
import ssl
import threading
import time
import urllib.parse
import urllib.request
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from tianlong.core.ids import digest

log = logging.getLogger(__name__)


class LLMUnavailable(Exception):
    """没有配置模型、网络失败、配额耗尽、输出无法解析：调用方一律回退。"""


@dataclass(frozen=True, slots=True)
class CallStat:
    model: str
    kind: str              # generate / stream
    first_ms: float | None  # 首个文字到达（generate 与 total 相同）
    total_ms: float
    chars: int
    ok: bool


class LLMClient(Protocol):
    model: str

    def generate(
        self, prompt: str, *, system: str | None = None, schema: dict[str, Any] | None = None,
        temperature: float | None = None, max_tokens: int | None = None,
    ) -> str: ...

    def stream(self, prompt: str, *, system: str | None = None, max_tokens: int | None = None) -> Iterator[str]: ...


class _Stats:
    """最近若干次调用的耗时（线程安全、有界）：评测读它，玩家看不见。"""

    def __init__(self, keep: int = 512) -> None:
        self._lock = threading.Lock()
        self.calls: deque[CallStat] = deque(maxlen=keep)

    def add(self, stat: CallStat) -> None:
        with self._lock:
            self.calls.append(stat)

    def snapshot(self) -> list[CallStat]:
        with self._lock:
            return list(self.calls)


# ============================================================
#  Gemini（REST，无 SDK 依赖）
# ============================================================

_HOST = "generativelanguage.googleapis.com"
_FALLBACK = {"lite": "gemini-flash-lite-latest", "flash": "gemini-flash-latest"}
# 复用的长连接已被对端关掉：请求没送出去，或一个字节都没收到（RemoteDisconnected 是 ConnectionResetError 的子类）。
# 只有这几种才换条新连接重发一次；超时不在其中——请求已经送达，对端可能还在算、照样计费
_STALE = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, ssl.SSLEOFError, http.client.CannotSendRequest)
_BUSY = frozenset({429, 500, 503})
_BACKOFF = 0.2           # 忙时重试前稍等（秒）
_AUTO = object()         # 思考字段按 (型号, 档位) 推出；接口拒绝之后换成降档后的写法


def _thinking_config(model: str, level: str | None) -> dict[str, Any] | None:
    """主持不需要深思：3.x 用 thinkingLevel，2.5 用 thinkingBudget（lite 默认不思考）。level=None 用模型默认。"""
    if not level:
        return None
    if model.startswith("gemini-2.5"):
        return {"thinkingBudget": 0 if level in ("minimal", "low") and "pro" not in model else 128}
    return {"thinkingLevel": level}


def _step_down(sent: dict[str, Any]) -> dict[str, Any] | None:
    """思考字段被拒时的下一档：先退到 low，再换成 2.5 的预算写法，最后才去掉——去掉就是模型默认，3.x 默认深思，最慢。"""
    level = sent.get("thinkingLevel")
    if level and level != "low":
        return {"thinkingLevel": "low"}
    if level:
        return {"thinkingBudget": 0}
    return None


def _retry_after(resp: http.client.HTTPResponse) -> float:
    """Retry-After 的秒数；没给算 0，给的是日期或看不懂算无穷（不重试）。"""
    value = resp.getheader("Retry-After")
    if not value:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return math.inf


def _finish_of(payload: dict[str, Any]) -> str | None:
    try:
        return payload["candidates"][0].get("finishReason")
    except (KeyError, IndexError, TypeError, AttributeError):
        return None


def _text_of(payload: dict[str, Any]) -> str | None:
    """候选里的正文（去掉思考片段）；没有候选内容返回 None。"""
    try:
        parts = payload["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError):
        return None
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


class GeminiClient:
    """thinking：思考档位（minimal/low/medium/high）；接口不认所发的设置就逐级降档（→ low → thinkingBudget 0 → 去掉）重试，并记住。
    每个线程一条 HTTPS 长连接（走环境里的 HTTPS 代理时经 CONNECT 隧道）。只有复用的连接已被对端关掉（请求没送出、一个字节没收到）
    才换新连接重发一次，超时从不重发；出过错或没读完的连接一律关掉，下次重连。generate 遇到 429/500/503 且失败来得快
    （quick_retry 秒内、Retry-After 不超过 1 秒）时等 0.2 秒重试一次；stream 不重试（对冲层会立即换备用模型）。
    对外只抛 LLMUnavailable，异常信息里绝不含密钥。"""

    quick_retry = 2.0       # 首次失败在这么多秒内返回才值得重试：慢的失败再来一遍只会让等待翻倍

    def __init__(self, api_key: str, model: str = "gemini-3.8-flash", timeout: float = 30.0,
                 thinking: str | None = "low", fallback: str | None = "flash") -> None:
        if not api_key:
            raise LLMUnavailable("缺少 GEMINI_API_KEY")
        self._key = api_key
        self.model = model
        self.timeout = timeout
        self.thinking = thinking
        self._think: Any = _AUTO
        self._fallback = _FALLBACK.get(fallback or "", fallback)
        self._local = threading.local()
        self._ctx = ssl.create_default_context()
        self.stats = _Stats()
        proxy = urllib.request.getproxies().get("https")
        self._proxy = urllib.parse.urlsplit(proxy) if proxy and not urllib.request.proxy_bypass(_HOST) else None

    def _unavailable(self, message: str) -> LLMUnavailable:
        """异常信息会进日志与评测记录：对端若把密钥回显在错误正文里，这里抹掉。"""
        return LLMUnavailable(message.replace(self._key, "***"))

    # ---------------- 连接 ----------------

    def _conn(self) -> http.client.HTTPSConnection:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            return conn
        if self._proxy is not None:
            # HTTP 代理经 CONNECT 隧道：TCP 连代理，隧道建好之后再与目标做 TLS 握手
            conn = http.client.HTTPSConnection(self._proxy.hostname, self._proxy.port or 80, timeout=self.timeout,
                                               context=self._ctx)
            headers = {}
            if self._proxy.username:
                cred = f"{urllib.parse.unquote(self._proxy.username)}:{urllib.parse.unquote(self._proxy.password or '')}"
                headers["Proxy-Authorization"] = "Basic " + base64.b64encode(cred.encode()).decode()
            conn.set_tunnel(_HOST, 443, headers=headers)
        else:
            conn = http.client.HTTPSConnection(_HOST, 443, timeout=self.timeout, context=self._ctx)
        self._local.conn = conn
        return conn

    def _drop(self) -> None:
        """出错或响应没读完：这条连接不能再复用（留着半读的响应，下一次请求会写进旧连接、被执行两遍）——关掉，下次重连。"""
        conn = getattr(self._local, "conn", None)
        self._local.conn = None
        if conn is not None:
            conn.close()

    def _post(self, path: str, body: dict[str, Any]) -> http.client.HTTPResponse:
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json", "X-goog-api-key": self._key, "Connection": "keep-alive"}
        for attempt in (0, 1):
            kept = getattr(self._local, "conn", None)
            reused = kept is not None and kept.sock is not None
            conn = self._conn()
            try:
                conn.request("POST", path, body=data, headers=headers)
                return conn.getresponse()
            except (OSError, http.client.HTTPException) as e:
                self._drop()
                if attempt == 0 and reused and isinstance(e, _STALE):
                    continue                                 # 闲置的长连接已被对端关掉：换新连接重发一次
                raise self._unavailable(f"Gemini 连接失败: {type(e).__name__}: {e}") from e
        raise AssertionError("unreachable")

    # ---------------- 请求体 ----------------

    def _body(self, prompt: str, system: str | None, schema: dict[str, Any] | None, temperature: float | None,
              max_tokens: int | None, model: str) -> dict[str, Any]:
        config: dict[str, Any] = {}
        if temperature is not None and not model.startswith("gemini-3"):
            config["temperature"] = temperature           # 3.x 建议保持默认温度
        if max_tokens:
            config["maxOutputTokens"] = max_tokens
        if schema is not None:
            config["responseMimeType"] = "application/json"
            config["responseSchema"] = schema
        think = self._think
        thinking = _thinking_config(model, self.thinking) if think is _AUTO else think
        if thinking:
            config["thinkingConfig"] = dict(thinking)
        body: dict[str, Any] = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": config}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return body

    def _call(self, verb: str, body_of: Callable[[str], dict[str, Any]],
              busy_retry: bool = False) -> tuple[http.client.HTTPResponse, str]:
        """发出请求。思考设置被拒逐级降档、型号不可用退回别名、忙且失败来得快时稍等重试（busy_retry）——各有上限，并记住结论。
        降不降档看这一次真正发出去的请求体，不看共享的 self 状态（并发的另一路可能已经改过）。"""
        t0 = time.perf_counter()
        model, busy = self.model, int(busy_retry)
        for _ in range(6):                                  # 首发 + 至多三次降档 + 一次退别名 + 一次忙时重试
            body = body_of(model)
            resp = self._post(f"/v1beta/models/{model}:{verb}", body)
            if resp.status == 200:
                return resp, model
            detail = resp.read().decode("utf-8", "replace")[:400]
            sent = body["generationConfig"].get("thinkingConfig")
            if resp.status == 400 and sent and "thinking" in detail.lower():
                self._think = step = _step_down(sent)
                log.warning("模型 %s 不接受思考设置 %s，改用 %s", model, sent, step or "模型默认")
                continue
            if resp.status == 404 and self._fallback and model != self._fallback:
                log.warning("模型 %s 不可用，退回 %s", model, self._fallback)
                self.model = model = self._fallback
                continue
            if resp.status in _BUSY and busy and time.perf_counter() - t0 <= self.quick_retry \
                    and _retry_after(resp) <= 1.0:
                busy -= 1
                log.warning("模型 %s 忙（%s），%.1f 秒后重试一次", model, resp.status, _BACKOFF)
                time.sleep(_BACKOFF)
                continue
            raise self._unavailable(f"Gemini {resp.status}: {detail}")
        raise LLMUnavailable("Gemini 请求被反复拒绝")

    def _check(self, payload: Any) -> None:
        """HTTP 200 里也可能装着失败：不是 JSON 对象、流内错误事件、提示词被拒。"""
        if not isinstance(payload, dict):
            raise self._unavailable(f"Gemini 返回不是 JSON 对象: {str(payload)[:200]}")
        if "error" in payload:
            raise self._unavailable(f"Gemini 流内错误: {str(payload['error'])[:300]}")
        block = (payload.get("promptFeedback") or {}).get("blockReason")
        if block:
            raise self._unavailable(f"Gemini 拒绝提示词: {block}")

    # ---------------- 调用 ----------------

    def generate(
        self, prompt: str, *, system: str | None = None, schema: dict[str, Any] | None = None,
        temperature: float | None = None, max_tokens: int | None = None,
    ) -> str:
        t0 = time.perf_counter()
        ok, text = False, ""
        try:
            resp, _ = self._call("generateContent",
                                 lambda m: self._body(prompt, system, schema, temperature, max_tokens, m), busy_retry=True)
            payload = json.loads(resp.read().decode("utf-8"))
            self._check(payload)
            finish = _finish_of(payload)
            if finish not in (None, "STOP"):
                raise self._unavailable(f"Gemini 未正常结束: {finish}")
            got = _text_of(payload)
            if got is None:
                raise self._unavailable(f"Gemini 返回无内容: {str(payload)[:300]}")
            text = got.strip()
            ok = True
            return text
        except LLMUnavailable:
            raise
        except Exception as e:  # noqa: BLE001 —— 契约：对外只抛 LLMUnavailable（半截正文、坏编码、怪形状的 JSON 都算）
            raise self._unavailable(f"Gemini 返回无法读取: {type(e).__name__}: {e}") from e
        finally:
            if not ok:
                self._drop()
            ms = (time.perf_counter() - t0) * 1000
            self.stats.add(CallStat(self.model, "generate", ms if ok else None, ms, len(text), ok))

    def stream(self, prompt: str, *, system: str | None = None, max_tokens: int | None = None) -> Iterator[str]:
        """SSE 流：逐段交出文字增量。流内错误、提示词被拒、结束原因不是 STOP（SAFETY/RECITATION/MAX_TOKENS/OTHER……）、
        没有结束标记就断了，都在交出已收到的文字之后抛 LLMUnavailable（调用方决定保留已交付的部分还是回退）。"""
        t0 = time.perf_counter()
        first: float | None = None
        chars, ok = 0, False
        try:
            resp, _ = self._call("streamGenerateContent?alt=sse",
                                 lambda m: self._body(prompt, system, None, None, max_tokens, m))
            finish: str | None = None
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                try:
                    event = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                self._check(event)
                piece = _text_of(event)
                if piece:
                    if first is None:
                        first = (time.perf_counter() - t0) * 1000
                    chars += len(piece)
                    yield piece
                finish = _finish_of(event) or finish
            if finish != "STOP":
                raise self._unavailable(f"Gemini 流未正常结束: {finish or '没有结束标记就断了'}")
            ok = True
        except LLMUnavailable:
            raise
        except Exception as e:  # noqa: BLE001 —— 契约：对外只抛 LLMUnavailable
            raise self._unavailable(f"Gemini 流中断: {type(e).__name__}: {e}") from e
        finally:
            if not ok:
                # 调用方提前收手（生成器被关闭）或中途出错：响应没读完，这条连接不能再复用——关掉，下次重连
                self._drop()
            self.stats.add(CallStat(self.model, "stream", first, (time.perf_counter() - t0) * 1000, chars, ok))


# ============================================================
#  磁盘缓存：同一 (模型, 系统提示, 提示, 模式, 温度) 只花一次钱——只在开发与评测时显式打开，
#  游戏里默认关闭：同一提示词回放同一段文字，读起来就是重复
# ============================================================


class CachedLLM:
    def __init__(self, inner: LLMClient, cache_dir: Path | str = ".cache/llm") -> None:
        self.inner = inner
        self.model = inner.model
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, *parts: Any) -> Path:
        return self.dir / f"{digest(self.model, *parts)}.json"

    def _load(self, path: Path) -> str | None:
        """命中返回文字；没有算没命中，坏条目（写到一半被打断、被改坏）也算没命中并删掉，重新向模型要。"""
        try:
            text = json.loads(path.read_text("utf-8"))["text"]
        except FileNotFoundError:
            return None
        except (OSError, ValueError, KeyError, TypeError):
            text = None
        if isinstance(text, str):
            return text
        log.warning("缓存条目损坏，作废重取：%s", path.name)
        path.unlink(missing_ok=True)
        return None

    def _save(self, path: Path, text: str) -> None:
        """先写临时文件再原子替换：写到一半被打断也不会留下半截 JSON。写不进去只记一笔，不妨碍这次调用。"""
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            tmp.write_text(json.dumps({"model": self.model, "text": text}, ensure_ascii=False), "utf-8")
            os.replace(tmp, path)
        except OSError as e:
            log.warning("缓存写入失败：%s", e)
        finally:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)                 # 替换成功后它已不在；被打断时清掉半截

    def generate(
        self, prompt: str, *, system: str | None = None, schema: dict[str, Any] | None = None,
        temperature: float | None = None, max_tokens: int | None = None,
    ) -> str:
        path = self._path(system, prompt, json.dumps(schema, sort_keys=True), temperature)
        hit = self._load(path)
        if hit is not None:
            return hit
        extra = {"max_tokens": max_tokens} if max_tokens else {}       # 老式客户端不认 max_tokens：没给就不传
        text = self.inner.generate(prompt, system=system, schema=schema, temperature=temperature, **extra)
        self._save(path, text)
        return text

    def stream(self, prompt: str, *, system: str | None = None, max_tokens: int | None = None) -> Iterator[str]:
        path = self._path("stream", system, prompt)
        hit = self._load(path)
        if hit is not None:
            yield hit
            return
        pieces = []
        for piece in self.inner.stream(prompt, system=system, max_tokens=max_tokens):
            pieces.append(piece)
            yield piece
        self._save(path, "".join(pieces))                    # 只有整段流完才写：中途失败或被作废的不进缓存


# ============================================================
#  离线脚本回放：respond(prompt, system, schema) → 文字；可模拟首字延迟、吐字速度与失败
# ============================================================


class ScriptedLLM:
    def __init__(self, respond: Callable[[str, str | None, dict[str, Any] | None], str], model: str = "scripted",
                 first_delay: float = 0.0, chars_per_second: float = 0.0, chunk: int = 6) -> None:
        self.respond = respond
        self.model = model
        self.first_delay = first_delay
        self.cps = chars_per_second
        self.chunk = chunk
        self.stats = _Stats()
        self.prompts: list[tuple[str | None, str]] = []      # (system, prompt)：测试据此断言模型看到了什么

    def generate(
        self, prompt: str, *, system: str | None = None, schema: dict[str, Any] | None = None,
        temperature: float | None = None, max_tokens: int | None = None,
    ) -> str:
        t0 = time.perf_counter()
        self.prompts.append((system, prompt))
        text = self.respond(prompt, system, schema)
        time.sleep(self.first_delay + (len(text) / self.cps if self.cps else 0.0))
        ms = (time.perf_counter() - t0) * 1000
        self.stats.add(CallStat(self.model, "generate", ms, ms, len(text), True))
        return text

    def stream(self, prompt: str, *, system: str | None = None, max_tokens: int | None = None) -> Iterator[str]:
        t0 = time.perf_counter()
        self.prompts.append((system, prompt))
        text = self.respond(prompt, system, None)
        time.sleep(self.first_delay)
        first = (time.perf_counter() - t0) * 1000
        for i in range(0, len(text), self.chunk):
            piece = text[i:i + self.chunk]
            if self.cps:
                time.sleep(len(piece) / self.cps)
            yield piece
        self.stats.add(CallStat(self.model, "stream", first, (time.perf_counter() - t0) * 1000, len(text), True))


# ============================================================
#  从环境变量装配：叙述模型与快模型分开；叙述模型默认包一层首字对冲；缓存只在 TIANLONG_LLM_CACHE=1 时打开
# ============================================================

HEDGE_AFTER = 2.5        # GEMINI_HEDGE_AFTER 的默认值（秒）


def _caching(cache_dir: Path | str | None) -> bool:
    return os.environ.get("TIANLONG_LLM_CACHE", "") == "1" and cache_dir is not None


def _client(model_var: str, default_model: str, thinking_var: str, default_thinking: str, fallback: str,
            timeout: float, cache_dir: Path | str | None) -> LLMClient | None:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        return None
    model = os.environ.get(model_var, "").strip() or default_model
    thinking = os.environ.get(thinking_var, "").strip() or default_thinking
    client: LLMClient = GeminiClient(key, model, timeout=timeout, thinking=None if thinking == "default" else thinking,
                                     fallback=fallback)
    if _caching(cache_dir):
        client = CachedLLM(client, cache_dir)
    return client


def _hedge_after() -> float:
    """GEMINI_HEDGE_AFTER：主模型多少秒没出字就请替补；没设用默认，看不懂的值记一笔后用默认，0（或负数）表示不对冲。"""
    raw = os.environ.get("GEMINI_HEDGE_AFTER", "").strip()
    if not raw:
        return HEDGE_AFTER
    try:
        after = float(raw)
    except ValueError:
        after = math.nan
    if math.isnan(after) or math.isinf(after):
        log.warning("GEMINI_HEDGE_AFTER=%r 不是有限的秒数，按默认 %.1f 秒", raw, HEDGE_AFTER)
        return HEDGE_AFTER
    return after


def llm_from_env(cache_dir: Path | str | None = ".cache/llm") -> LLMClient | None:
    """叙述模型（GEMINI_MODEL，默认 gemini-3.8-flash，思考档位 GEMINI_THINKING 默认 low）；没有密钥返回 None（全链路走模板）。
    默认包一层首字对冲 HedgedLLM：主模型 GEMINI_HEDGE_AFTER 秒（默认 2.5）内没出字或提前出错，就把同一请求发给一个与
    fast_llm_from_env() 同样装配的独立快模型；GEMINI_HEDGE_AFTER=0 只用主模型。
    开了缓存（TIANLONG_LLM_CACHE=1）就不对冲：缓存是为了逐字复现与重跑不花钱，谁先出字取决于网速，对冲会让重跑换一段文字、重新花钱。"""
    primary = _client("GEMINI_MODEL", "gemini-3.8-flash", "GEMINI_THINKING", "low", "flash", 30.0, cache_dir)
    if primary is None or _caching(cache_dir):
        return primary
    after = _hedge_after()
    if after <= 0:
        return primary
    backup = fast_llm_from_env(cache_dir)
    if backup is None:
        return primary
    # hedge 依赖本模块：用到时才导入，免得循环
    from tianlong.language.hedge import HedgedLLM

    return HedgedLLM(primary, backup, after=after)


def fast_llm_from_env(cache_dir: Path | str | None = ".cache/llm") -> LLMClient | None:
    """解释玩家输入用的快模型（GEMINI_FAST_MODEL，默认 gemini-3.1-flash-lite，思考档位 GEMINI_FAST_THINKING 默认 minimal）。"""
    return _client("GEMINI_FAST_MODEL", "gemini-3.1-flash-lite", "GEMINI_FAST_THINKING", "minimal", "lite", 15.0,
                   cache_dir)


def parse_json(text: str) -> Any:
    """容忍模型把 JSON 包进 ```json 代码块。"""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    try:
        return json.loads(t)
    except json.JSONDecodeError as e:
        raise LLMUnavailable(f"无法解析模型 JSON: {text[:200]}") from e
