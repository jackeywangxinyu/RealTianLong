"""
[INPUT]: 依赖标准库 http.client / ssl / urllib / json / os / threading / time / pathlib，core/ids 的 digest
[OUTPUT]: 对外提供 LLMClient 协议（generate + stream）、LLMUnavailable、CallStat（每次调用的首字与总耗时）、
          GeminiClient（REST：长连接、思考档位、SSE 流式、型号不可用时退回别名）、CachedLLM（磁盘缓存，开发/评测用）、
          ScriptedLLM（离线脚本回放：可模拟延迟与流式，测试与评测管道用）、
          llm_from_env()（叙述模型）/ fast_llm_from_env()（解释用的快模型）、parse_json()
[POS]: language 的模型接入层；密钥只从环境变量读取。速度是主持体验的一半：显式设置思考档位（主持不需要深思）、
       每个线程复用一条 HTTPS 长连接、叙述走 streamGenerateContent 边生成边交付，每次调用记下首字与总耗时。
       所有调用方都必须在 LLMUnavailable 时回退到确定性模板——模型是锦上添花，不是承重墙
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import base64
import http.client
import json
import logging
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


def _thinking_config(model: str, level: str | None) -> dict[str, Any] | None:
    """主持不需要深思：3.x 用 thinkingLevel，2.5 用 thinkingBudget（lite 默认不思考）。level=None 用模型默认。"""
    if not level:
        return None
    if model.startswith("gemini-2.5"):
        return {"thinkingBudget": 0 if level in ("minimal", "low") and "pro" not in model else 128}
    return {"thinkingLevel": level}


class GeminiClient:
    """thinking：思考档位（minimal/low/medium/high；接口不认这个字段时去掉后重试一次，并记住）。
    每个线程一条 HTTPS 长连接（走环境里的 HTTPS 代理时经 CONNECT 隧道）；连接失效自动重连一次。"""

    def __init__(self, api_key: str, model: str = "gemini-3.8-flash", timeout: float = 30.0,
                 thinking: str | None = "low", fallback: str | None = "flash") -> None:
        if not api_key:
            raise LLMUnavailable("缺少 GEMINI_API_KEY")
        self._key = api_key
        self.model = model
        self.timeout = timeout
        self.thinking = thinking
        self._fallback = _FALLBACK.get(fallback or "", fallback)
        self._local = threading.local()
        self._ctx = ssl.create_default_context()
        self.stats = _Stats()
        proxy = urllib.request.getproxies().get("https")
        self._proxy = urllib.parse.urlsplit(proxy) if proxy and not urllib.request.proxy_bypass(_HOST) else None

    # ---------------- 连接 ----------------

    def _conn(self, fresh: bool = False) -> http.client.HTTPSConnection:
        conn = getattr(self._local, "conn", None)
        if conn is not None and not fresh:
            return conn
        if conn is not None:
            conn.close()
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

    def _post(self, path: str, body: dict[str, Any]) -> http.client.HTTPResponse:
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json", "X-goog-api-key": self._key, "Connection": "keep-alive"}
        for attempt in (0, 1):
            conn = self._conn(fresh=attempt > 0)
            try:
                conn.request("POST", path, body=data, headers=headers)
                return conn.getresponse()
            except (OSError, http.client.HTTPException) as e:
                if attempt:
                    raise LLMUnavailable(f"Gemini 连接失败: {e}") from e     # 复用的连接可能已被对端关闭：重连一次
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
        thinking = _thinking_config(model, self.thinking)
        if thinking:
            config["thinkingConfig"] = thinking
        body: dict[str, Any] = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": config}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return body

    def _call(self, verb: str, body_of: Callable[[str], dict[str, Any]]) -> tuple[http.client.HTTPResponse, str]:
        """发出请求；思考字段被拒去掉重试，型号不可用退回别名重试——各至多一次，并记住结论。"""
        model = self.model
        for _ in range(3):
            path = f"/v1beta/models/{model}:{verb}"
            resp = self._post(path, body_of(model))
            if resp.status == 200:
                return resp, model
            detail = resp.read().decode("utf-8", "replace")[:400]
            if resp.status == 400 and "thinking" in detail.lower() and self.thinking:
                log.warning("模型 %s 不接受思考档位，改用默认：%s", model, detail[:120])
                self.thinking = None
                continue
            if resp.status == 404 and self._fallback and model != self._fallback:
                log.warning("模型 %s 不可用，退回 %s", model, self._fallback)
                self.model = model = self._fallback
                continue
            raise LLMUnavailable(f"Gemini {resp.status}: {detail}")
        raise LLMUnavailable("Gemini 请求被反复拒绝")

    # ---------------- 调用 ----------------

    def generate(
        self, prompt: str, *, system: str | None = None, schema: dict[str, Any] | None = None,
        temperature: float | None = None, max_tokens: int | None = None,
    ) -> str:
        t0 = time.perf_counter()
        ok, text = False, ""
        try:
            resp, _ = self._call("generateContent",
                                 lambda m: self._body(prompt, system, schema, temperature, max_tokens, m))
            try:
                payload = json.loads(resp.read().decode("utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                raise LLMUnavailable(f"Gemini 返回无法解析: {e}") from e
            text = _text_of(payload).strip()
            ok = True
            return text
        finally:
            ms = (time.perf_counter() - t0) * 1000
            self.stats.add(CallStat(self.model, "generate", ms if ok else None, ms, len(text), ok))

    def stream(self, prompt: str, *, system: str | None = None, max_tokens: int | None = None) -> Iterator[str]:
        """SSE 流：逐段交出文字增量。中途失败抛 LLMUnavailable（调用方决定保留已交付的部分还是回退）。"""
        t0 = time.perf_counter()
        first: float | None = None
        chars, ok = 0, False
        try:
            resp, _ = self._call("streamGenerateContent?alt=sse",
                                 lambda m: self._body(prompt, system, None, None, max_tokens, m))
            try:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    try:
                        piece = _text_of(json.loads(line[5:].strip()), strict=False)
                    except json.JSONDecodeError:
                        continue
                    if piece:
                        if first is None:
                            first = (time.perf_counter() - t0) * 1000
                        chars += len(piece)
                        yield piece
            except (OSError, http.client.HTTPException) as e:
                raise LLMUnavailable(f"Gemini 流中断: {e}") from e
            ok = True
        finally:
            if not ok:
                # 调用方提前收手（生成器被关闭）或中途出错：响应没读完，这条连接不能再复用——关掉，下次重连
                conn = getattr(self._local, "conn", None)
                if conn is not None:
                    conn.close()
                self._local.conn = None
            self.stats.add(CallStat(self.model, "stream", first, (time.perf_counter() - t0) * 1000, chars, ok))


def _text_of(payload: dict[str, Any], strict: bool = True) -> str:
    try:
        parts = payload["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError) as e:
        if strict:
            raise LLMUnavailable(f"Gemini 返回无内容: {str(payload)[:300]}") from e
        return ""
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


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

    def generate(
        self, prompt: str, *, system: str | None = None, schema: dict[str, Any] | None = None,
        temperature: float | None = None, max_tokens: int | None = None,
    ) -> str:
        path = self._path(system, prompt, json.dumps(schema, sort_keys=True), temperature)
        if path.exists():
            return json.loads(path.read_text("utf-8"))["text"]
        extra = {"max_tokens": max_tokens} if max_tokens else {}       # 老式客户端不认 max_tokens：没给就不传
        text = self.inner.generate(prompt, system=system, schema=schema, temperature=temperature, **extra)
        path.write_text(json.dumps({"model": self.model, "text": text}, ensure_ascii=False), "utf-8")
        return text

    def stream(self, prompt: str, *, system: str | None = None, max_tokens: int | None = None) -> Iterator[str]:
        path = self._path("stream", system, prompt)
        if path.exists():
            yield json.loads(path.read_text("utf-8"))["text"]
            return
        pieces = []
        for piece in self.inner.stream(prompt, system=system, max_tokens=max_tokens):
            pieces.append(piece)
            yield piece
        path.write_text(json.dumps({"model": self.model, "text": "".join(pieces)}, ensure_ascii=False), "utf-8")


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
#  从环境变量装配：叙述模型与快模型分开；缓存只在 TIANLONG_LLM_CACHE=1 时打开
# ============================================================


def _client(model_var: str, default_model: str, thinking_var: str, default_thinking: str, fallback: str,
            timeout: float, cache_dir: Path | str | None) -> LLMClient | None:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        return None
    model = os.environ.get(model_var, "").strip() or default_model
    thinking = os.environ.get(thinking_var, "").strip() or default_thinking
    client: LLMClient = GeminiClient(key, model, timeout=timeout, thinking=None if thinking == "default" else thinking,
                                     fallback=fallback)
    if os.environ.get("TIANLONG_LLM_CACHE", "") == "1" and cache_dir is not None:
        client = CachedLLM(client, cache_dir)
    return client


def llm_from_env(cache_dir: Path | str | None = ".cache/llm") -> LLMClient | None:
    """叙述模型（GEMINI_MODEL，默认 gemini-3.8-flash，思考档位 GEMINI_THINKING 默认 low）；没有密钥返回 None（全链路走模板）。"""
    return _client("GEMINI_MODEL", "gemini-3.8-flash", "GEMINI_THINKING", "low", "flash", 30.0, cache_dir)


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
