"""
[INPUT]: 依赖标准库 urllib / json / os / pathlib，core/ids 的 digest
[OUTPUT]: 对外提供 LLMClient 协议、LLMUnavailable、GeminiClient、CachedLLM、llm_from_env()
[POS]: language 的模型接入层；密钥只从环境变量读取，磁盘缓存让开发期重复调用零成本。
       所有调用方都必须在 LLMUnavailable 时回退到确定性模板——模型是锦上添花，不是承重墙
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Protocol

from tianlong.core.ids import digest

log = logging.getLogger(__name__)


class LLMUnavailable(Exception):
    """没有配置模型、网络失败、配额耗尽、输出无法解析：调用方一律回退。"""


class LLMClient(Protocol):
    model: str

    def generate(
        self, prompt: str, *, system: str | None = None, schema: dict[str, Any] | None = None,
        temperature: float = 0.4,
    ) -> str: ...


# ============================================================
#  Gemini（generateContent REST，无 SDK 依赖）
# ============================================================

_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class GeminiClient:
    def __init__(self, api_key: str, model: str = "gemini-flash-latest", timeout: float = 30.0) -> None:
        if not api_key:
            raise LLMUnavailable("缺少 GEMINI_API_KEY")
        self._key = api_key
        self.model = model
        self.timeout = timeout

    def generate(
        self, prompt: str, *, system: str | None = None, schema: dict[str, Any] | None = None,
        temperature: float = 0.4,
    ) -> str:
        config: dict[str, Any] = {"temperature": temperature}
        if schema is not None:
            config["responseMimeType"] = "application/json"
            config["responseSchema"] = schema
        body: dict[str, Any] = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": config}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        req = urllib.request.Request(
            _ENDPOINT.format(model=self.model),
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-goog-api-key": self._key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            detail = e.read().decode("utf-8", "replace")[:300] if isinstance(e, urllib.error.HTTPError) else str(e)
            raise LLMUnavailable(f"Gemini 调用失败: {detail}") from e
        try:
            parts = payload["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
        except (KeyError, IndexError) as e:
            raise LLMUnavailable(f"Gemini 返回无内容: {str(payload)[:300]}") from e


# ============================================================
#  磁盘缓存：同一 (模型, 系统提示, 提示, 模式, 温度) 只花一次钱
# ============================================================


class CachedLLM:
    def __init__(self, inner: LLMClient, cache_dir: Path | str = ".cache/llm") -> None:
        self.inner = inner
        self.model = inner.model
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self, prompt: str, *, system: str | None = None, schema: dict[str, Any] | None = None,
        temperature: float = 0.4,
    ) -> str:
        key = digest(self.model, system, prompt, json.dumps(schema, sort_keys=True), temperature)
        path = self.dir / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text("utf-8"))["text"]
        text = self.inner.generate(prompt, system=system, schema=schema, temperature=temperature)
        path.write_text(json.dumps({"model": self.model, "text": text}, ensure_ascii=False), "utf-8")
        return text


def llm_from_env(cache_dir: Path | str = ".cache/llm") -> LLMClient | None:
    """有 GEMINI_API_KEY 则返回带缓存的 Gemini 客户端，否则返回 None（全链路走模板）。"""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        return None
    model = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
    return CachedLLM(GeminiClient(key, model), cache_dir)


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
