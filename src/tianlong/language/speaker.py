"""
[INPUT]: 依赖 language/llm 的 LLMClient / LLMUnavailable，language/templates 的 render_fact，cognition 的 Candidate，core/profiles 的 Profile
[OUTPUT]: 对外提供 Speaker 协议、TemplateSpeaker、LLMSpeaker（带模板回退）
[POS]: language 的对白渲染；把结构化言语行动（操作 + 对象 + 命题）变成符合人设的一句话。
       事实内容以命题为准，原话只是修辞——所以 LLM 润色失败时，模板说出同样的命题
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import logging
from typing import Protocol

from tianlong.cognition import Candidate
from tianlong.core import Op
from tianlong.core.profiles import Profile
from tianlong.language.llm import LLMClient, LLMUnavailable
from tianlong.language.templates import Names, render_fact

log = logging.getLogger(__name__)


class Speaker(Protocol):
    def utter(self, profile: Profile, cand: Candidate, names: Names) -> str | None: ...


class TemplateSpeaker:
    def utter(self, profile: Profile, cand: Candidate, names: Names) -> str | None:
        if cand.op not in (Op.TELL, Op.ASK) or cand.topic is None:
            return None
        return render_fact(cand.topic, names)


_SYSTEM = (
    "你是文字游戏里的一个角色，只负责把给定的言语意图说成一句符合人设的中文对白。"
    "必须准确表达给定的命题或问题，不得添加任何新事实、新人物、新物品；不超过30字；只输出对白本身。"
)


class LLMSpeaker:
    def __init__(self, llm: LLMClient, fallback: Speaker | None = None) -> None:
        self.llm = llm
        self.fallback = fallback or TemplateSpeaker()

    def utter(self, profile: Profile, cand: Candidate, names: Names) -> str | None:
        plain = self.fallback.utter(profile, cand, names)
        if plain is None:
            return None
        act = "告诉对方" if cand.op == Op.TELL else "询问对方"
        sk = names.get(cand.target or "")
        listener = sk.name if sk else cand.target
        prompt = f"人设：{profile.persona}\n对象：{listener}\n意图：{act}“{plain}”\n对白："
        try:
            line = self.llm.generate(prompt, system=_SYSTEM, temperature=0.7).strip().strip("“”\"")
            return line or plain
        except LLMUnavailable as e:
            log.warning("对白润色失败，回退模板: %s", e)
            return plain
