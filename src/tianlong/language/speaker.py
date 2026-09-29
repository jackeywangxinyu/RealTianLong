"""
[INPUT]: 依赖 language/llm 的 LLMClient / LLMUnavailable，language/templates 的 render_fact，
         language/render 的 check_utterance / MIN_ALIAS，cognition 的 Candidate，core/profiles 的 Profile
[OUTPUT]: 对外提供 Speaker 协议、TemplateSpeaker、LLMSpeaker（带语义闸门与模板回退；可给场景名字全集与别称作拒绝全集）
[POS]: language 的对白渲染；把结构化言语行动（操作 + 对象 + 命题）变成符合人设的一句话。
       事实内容以命题为准，原话只是修辞——所以 LLM 润色失败时，模板说出同样的命题；
       润色成功也要过闸门：只许点名说话者、听者与话题里的人和物（名或别称），不许多出意图里没有的承诺或状态，必须提到话题主语，
       否则回退模板。拒绝全集是场景全部实体名与别称加上说话者认识的名字——说话者没听说过的真实实体同样不许点名。
       这句原话会随意图进入世界事件、被听者当作“他说过的话”记住，分歧一旦落库就无法撤回
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol

from tianlong.cognition import Candidate
from tianlong.core import Op
from tianlong.core.profiles import Profile
from tianlong.language.llm import LLMClient, LLMUnavailable
from tianlong.language.render import MIN_ALIAS, check_utterance
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


def _name(names: Names, eid: str | None) -> str | None:
    if eid is None:
        return None
    sk = names.get(eid)
    return sk.name if sk else eid


class LLMSpeaker:
    """universe 是场景全部实体名、aliases 是场景别称（实体 ID → 别称）：只用于拒绝——说话者没听说过的真实实体
    （“账簿”“琅嬛福地”）一旦被润色点名，就会作为原话落库、被听者记住，再被叙述者当作有出处的名字照搬。"""

    def __init__(self, llm: LLMClient, fallback: Speaker | None = None, universe: Iterable[str] = (),
                 aliases: Mapping[str, Sequence[str]] | None = None) -> None:
        self.llm = llm
        self.fallback = fallback or TemplateSpeaker()
        self.universe = frozenset(universe)
        self.aliases = {k: tuple(v) for k, v in (aliases or {}).items()}

    def utter(self, profile: Profile, cand: Candidate, names: Names) -> str | None:
        plain = self.fallback.utter(profile, cand, names)
        if plain is None or cand.topic is None:
            return plain
        act = "告诉对方" if cand.op == Op.TELL else "询问对方"
        listener = _name(names, cand.target)
        prompt = f"人设：{profile.persona}\n对象：{listener}\n意图：{act}“{plain}”\n对白："
        try:
            line = self.llm.generate(prompt, system=_SYSTEM, temperature=0.7).strip().strip("“”\"")
        except LLMUnavailable as e:
            log.warning("对白润色失败，回退模板: %s", e)
            return plain
        if not line:
            return plain
        # ---- 语义闸门：说话者、听者、话题主语与宾语之外的人与物不许点名；主语必须提到（本人可称“我”，听者可称“你”）----
        # 拒绝全集 = 场景全部实体名与别称 + 说话者认识的名字：说话者没听说过的实体同样不许点名
        prop = cand.topic.prop
        value = prop.value if not prop.is_attr and isinstance(prop.value, str) else None
        ids = [x for x in (profile.agent, cand.target, prop.subject, value) if x]
        allowed = {n for x in ids for n in (_name(names, x), *self.aliases.get(x, ())) if n}
        forms = [_name(names, prop.subject) or prop.subject, *self.aliases.get(prop.subject, ())]
        forms += ["我"] if prop.subject == profile.agent else []
        forms += ["你"] if prop.subject == cand.target else []
        universe = self.universe | {sk.name for sk in names.values()} | {
            a for al in self.aliases.values() for a in al if len(a) >= MIN_ALIAS}
        violations = check_utterance(line, allowed, universe, plain, forms)
        if violations:
            log.info("对白未通过语义闸门，回退模板: %s", violations)
            return plain
        return line
