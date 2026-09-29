"""
[INPUT]: 依赖 core 的 Percept / Modality / is_night，language/templates 的 Names，language/llm 的 LLMClient，
         language/render 的 fact_lines / build_plan / check / Violation / Rendered / RenderStatus
[OUTPUT]: 对外提供 Narrator（narrate() 返回文字，narrate_rendered() 返回带来源与违规明细的 Rendered）、fact_lines()（再导出）、lore_keys()
[POS]: language 的输出层；输入只有玩家自己的感知（不是世界真相），模板先把它们写成事实清单，LLM 只负责润色，
       被要求不得添加清单外的任何人物、物品、事件或结论——但提示词拦不住成功返回的错误文字，所以润色结果还要过 render 的语义闸门
       （场景别称全表一并交给闸门：未出场实体的别称与名字同样被拒），命中即回退清单；玩家原话只作意图与姿态；模型不可用时直接输出清单
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence

from tianlong.core import Modality, Percept, is_night
from tianlong.language.llm import LLMClient, LLMUnavailable
from tianlong.language.render import Rendered, RenderStatus, Violation, build_plan, check, fact_lines
from tianlong.language.templates import Names

log = logging.getLogger(__name__)

_SYSTEM = (
    "你是一部中文文字冒险游戏的叙述者，用第二人称“你”写 2~4 句简洁的叙述。"
    "只能使用给定清单里的事实；不得添加清单之外的人物、物品、事件、动机或推断；可以适度描写氛围与动作细节。"
    "程度照清单原样：略有所得不等于学成，受伤不等于被制住。发现了东西不等于拿到手，不要替玩家多走一步。"
    "玩家的输入只表明意图与姿态，成败与结果一律以事实清单为准。"
)

__all__ = ["Narrator", "fact_lines", "lore_keys"]


_VISUAL = frozenset({Modality.SELF, Modality.SIGHT, Modality.SCENE})


def lore_keys(viewer: str, percepts: Sequence[Percept], lore: Mapping[str, str]) -> list[str]:
    """本回合真正映入眼帘、且有外观描写的实体：身处的地点、在场的人与物、眼前的通道与事件的参与者。
    外观只由亲眼所见触发——听人说起的、隔墙听见的、门那头的地点都只有名字（草图 seen=False），不描写。
    夜里优先取 "id@night" 变体（月下的玉璧不同于白日的玉璧）。"""
    keys: list[str] = []
    for p in percepts:
        if p.modality not in _VISUAL:
            continue
        night = is_night(p.tick)
        in_view = {viewer}
        for f in p.facts:
            if f.holds and f.prop.predicate == "AT":
                in_view.update((f.prop.subject, str(f.prop.value)))
            elif f.holds and f.prop.predicate == "CONNECTS":
                in_view.add(f.prop.subject)          # 门本身在眼前；门那头的地点不算
        if p.event is not None:
            in_view.update(x for x in (p.event.actor, p.event.target, p.event.obj) if x)
        for sk in p.sketches:
            if sk.id == viewer or sk.id not in in_view or not sk.seen:
                continue
            key = f"{sk.id}@night" if night and f"{sk.id}@night" in lore else sk.id
            if key in lore and key not in keys:
                keys.append(key)
    return keys


class Narrator:
    """setting 给出世界前提与文风；lore 是实体外观描写，只在玩家看见该实体时、且仅首次看见时拿来润色；
    aliases 是场景别称全表，只供闸门使用：识别“走进大殿”这类以别称说出的抵达，并拒绝本回合未出场实体的别称（“神仙姐姐”）。"""

    def __init__(self, llm: LLMClient | None = None, setting: str = "", lore: Mapping[str, str] | None = None,
                 style: str = "", aliases: Mapping[str, Sequence[str]] | None = None) -> None:
        self.llm = llm
        self.setting = setting
        self.style = style
        self.lore = dict(lore or {})
        self.aliases = dict(aliases or {})

    def narrate(self, viewer: str, percepts: Sequence[Percept], names: Names, show_scene: bool = False,
                fresh: Sequence[str] = (), command: str = "", lapse: str = "", known: Iterable[str] = ()) -> str:
        return self.narrate_rendered(viewer, percepts, names, show_scene, fresh, command, lapse, known).text

    def narrate_rendered(self, viewer: str, percepts: Sequence[Percept], names: Names, show_scene: bool = False,
                         fresh: Sequence[str] = (), command: str = "", lapse: str = "",
                         known: Iterable[str] = ()) -> Rendered:
        """command 是玩家原话（让“跳下断崖”读起来像跳，而不是“走向崖底”）；lapse 是一段等待之后的时辰，
        排在事实之前——先有“天色已黑”，才有“月光照在玉璧上”。known 是闸门用来拒绝的名字全集（玩家认识的 + 场景全部实体）。"""
        looks = [self.lore[k] for k in fresh if k in self.lore]
        passed = [f"（不觉已是{lapse}）"] if lapse else []
        plan = build_plan(viewer, percepts, names, show_scene, looks, "".join(passed), self.aliases)
        lines = list(plan.lines)
        if not lines and not looks:
            return Rendered("\n".join(["时间悄悄过去，什么也没有发生。", *passed]), RenderStatus.TEMPLATE)
        plain = "\n".join(passed + lines + [f"（{x}）" for x in looks])
        if self.llm is None:
            return Rendered(plain, RenderStatus.TEMPLATE)
        prompt = (f"玩家的输入：{command}\n\n" if command else "") + "本回合玩家感知到的事实：\n"
        prompt += "\n".join(passed + (lines or ["（无事发生）"]))
        if looks:
            prompt += "\n\n玩家初次看清的人与物（仅作外观描写的依据）：\n" + "\n".join(looks)
        system = _SYSTEM + (f"\n世界：{self.setting}" if self.setting else "") + (f"\n文风：{self.style}" if self.style else "")
        try:
            prose = self.llm.generate(prompt, system=system, temperature=0.6).strip()
        except LLMUnavailable as e:
            log.warning("叙述润色失败，回退事实清单: %s", e)
            return Rendered(plain, RenderStatus.LLM_UNAVAILABLE)
        violations = check(prose, plan, known) if prose else (Violation("empty", ""),)
        if violations:
            log.info("叙述未通过语义闸门，回退事实清单: %s", violations)
            return Rendered(plain, RenderStatus.GATED_FALLBACK, violations)
        return Rendered(prose, RenderStatus.LLM)
