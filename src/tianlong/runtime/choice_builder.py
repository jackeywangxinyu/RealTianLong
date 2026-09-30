"""
[INPUT]: 玩家 BeliefStore、当前 suggestions() 与确定性规则解析器
[OUTPUT]: build_choices() 冻结当前菜单中的完整 Parsed
[POS]: 玩家决策层的生成入口，只有玩家认知，没有世界真相；先冻结执行语义，随后替换选项筛选。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from tianlong.cognition.beliefs import BeliefStore
from tianlong.language.parser import rule_parse
from tianlong.runtime.choice_model import ChoiceSpec
from tianlong.runtime.suggest import suggestions


def build_choices(me: BeliefStore, limit: int = 3) -> tuple[ChoiceSpec, ...]:
    choices = []
    for label in suggestions(me, limit):
        parsed = rule_parse(label, me)
        if parsed.candidate is not None:
            choices.append(ChoiceSpec.of(label, parsed, parsed.candidate.target))
    return tuple(choices)

