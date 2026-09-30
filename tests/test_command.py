"""
[INPUT]: 依赖 tianlong.language 的 command / parser，tianlong.cognition 的 BeliefStore，tianlong.scenarios 的 build_wuliang
[OUTPUT]: 输入语态验收（I01–I05）：否定不执行、条件不立即执行、转述不当成玩家行动、复合与疑问不走快路径、肯定指令仍走快路径、
          LLM 声称“照做”也压不过规则看见的否定、LLM 失败绝不回退到未确认的候选、LLM 回复不成形不崩也不推进、
          非即时语态不推进世界版本与时钟
[POS]: tests 的输入语义层；把“同一句话的了解、描述、计划、实际执行必须分开”写成可证伪断言
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import json

import pytest

from tianlong.cognition import BeliefStore
from tianlong.core import Op
from tianlong.language.command import SpeechMode
from tianlong.language.llm import LLMUnavailable
from tianlong.language.parser import IntentParser, rule_parse
from tianlong.scenarios import build_wuliang


class FakeLLM:
    model = "fake"

    def __init__(self, reply: dict | None = None, fail: bool = False) -> None:
        self.reply, self.fail, self.calls = reply, fail, []

    def generate(self, prompt, *, system=None, schema=None, temperature=0.4):
        self.calls.append(prompt)
        if self.fail:
            raise LLMUnavailable("boom")
        return json.dumps(self.reply)


def _attack(mode: str = "immediate", actor: str = "player") -> dict:
    return {"mode": mode, "actor": actor, "op": "attack", "target": "gongguangjie", "obj": None, "manner": "normal",
            "topic_subject": None, "topic_value": None, "topic_holds": True, "clarification": ""}


@pytest.fixture
def duanyu():
    sc = build_wuliang()
    return BeliefStore("duanyu").revise_all(sc.priors["duanyu"])[0], sc.aliases


@pytest.mark.parametrize("text, mode", [
    ("我不攻击龚光杰", SpeechMode.NEGATED),                       # I01
    ("千万别向龚光杰出手", SpeechMode.NEGATED),
    ("我绝不会对龚光杰动手", SpeechMode.NEGATED),
    ("如果龚光杰攻击我，我才还手", SpeechMode.CONDITIONAL),         # I02
    ("等龚光杰走了就向他出手", SpeechMode.CONDITIONAL),
    ("龚光杰刚刚攻击了我", SpeechMode.NARRATIVE),                  # I03
    ("龚光杰说要打我", SpeechMode.NARRATIVE),
    ("先看清楚，再决定是否拿易经", SpeechMode.CONDITIONAL),         # I04
    ("向龚光杰出手，然后去后院", SpeechMode.COMPOUND),
    ("能向龚光杰出手吗", SpeechMode.QUESTION),
])
def test_non_immediate_modes_never_yield_a_candidate(duanyu, text, mode):
    store, aliases = duanyu
    p = rule_parse(text, store, aliases)
    assert p.command.mode == mode, p.command
    assert p.candidate is None and p.clarification


@pytest.mark.parametrize("text, op", [
    ("向龚光杰出手", Op.ATTACK),                                  # I05：明确肯定的指令仍走快路径
    ("环顾四周", Op.INSPECT),
    ("等到天黑", Op.WAIT),
    ("告诉马五爷“我不在后院”", Op.TELL),                            # 引语与命题里的否定不是对行动的否定
    ("问马五爷钟姑娘在哪", Op.ASK),
])
def test_affirmative_commands_keep_the_fast_path(duanyu, text, op):
    store, aliases = duanyu
    p = rule_parse(text, store, aliases)
    assert p.command.immediate and p.candidate is not None and p.candidate.op == op and p.source == "rules"


def test_llm_cannot_override_a_detected_negation(duanyu):
    store, aliases = duanyu
    llm = FakeLLM(_attack())                     # 模型声称“玩家此刻要出手”
    p = IntentParser(llm, aliases=aliases).parse("我不攻击龚光杰", store)
    assert llm.calls and p.candidate is None


@pytest.mark.parametrize("mode, actor", [("conditional", "player"), ("narrative", "other"), ("immediate", "other")])
def test_llm_must_declare_immediate_player_action(duanyu, mode, actor):
    store, aliases = duanyu
    p = IntentParser(FakeLLM(_attack(mode, actor)), aliases=aliases).parse("如果龚光杰攻击我，我才还手", store)
    assert p.candidate is None and p.clarification


def test_llm_failure_never_falls_back_to_unconfirmed_candidate(duanyu):
    store, aliases = duanyu
    for text in ("如果龚光杰攻击我，我才还手", "龚光杰刚刚攻击了我", "我不攻击龚光杰"):
        p = IntentParser(FakeLLM(fail=True), prefer_llm=True, aliases=aliases).parse(text, store)
        assert p.candidate is None, text


@pytest.mark.parametrize("reply", [
    [1, 2],                                                                        # 不是对象
    {"mode": "immediate", "actor": "player", "op": "fly", "manner": "normal"},     # 不存在的操作
    {**_attack(), "target": ["gongguangjie"], "manner": "sideways", "clarification": ["?"]},   # 字段类型与取值都不对
])
def test_malformed_llm_reply_never_crashes_the_parser(duanyu, reply):
    store, aliases = duanyu
    p = IntentParser(FakeLLM(reply), aliases=aliases).parse("嗯，钟灵这姑娘挺有意思", store)
    assert p.candidate is None and isinstance(p.clarification, str), "回复不成形：不崩、不推进，只给追问"


def test_non_immediate_input_does_not_advance_world():
    pytest.importorskip("langgraph")
    pytest.importorskip("qdrant_client")
    from tianlong.runtime.session import GameSession

    s = GameSession(build_wuliang())
    head = s.authority.head()
    for text in ("我不攻击龚光杰", "如果龚光杰攻击我，我才还手", "龚光杰刚刚攻击了我"):
        r = s.turn(text)
        assert not r.advanced and not r.events
    after = s.authority.head()
    assert (after.version, after.clock) == (head.version, head.clock), "解析不成即时行动：世界版本与时钟都不动"
