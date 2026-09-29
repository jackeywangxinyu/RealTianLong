"""
[INPUT]: 依赖 tianlong.language 的 interpret / parser / llm（ScriptedLLM），tianlong.cognition 的 BeliefStore，tianlong.kernel 的 Kernel，
         tianlong.scenarios 的 build_wuliang
[OUTPUT]: 主持层解释器验收：GM 前缀、元指令与场外问题不调模型；高精度整句命令走 0 次模型调用的快路径，单字关键词一律交给模型；
          模型解释出的说话（言语行为、去掉包装的原话、自由发问）、姿态、溜出大殿、拿剑再刺的多步计划逐项校验后成立；
          编造的物品（掏出北冥神功）、陌生 id、不在手里的东西被拒且不推进；多步计划在第一个不成立的步骤处截断；
          规则看见的否定与非即时语态压过模型；提示词只含玩家认识的实体（至多 40 个）与最近两段正文；
          模型失败与模板模式退回规则解析，规则层修掉的误判（打招呼/打量不是动手、救命/挡脸不是施用、走出大殿、陌生秘籍、原话包装）
[POS]: tests 的主持层输入语义；把“模型听得懂任何话，但变不出玩家没有的东西、越不过玩家的认知”写成可证伪断言
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import json
import time

import pytest

from tianlong.cognition import BeliefStore
from tianlong.core import EntitySketch, Fact, Kind, Manner, Modality, Op, Percept, Proposition, Rel, Social
from tianlong.kernel import Kernel
from tianlong.language.interpret import MAX_TABLE, Interpreter
from tianlong.language.llm import LLMUnavailable, ScriptedLLM
from tianlong.language.parser import MoveKind
from tianlong.scenarios import build_wuliang

from .conftest import make_intent

# ============================================================
#  夹具：段誉的开场认知；模型按脚本回放固定 JSON
# ============================================================


def _reply(**fields) -> str:
    base = {"kind": "unclear", "mode": "immediate", "actor": "player", "steps": [], "listener": None, "speech": "tell",
            "line": "", "social": "none", "topic_subject": None, "topic_value": None, "topic_holds": True,
            "missing": "", "reply": ""}
    base.update(fields)
    return json.dumps(base, ensure_ascii=False)


def _step(op: str, target: str | None = None, obj: str | None = None, manner: str = "normal") -> dict:
    return {"op": op, "target": target, "obj": obj, "manner": manner}


def _llm(**fields) -> ScriptedLLM:
    text = _reply(**fields)
    return ScriptedLLM(lambda prompt, system, schema: text, model="fast")


@pytest.fixture(scope="module")
def world():
    return build_wuliang()


@pytest.fixture
def duanyu(world):
    return BeliefStore("duanyu").revise_all(world.priors["duanyu"])[0]


def _interp(world, llm=None) -> Interpreter:
    return Interpreter(llm, world.aliases)


def _walk(world, *moves: tuple[str, str]) -> BeliefStore:
    """让段誉按真实内核一步步走过去，认知只由他自己的感知折叠而成。"""
    state = world.state
    store = BeliefStore("duanyu").revise_all(world.priors["duanyu"])[0]
    for target, door in moves:
        r = Kernel().step(state, [make_intent("duanyu", Op.MOVE, target, door, based_on=state.version)])
        state = r.state
        store = store.revise_all(o.percept for o in r.observations if o.observer == "duanyu")[0]
    assert store.location_of("duanyu") == moves[-1][0]
    return store


# ============================================================
#  前缀、元指令、场外问题：不调模型、不推进时间
# ============================================================


@pytest.mark.parametrize("text, kind, question", [
    ("GM：我该做什么", MoveKind.ASK_GM, "我该做什么"),
    ("gm: 这局怎么赢", MoveKind.ASK_GM, "这局怎么赢"),
    ("我该做什么", MoveKind.ASK_GM, "我该做什么"),
    ("我现在在哪？", MoveKind.ASK_GM, "我现在在哪？"),
    ("我身上有什么", MoveKind.ASK_GM, "我身上有什么"),
    ("现在怎么办", MoveKind.ASK_GM, "现在怎么办"),
    ("/hint", MoveKind.META, "hint"),
    ("/recap", MoveKind.META, "recap"),
    ("/beliefs", MoveKind.META, "beliefs"),
])
def test_gm_prefix_meta_and_ooc_need_no_model(world, duanyu, text, kind, question):
    llm = _llm(kind="act")
    p = _interp(world, llm).interpret(text, duanyu)
    assert (p.kind, p.question, p.candidate) == (kind, question, None)
    assert llm.prompts == [], "前缀、元指令与明显的场外问题不需要模型"


def test_unknown_meta_command_lists_the_real_ones(world, duanyu):
    p = _interp(world).interpret("/save", duanyu)
    assert p.kind == MoveKind.UNCLEAR and p.candidate is None and "/hint" in p.clarification


def test_asking_an_npc_is_not_an_ooc_question(world, duanyu):
    llm = _llm(kind="say", listener="zhongling", speech="ask", line="怎么办？", social="plead")
    p = _interp(world, llm).interpret("问钟灵怎么办", duanyu)
    assert p.kind == MoveKind.SAY and p.candidate.target == "zhongling" and len(llm.prompts) == 1


# ============================================================
#  规则快路径：整句命令 0 次模型调用；单字关键词绝不单独成立
# ============================================================


@pytest.mark.parametrize("text, op, target, obj", [
    ("环顾四周", Op.INSPECT, "hall", None),
    ("看看四周", Op.INSPECT, "hall", None),
    ("等待", Op.WAIT, None, None),
    ("等一会", Op.WAIT, None, None),
    ("等到天黑", Op.WAIT, None, None),
    ("去后院", Op.MOVE, "houyuan", "d_corridor"),
    ("我去后院。", Op.MOVE, "houyuan", "d_corridor"),
    ("研读易经", Op.STUDY, "yijing", None),
    ("拿起长剑", Op.TAKE, "sword", None),
    ("向龚光杰出手", Op.ATTACK, "gongguangjie", None),
    ("查看兵器架", Op.INSPECT, "swordrack", None),
    ("磕头", Op.INSPECT, "hall", None),
    ("问马五爷钟姑娘在哪", Op.ASK, "mawude", None),
    ("告诉钟灵长剑在兵器架", Op.TELL, "zhongling", None),
])
def test_fast_path_hits_make_zero_model_calls(world, duanyu, text, op, target, obj):
    llm = _llm(kind="unclear")
    p = _interp(world, llm).interpret(text, duanyu)
    assert p.candidate is not None and (p.candidate.op, p.candidate.target, p.candidate.obj) == (op, target, obj)
    assert p.source == "rules" and llm.prompts == []
    assert p.kind == (MoveKind.SAY if op in (Op.TELL, Op.ASK) else MoveKind.ACT)


def test_fast_path_details(world, duanyu):
    it = _interp(world)
    night = it.interpret("等到天黑", duanyu)
    assert night.until == "night" and it.interpret("等一会", duanyu).repeat == 10
    assert it.interpret("磕头", duanyu).candidate.social == Social.SUBMIT, "磕头是服软"
    ask = it.interpret("问马五爷钟姑娘在哪", duanyu)
    assert ask.candidate.topic == Fact(Proposition.rel("zhongling", Rel.AT, None), True)
    assert ask.utterance == "钟姑娘在哪？", "原话只引问出口的那句，不是指令包装"
    tell = it.interpret("告诉钟灵长剑在兵器架", duanyu)
    assert tell.candidate.topic == Fact(Proposition.rel("sword", Rel.AT, "swordrack"), True)


def test_fast_path_follows_the_players_map(world):
    cliff = _walk(world, ("houyuan", "d_corridor"), ("houshan", "d_backgate"), ("yading", "d_path"))
    llm = _llm(kind="unclear")
    p = _interp(world, llm).interpret("跳下断崖", cliff)
    assert (p.candidate.op, p.candidate.target, p.candidate.obj) == (Op.MOVE, "jianhu", "d_cliff")
    assert llm.prompts == []


def test_fast_path_refusals_are_in_fiction_and_do_not_advance(world, duanyu):
    llm = _llm(kind="unclear")
    it = _interp(world, llm)
    here = it.interpret("去大殿", duanyu)
    assert here.candidate is None and here.clarification == "你已经在剑湖宫大殿了。"
    assert llm.prompts == [], "认得出的整句命令做不到，也不必问模型"


@pytest.mark.parametrize("text", [
    "打他", "用易经", "下去", "走", "救她", "说吧", "学武", "读", "给他", "到外面去", "回去", "偷东西", "等",
    "打量一下钟灵", "跟钟灵打个招呼", "往外走", "研读北冥神功",
])
def test_risky_single_character_inputs_go_to_the_model(world, duanyu, text):
    llm = _llm(kind="unclear", reply="你想做什么？")
    p = _interp(world, llm).interpret(text, duanyu)
    assert len(llm.prompts) == 1, "单字关键词不在快路径上单独成立"
    assert p.candidate is None and p.source == "llm"


def test_fast_path_is_sub_millisecond_scale(world, duanyu):
    it = _interp(world)
    texts = ["环顾四周", "去后院", "研读易经", "向龚光杰出手", "等到天黑"]
    t0 = time.perf_counter()
    for _ in range(40):
        for text in texts:
            it.interpret(text, duanyu)
    mean_ms = (time.perf_counter() - t0) * 1000 / (40 * len(texts))
    assert mean_ms < 5.0, f"快路径平均 {mean_ms:.2f} ms"


# ============================================================
#  模型路径：说话、姿态、离开此地、多步计划
# ============================================================


@pytest.mark.parametrize("line", ["", "和钟灵说话"])        # 模型照抄了原句：包装剥掉之后什么也没说
def test_talking_to_zhongling_is_a_greeting(world, duanyu, line):
    llm = _llm(kind="say", listener="zhongling", speech="tell", line=line, social="greet")
    p = _interp(world, llm).interpret("和钟灵说话", duanyu)
    c = p.candidate
    assert p.kind == MoveKind.SAY and p.source == "llm"
    assert (c.op, c.target, c.topic, c.social) == (Op.TELL, "zhongling", None, Social.GREET)
    assert p.utterance is None, "没说出口具体的话，就不编一句"
    system, prompt = llm.prompts[0]
    assert "解释器" in system and prompt.endswith("玩家输入：和钟灵说话")


def test_free_question_keeps_only_the_words_asked(world, duanyu):
    # 模型没去掉包装：解释器确定性地剥掉“问左子穆”
    llm = _llm(kind="say", listener="zuozimu", speech="ask", line="问左子穆这是哪里", social="none")
    p = _interp(world, llm).interpret("问左子穆这是哪里", duanyu)
    c = p.candidate
    assert (c.op, c.target, c.topic, c.social) == (Op.ASK, "zuozimu", None, None)
    assert p.kind == MoveKind.SAY and p.utterance == "这是哪里？"


def test_say_strips_wrappers_and_keeps_a_location_claim(world, duanyu):
    llm = _llm(kind="say", listener="mawude", speech="tell", line="对马五爷说：“长剑不在兵器架上。”", social="explain",
               topic_subject="sword", topic_value="swordrack", topic_holds=False)
    p = _interp(world, llm).interpret("悄悄跟马五爷说长剑不在兵器架上", duanyu)
    c = p.candidate
    assert c.topic == Fact(Proposition.rel("sword", Rel.AT, "swordrack"), False)
    assert c.manner == Manner.CAREFUL and c.social == Social.EXPLAIN and p.utterance == "长剑不在兵器架上。"


def test_sitting_down_for_tea_is_a_gesture(world, duanyu):
    llm = _llm(kind="gesture", line="我坐下来喝了口茶", social="none")
    p = _interp(world, llm).interpret("我坐下来喝口茶", duanyu)
    assert p.kind == MoveKind.GESTURE and p.candidate.op == Op.WAIT and p.candidate.social is None
    assert p.utterance == "坐下来喝了口茶", "姿态是不带主语的动作短语"


@pytest.mark.parametrize("step", [
    _step("move", "shandao", "d_gate", "careful"),
    _step("move", "d_gate", None, "normal"),           # 只给门：去门那头；原话里的“偷偷”补上方式
    _step("move", "hall", "d_gate", "careful"),        # 给了脚下的地点加一扇门：经这扇门离开
])
def test_sneaking_out_of_the_hall_moves_through_a_known_exit(world, duanyu, step):
    llm = _llm(kind="act", steps=[step])
    p = _interp(world, llm).interpret("偷偷溜出大殿", duanyu)
    c = p.candidate
    assert p.kind == MoveKind.ACT and (c.op, c.target, c.obj, c.manner) == (Op.MOVE, "shandao", "d_gate", Manner.CAREFUL)


def test_leaving_without_a_door_asks_which_way(world, duanyu):
    llm = _llm(kind="act", steps=[_step("move", "hall")])
    p = _interp(world, llm).interpret("溜出大殿", duanyu)
    assert p.candidate is None and "回廊" in p.clarification and "剑湖宫宫门" in p.clarification


def test_take_the_sword_then_stab(world, duanyu):
    llm = _llm(kind="act", steps=[_step("take", "sword"), _step("attack", "gongguangjie")])
    p = _interp(world, llm).interpret("拿起长剑向龚光杰刺去", duanyu)
    assert (p.candidate.op, p.candidate.target) == (Op.TAKE, "sword")
    assert [(c.op, c.target) for c in p.followups] == [(Op.ATTACK, "gongguangjie")]
    assert p.kind == MoveKind.ACT


def test_pronouns_resolve_against_recent_passages(world, duanyu):
    recent = ("你来到剑湖宫大殿。", "梁上一个少女咯咯笑道：“好玩，好玩。”钟灵晃着两条腿，向你眨了眨眼。", "马五德干咳一声。")
    llm = _llm(kind="say", listener="zhongling", speech="tell", line="多谢姑娘", social="thank")
    p = _interp(world, llm).interpret("向她道谢：多谢姑娘", duanyu, recent)
    assert p.candidate.target == "zhongling" and p.candidate.social == Social.THANK
    prompt = llm.prompts[0][1]
    assert "钟灵晃着两条腿" in prompt and "马五德干咳一声" in prompt and "你来到剑湖宫大殿" not in prompt, "只附最近两段"


# ============================================================
#  校验：编造的前提、陌生 id、不在手里的东西——一律拒绝且不推进
# ============================================================


@pytest.mark.parametrize("fields", [
    {"kind": "act", "steps": [_step("study", None)], "missing": "北冥神功"},
    {"kind": "act", "steps": [_step("study", "yijing")], "missing": ""},          # 模型误读成手里那本：确定性闸门仍拦下
    {"kind": "unclear", "reply": "你想拿北冥神功做什么？"},
])
def test_conjured_item_is_refused_without_advancing(world, duanyu, fields):
    p = _interp(world, _llm(**fields)).interpret("我从怀里掏出北冥神功", duanyu)
    assert p.candidate is None and p.followups == () and p.kind == MoveKind.UNCLEAR
    assert p.clarification == "你身上并没有北冥神功。"


def test_known_item_not_in_hand_is_refused(world, duanyu):
    p = _interp(world, _llm(kind="act", steps=[_step("use", "gongguangjie", "sword")])).interpret(
        "用长剑给龚光杰治伤", duanyu)
    assert p.candidate is None and p.clarification == "你身上并没有长剑。"
    g = _interp(world, _llm(kind="gesture", line="拔出长剑", social="threaten")).interpret("我拔出长剑", duanyu)
    assert g.candidate is None and g.clarification == "你身上并没有长剑。", "姿态也变不出不在手里的兵刃"


@pytest.mark.parametrize("fields", [
    {"kind": "act", "steps": [_step("take", "scroll_bm")]},                       # 真实存在、玩家却不认识的 id
    {"kind": "act", "steps": [_step("attack", "qiaofeng")]},                      # 根本不存在的 id
    {"kind": "say", "listener": "sikongxuan", "line": "司空帮主好"},
])
def test_model_answering_with_an_unknown_id_is_rejected(world, duanyu, fields):
    llm = _llm(**fields)
    p = _interp(world, llm).interpret("拿起那卷帛书，再招呼一声", duanyu)
    assert p.candidate is None and p.kind == MoveKind.UNCLEAR
    for leaked in ("scroll_bm", "qiaofeng", "sikongxuan", "北冥", "司空"):
        assert leaked not in p.clarification, "不回显模型编出来的 id"
        assert leaked not in llm.prompts[0][1], "模型只看得见玩家认识的实体"


def test_first_failing_step_truncates_the_plan(world, duanyu):
    llm = _llm(kind="act", steps=[_step("take", "sword"), _step("attack", "qiaofeng"), _step("move", "houyuan")])
    p = _interp(world, llm).interpret("拿起长剑，砍翻乔峰，再去后院", duanyu)
    assert (p.candidate.op, p.candidate.target) == (Op.TAKE, "sword") and p.followups == ()
    first_bad = _interp(world, _llm(kind="act", steps=[_step("give", "zhongling", "sword"), _step("take", "sword")]))
    q = first_bad.interpret("把长剑交给钟灵", duanyu)
    assert q.candidate is None and q.clarification == "你身上并没有长剑。", "第一步就不成立：整句不执行，不拿后面的步骤顶替"


def test_items_taken_earlier_in_the_plan_count_as_held(world, duanyu):
    llm = _llm(kind="act", steps=[_step("take", "sword"), _step("give", "zhongling", "sword")])
    p = _interp(world, llm).interpret("拿起长剑递给钟灵", duanyu)
    assert p.candidate.op == Op.TAKE and [(c.op, c.target, c.obj) for c in p.followups] == [
        (Op.GIVE, "zhongling", "sword")]


# ============================================================
#  语态：规则看见的否定压过模型；只有即时 + 玩家才推进
# ============================================================


@pytest.mark.parametrize("fields", [
    {"kind": "act", "mode": "negated", "steps": [_step("move", "houyuan")]},
    {"kind": "act", "mode": "immediate", "steps": [_step("move", "houyuan")]},    # 模型声称“照做”
])
def test_negation_never_advances(world, duanyu, fields):
    p = _interp(world, _llm(**fields)).interpret("我不去后院", duanyu)
    assert p.candidate is None and p.followups == () and p.clarification


@pytest.mark.parametrize("mode, actor", [("conditional", "player"), ("narrative", "other"), ("immediate", "other"),
                                         ("question", "player")])
def test_model_must_declare_an_immediate_player_move(world, duanyu, mode, actor):
    llm = _llm(kind="act", mode=mode, actor=actor, steps=[_step("attack", "gongguangjie")])
    p = _interp(world, llm).interpret("如果龚光杰攻击我，我才还手", duanyu)
    assert p.candidate is None and p.clarification and p.kind == MoveKind.UNCLEAR


def test_model_failure_falls_back_to_rules_never_to_unconfirmed_candidates(world, duanyu):
    def boom(prompt, system, schema):
        raise LLMUnavailable("offline")

    it = Interpreter(ScriptedLLM(boom), world.aliases)
    for text in ("我不去后院", "如果龚光杰攻击我，我才还手", "龚光杰刚刚攻击了我"):
        assert it.interpret(text, duanyu).candidate is None, text
    greet = it.interpret("跟钟灵打个招呼", duanyu)
    assert greet.kind == MoveKind.SAY and greet.candidate.social == Social.GREET
    garbage = Interpreter(ScriptedLLM(lambda *a: "不是 JSON"), world.aliases).interpret("我不去后院", duanyu)
    assert garbage.candidate is None
    assert _interp(world).interpret("我不去后院", duanyu).candidate is None, "模板模式同样不推进"


@pytest.mark.parametrize("fields", [
    {"kind": "act", "steps": "take sword"},
    {"kind": "act", "steps": [["take", "sword"], {"op": ["take"], "target": ["sword"]}]},
    {"kind": "say", "listener": ["zhongling"], "social": ["greet"], "line": 3},
    {"kind": ["act"], "mode": ["immediate"]},
    {"kind": "gesture", "line": None, "social": {"x": 1}},
])
def test_malformed_model_output_never_crashes_or_advances(world, duanyu, fields):
    p = _interp(world, _llm(**fields)).interpret("随便做点什么吧", duanyu)
    assert p.candidate is None or (p.kind == MoveKind.GESTURE and p.candidate.op == Op.WAIT)


# ============================================================
#  提示词：只列玩家认识的实体，至多 MAX_TABLE 个，点名与在场的优先
# ============================================================


def test_prompt_lists_only_known_entities(world, duanyu):
    llm = _llm(kind="unclear")
    _interp(world, llm).interpret("四处转转", duanyu)
    system, prompt = llm.prompts[0]
    assert "钟灵/钟姑娘/灵儿" in prompt and "剑湖宫后院" in prompt and "d_corridor" in prompt, "别称、出路与门那头"
    assert "易经（yijing）" in prompt, "身上带着的东西"
    for unknown in ("司空玄", "北冥神功", "澜沧江", "无量玉璧", "神农帮"):
        assert unknown not in prompt and unknown not in system
    assert len(system) + len(prompt) < 2500


def test_prompt_table_is_capped_and_prioritised():
    far = [EntitySketch(f"i{n:02d}", Kind.ITEM, f"杂物{n:02d}") for n in range(60)]
    places = (EntitySketch("room", Kind.PLACE, "小屋"), EntitySketch("yard", Kind.PLACE, "院子"),
              EntitySketch("p", Kind.PERSON, "我自己"), EntitySketch("zz", Kind.PERSON, "赵甲"))
    facts = (Fact(Proposition.rel("p", Rel.AT, "room")), Fact(Proposition.rel("zz", Rel.AT, "room")),
             *(Fact(Proposition.rel(s.id, Rel.AT, "yard")) for s in far))
    store = BeliefStore("p").revise(Percept(0, Modality.SIGHT, None, facts, (), (*places, *far)))[0]
    llm = _llm(kind="unclear")
    Interpreter(llm).interpret("把杂物59拿过来", store)
    rows = [line for line in llm.prompts[0][1].splitlines() if line.count("|") == 3 and not line.startswith("实体表")]
    ids = [r.split("|")[0] for r in rows]
    assert len(ids) == MAX_TABLE and ids[:4] == ["p", "i59", "room", "zz"], "自己、点名的、此地、在场的优先"


# ============================================================
#  模板模式（没有模型）：退回规则解析，规则层的误判已修掉
# ============================================================


def test_template_mode_greeting_and_looking_are_not_attacks(world, duanyu):
    it = _interp(world)
    greet = it.interpret("跟钟灵打个招呼", duanyu)
    assert greet.kind == MoveKind.SAY and greet.candidate.op == Op.TELL and greet.candidate.target == "zhongling"
    assert greet.candidate.social == Social.GREET and greet.utterance is None, "只有客套没有话：不拿指令冒充原话"
    look = it.interpret("打量一下钟灵", duanyu)
    assert look.kind == MoveKind.GESTURE and look.candidate.op == Op.WAIT and look.utterance == "打量一下钟灵"
    assert it.interpret("看看钟灵", duanyu).candidate.op == Op.WAIT, "看一个人是打量，不是搜身"
    kowtow = it.interpret("向龚光杰磕头", duanyu)
    assert kowtow.candidate.op == Op.WAIT and kowtow.candidate.social == Social.SUBMIT
    frisk = it.interpret("搜龚光杰", duanyu)
    assert (frisk.candidate.op, frisk.candidate.target) == (Op.INSPECT, "gongguangjie"), "明说要搜才是搜身"


def test_template_mode_cries_and_covering_the_face_are_not_use(world, duanyu):
    it = _interp(world)
    cry = it.interpret("大喊救命", duanyu)
    assert cry.kind == MoveKind.GESTURE and cry.candidate.op == Op.WAIT and cry.candidate.social == Social.PLEAD
    face = it.interpret("我用易经挡住脸", duanyu)
    assert face.candidate is None or face.candidate.op != Op.USE
    save = it.interpret("救钟灵", duanyu)
    assert save.candidate is None, "手里只有易经：不拿它去“救”人"


def test_template_mode_leaving_is_never_a_move_into_here(world, duanyu):
    from tianlong.cognition import Candidate
    from tianlong.language.parser import normalize

    out = _interp(world).interpret("走出大殿", duanyu)
    assert out.candidate is None and "回廊" in out.clarification and "剑湖宫宫门" in out.clarification
    assert normalize(Candidate(Op.MOVE, "hall"), duanyu).target is None, "目的地就是脚下，不算移动"
    rerouted = normalize(Candidate(Op.MOVE, "houyuan", "d_gate"), duanyu)
    assert rerouted.obj == "d_corridor", "点名的门玩家并不认为通往那里：换一条认为连通的"
    door = EntitySketch("d1", Kind.DOOR, "木门")
    places = (EntitySketch("room", Kind.PLACE, "小屋"), EntitySketch("yard", Kind.PLACE, "院子"),
              EntitySketch("p", Kind.PERSON, "我自己"))
    facts = (Fact(Proposition.rel("p", Rel.AT, "room")), Fact(Proposition.rel("d1", Rel.CONNECTS, "room")),
             Fact(Proposition.rel("d1", Rel.CONNECTS, "yard")))
    store = BeliefStore("p").revise(Percept(0, Modality.SIGHT, None, facts, (), (*places, door)))[0]
    only = Interpreter(None).interpret("离开这里", store)
    assert (only.candidate.op, only.candidate.target, only.candidate.obj) == (Op.MOVE, "yard", "d1"), "只有一条路就走它"


def test_template_mode_unknown_book_is_not_swapped_for_the_one_in_hand(world, duanyu):
    it = _interp(world)
    p = it.interpret("研读北冥神功", duanyu)
    assert p.candidate is None and p.clarification == "你身上并没有北冥神功。"
    assert it.interpret("研读", duanyu).candidate.target == "yijing", "没点名时仍默认手里唯一那本"


@pytest.mark.parametrize("text, op, listener, line", [
    ("问马五爷钟姑娘在哪", Op.ASK, "mawude", "钟姑娘在哪？"),
    ("问左子穆这是哪里", Op.ASK, "zuozimu", "这是哪里？"),
    ("对钟灵说：咱们快走", Op.TELL, "zhongling", "咱们快走"),
    ("告诉钟灵“我不在后院”", Op.TELL, "zhongling", "我不在后院"),
    ("向龚光杰赔罪道：是在下失礼了", Op.TELL, "gongguangjie", "是在下失礼了"),
])
def test_template_mode_quotes_only_what_is_said(world, duanyu, text, op, listener, line):
    p = _interp(world).interpret(text, duanyu)
    assert p.kind == MoveKind.SAY and (p.candidate.op, p.candidate.target) == (op, listener) and p.utterance == line


def test_template_mode_speech_without_a_topic_is_free_speech(world, duanyu):
    it = _interp(world)
    p = it.interpret("告诉钟灵龚光杰打了我", duanyu)
    assert p.candidate.topic is None and p.utterance == "龚光杰打了我", "说不清“谁在哪”就是闲话，不硬凑命题"
    q = it.interpret("问钟灵长剑好不好看", duanyu)
    assert q.candidate.topic is None and q.candidate.op == Op.ASK


def test_taking_from_a_rack_is_not_a_conjured_premise(world, duanyu):
    assert _interp(world).interpret("从兵器架上取出长剑", duanyu).candidate.op == Op.TAKE
    llm = _llm(kind="act", steps=[_step("take", "sword")])
    p = _interp(world, llm).interpret("从兵器架上抽出一柄长剑", duanyu)
    assert (p.candidate.op, p.candidate.target) == (Op.TAKE, "sword")


def test_words_to_nobody_are_a_public_remark(world, duanyu):
    llm = _llm(kind="say", listener=None, line="谁来评评这个理？", social="plead")
    p = _interp(world, llm).interpret("我大声说：谁来评评这个理？", duanyu)
    assert p.kind == MoveKind.GESTURE and p.candidate.op == Op.WAIT and p.utterance == "说道：“谁来评评这个理？”"
    g = _interp(world, _llm(kind="gesture", line="祭出倚天剑", missing="倚天剑")).interpret("我祭出倚天剑", duanyu)
    assert g.candidate is None and g.clarification == "你身上并没有倚天剑。"
