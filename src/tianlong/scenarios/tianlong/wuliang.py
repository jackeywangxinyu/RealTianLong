"""
[INPUT]: 依赖 core 的实体/关系/时间/命题/感知类型，core/profiles 的 Goal / GoalKind / Profile，kernel/perception 的 make_percept / scene_percept，
         scenarios/base 的 Scenario，scenarios/tianlong/lore 的 SETTING / LORE / ALIASES
[OUTPUT]: 对外提供 build_wuliang()：天龙八部·无量山小范围世界
[POS]: scenarios/tianlong 的第一幕。以金庸《天龙八部》世纪新修版开篇为蓝本，但不写剧本——只摆好世界、角色目标与各自所知，
       剧情由规则内核与角色认知自然涌现：比剑之后龚光杰寻衅、钟灵放貂护人、左子穆护短、入夜后干葛私奔灭口、
       崖底玉璧月夜显影、琅嬛福地里的两卷帛书
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from dataclasses import replace

from tianlong.core import Entity, Fact, Kind, Modality, Proposition, Rel, Relation, WorldState, at
from tianlong.core.profiles import Goal, GoalKind, Profile
from tianlong.kernel.perception import make_percept, scene_percept
from tianlong.scenarios.base import Scenario
from tianlong.scenarios.tianlong.lore import ALIASES, HINTS, LORE, SETTING, STYLE

# ============================================================
#  地图
#
#          神农帮营地 ──林间山径── 后山(禁地) ──后山小径── 后山崖顶
#           │                      │                        │ 断崖（只能下）
#       营地下山路             后院小门                     ▼
#           │                      │                  剑湖畔（无量玉璧）
#       无量山山道 ──宫门── 剑湖宫大殿 ──回廊── 剑湖宫后院    ┊ 石缝（暗门，月夜显形）
#                                                         石洞 ──石门── 琅嬛福地 ══隧道(只能出)══▶ 澜沧江畔
# ============================================================

START = at(1, 17, 40)         # 比剑方罢，酉时将近；戌时（19:00）入夜
NIGHTFALL = at(1, 19, 0)
ELOPE = at(1, 19, 20)          # 干光豪、葛光佩入夜后动身
SECT = ("zuozimu", "xinshuangqing", "gongguangjie", "ganguanghao", "geguangpei")


def _entities() -> list[Entity]:
    P, S, It, D, H = Kind.PLACE, Kind.SURFACE, Kind.ITEM, Kind.DOOR, Kind.PERSON
    return [
        # ---- 地点 ----
        Entity.make("shandao", P, "无量山山道"),
        Entity.make("hall", P, "剑湖宫大殿"),
        Entity.make("houyuan", P, "剑湖宫后院"),
        Entity.make("houshan", P, "后山"),
        Entity.make("yading", P, "后山崖顶"),
        Entity.make("jianhu", P, "剑湖畔"),
        Entity.make("shidong", P, "石洞"),
        Entity.make("langhuan", P, "琅嬛福地"),
        Entity.make("lancang", P, "澜沧江畔"),
        Entity.make("camp", P, "神农帮营地"),
        # ---- 通道 ----
        Entity.make("d_gate", D, "剑湖宫宫门"),
        Entity.make("d_corridor", D, "回廊"),
        Entity.make("d_backgate", D, "后院小门"),
        Entity.make("d_path", D, "后山小径"),
        Entity.make("d_trail", D, "林间山径"),
        Entity.make("d_camproad", D, "营地下山路"),
        Entity.make("d_cliff", D, "断崖", oneway="jianhu"),
        Entity.make("d_cave", D, "玉璧旁的石缝", hidden=True, night_only=True, clue="yubi"),
        Entity.make("d_stonedoor", D, "石门"),
        Entity.make("d_tunnel", D, "山腹隧道", oneway="lancang"),
        # ---- 陈设 ----
        Entity.make("swordrack", S, "兵器架"),
        Entity.make("yubi", S, "无量玉璧"),
        Entity.make("statue", S, "玉像"),
        Entity.make("putuan", S, "蒲团"),
        # ---- 物件 ----
        Entity.make("mink", It, "闪电貂", small=True, weapon=True, venom=True, edge=0.45),
        Entity.make("antidote", It, "解药", small=True, cures="poisoned"),
        Entity.make("sword", It, "长剑", weapon=True, edge=0.2),
        Entity.make("yijing", It, "易经", small=True),
        Entity.make("scroll_bm", It, "北冥神功帛卷", small=True, hidden=True, teaches="absorb", difficulty=4),
        Entity.make("scroll_lb", It, "凌波微步帛卷", small=True, hidden=True, teaches="evasion", difficulty=3),
        # ---- 人物 ----
        Entity.make("duanyu", H, "段誉", martial=0.0, agility=0.5, alertness=0.4),
        Entity.make("mawude", H, "马五德", martial=0.35, agility=0.4, alertness=0.5),
        Entity.make("zuozimu", H, "左子穆", martial=0.8, agility=0.6, alertness=0.7),
        Entity.make("xinshuangqing", H, "辛双清", martial=0.75, agility=0.6, alertness=0.7),
        Entity.make("gongguangjie", H, "龚光杰", martial=0.5, agility=0.6, alertness=0.5),
        Entity.make("ganguanghao", H, "干光豪", martial=0.5, agility=0.5, alertness=0.6),
        Entity.make("geguangpei", H, "葛光佩", martial=0.4, agility=0.5, alertness=0.6),
        Entity.make("zhongling", H, "钟灵", martial=0.25, agility=0.8, alertness=0.8),
        Entity.make("sikongxuan", H, "司空玄", martial=0.7, agility=0.5, alertness=0.7),
        Entity.make("shennong", H, "神农帮帮众", martial=0.4, agility=0.5, alertness=0.6),
    ]


_DOORS = {
    "d_gate": ("shandao", "hall"), "d_corridor": ("hall", "houyuan"), "d_backgate": ("houyuan", "houshan"),
    "d_path": ("houshan", "yading"), "d_trail": ("houshan", "camp"), "d_camproad": ("camp", "shandao"),
    "d_cliff": ("yading", "jianhu"), "d_cave": ("jianhu", "shidong"), "d_stonedoor": ("shidong", "langhuan"),
    "d_tunnel": ("langhuan", "lancang"),
}


def _relations() -> list[Relation]:
    R = Relation
    rels = [R(d, Rel.CONNECTS, p) for d, ends in _DOORS.items() for p in ends]
    placement = {
        "swordrack": "hall", "yubi": "jianhu", "statue": "langhuan", "putuan": "langhuan",
        "mink": "zhongling", "antidote": "zhongling", "sword": "swordrack", "yijing": "duanyu",
        "scroll_bm": "putuan", "scroll_lb": "putuan",
        "duanyu": "hall", "mawude": "hall", "zuozimu": "hall", "xinshuangqing": "hall", "gongguangjie": "hall",
        "ganguanghao": "hall", "geguangpei": "hall", "zhongling": "hall", "sikongxuan": "camp", "shennong": "shandao",
    }
    rels += [R(e, Rel.AT, h) for e, h in placement.items()]
    rels += [R("zhongling", Rel.OWNS, "mink"), R("zhongling", Rel.OWNS, "antidote"), R("duanyu", Rel.OWNS, "yijing")]
    return rels


def _layout(*doors: str) -> tuple[Fact, ...]:
    facts = [Fact(Proposition.rel(d, Rel.CONNECTS, p)) for d in doors for p in _DOORS[d]]
    if "d_cliff" in doors:
        facts.append(Fact(Proposition.attr("d_cliff", "oneway", "jianhu")))   # 本门弟子都知道断崖爬不上来
    return tuple(facts)


def _profiles() -> dict[str, Profile]:
    sect_allies = lambda me: tuple(p for p in SECT if p != me)  # noqa: E731
    return {
        "duanyu": Profile("duanyu", "书生", "大理段氏子弟，饱读诗书，生性仁厚，最厌习武；离家出走，随马五德上山观剑",
                          is_player=True),
        "gongguangjie": Profile(
            "gongguangjie", "东宗弟子", "东宗弟子，骄横好胜；方才比剑得胜，却被一个书生当众嗤笑，恼羞成怒",
            goals=(Goal(GoalKind.HOSTILE, person="duanyu", until="wounded"),), allies=sect_allies("gongguangjie")),
        "zhongling": Profile(
            "zhongling", "少女", "万劫谷少女，天真娇憨，胆大好事；见那书生挨打，心中不平；腰间藏着一只闪电貂",
            goals=(Goal(GoalKind.DEFEND, person="duanyu"),)),
        "zuozimu": Profile("zuozimu", "东宗掌门", "无量剑东宗掌门，多疑而护短，门下弟子吃了亏必要讨回",
                           allies=sect_allies("zuozimu")),
        "xinshuangqing": Profile("xinshuangqing", "西宗掌门", "无量剑西宗掌门，冷傲寡言，与东宗面和心不和",
                                 allies=sect_allies("xinshuangqing")),
        "ganguanghao": Profile(
            "ganguanghao", "东宗弟子", "东宗弟子，与西宗葛光佩私下相好；神农帮围山，二人打算入夜后私奔下山，最怕被人撞见",
            goals=(Goal(GoalKind.ESCAPE, home="camp", not_before=ELOPE),), allies=("geguangpei",)),
        "geguangpei": Profile(
            "geguangpei", "西宗弟子", "西宗女弟子，与干光豪私订终身，入夜后便要随他逃走",
            goals=(Goal(GoalKind.ESCAPE, home="camp", not_before=ELOPE),), allies=("ganguanghao",)),
        "mawude": Profile("mawude", "宾客", "普洱老武师，做茶叶生意，为人圆滑，不愿惹事"),
        "sikongxuan": Profile("sikongxuan", "神农帮帮主", "神农帮帮主，精于药理毒物，率众围住无量山，要强占后山采药",
                              goals=(Goal(GoalKind.GUARD, home="camp"),), allies=("shennong",)),
        "shennong": Profile("shennong", "神农帮帮众", "神农帮帮众，奉命把守下山的道路，不许任何人离开",
                            goals=(Goal(GoalKind.GUARD, home="shandao"),), allies=("sikongxuan",)),
    }


def build_wuliang(seed: int = 7) -> Scenario:
    state = WorldState.build(seed, START, _entities(), _relations())

    def past(facts: tuple[Fact, ...]):
        return replace(make_percept(state, Modality.SCENE, facts=facts), tick=START - 30)

    def now_seen(agent: str):
        return replace(scene_percept(state, agent), tick=START - 1)

    palace = _layout("d_gate", "d_corridor", "d_backgate", "d_path", "d_trail", "d_cliff")
    swords = (Fact(Proposition.rel("swordrack", Rel.AT, "hall")), Fact(Proposition.rel("sword", Rel.AT, "swordrack")))
    priors: dict[str, tuple] = {a: (past(palace + swords), now_seen(a)) for a in SECT}
    priors["duanyu"] = (now_seen("duanyu"),)
    priors["mawude"] = (past(_layout("d_gate", "d_corridor")), now_seen("mawude"))
    priors["zhongling"] = (past(_layout("d_gate")), now_seen("zhongling"))
    camp = _layout("d_camproad", "d_trail", "d_gate")
    priors["sikongxuan"] = (past(camp), now_seen("sikongxuan"))
    priors["shennong"] = (past(camp), now_seen("shennong"))
    return Scenario("wuliang", state, _profiles(), priors, SETTING, LORE, ALIASES, HINTS, STYLE)
