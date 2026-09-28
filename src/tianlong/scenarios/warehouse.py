"""
[INPUT]: 依赖 core 的实体/关系/时间/命题，kernel/perception 的 make_percept / scene_percept，scenarios/base 的 Scenario
[OUTPUT]: 对外提供 build_warehouse()：设计验收用例“仓库钥匙”
[POS]: scenarios 的标准验收场景；玩家在仓库、守卫在入口、船长在港口，钥匙在桌上且属于船长、匹配锁着的仓库门
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from dataclasses import replace

from tianlong.core import Entity, Fact, Kind, Modality, Proposition, Rel, Relation, WorldState, at
from tianlong.core.profiles import Goal, GoalKind, Profile
from tianlong.kernel.perception import make_percept, scene_percept
from tianlong.scenarios.base import Scenario

# ============================================================
#  地图
#
#   港口 ──码头栈道── 仓库入口 ──仓库大门── 仓库 ──仓库门(锁)── 内仓
#   船长               守卫                 玩家、桌面(钥匙)      账簿
# ============================================================

START = at(1, 8, 10)


def _entities() -> list[Entity]:
    P, S, It, D, H = Kind.PLACE, Kind.SURFACE, Kind.ITEM, Kind.DOOR, Kind.PERSON
    return [
        Entity.make("harbor", P, "港口"),
        Entity.make("entrance", P, "仓库入口"),
        Entity.make("warehouse", P, "仓库"),
        Entity.make("storeroom", P, "内仓"),
        Entity.make("path", D, "码头栈道"),
        Entity.make("door_main", D, "仓库大门"),
        Entity.make("door_store", D, "仓库门", locked=True),
        Entity.make("table", S, "桌面"),
        Entity.make("key", It, "钥匙", small=True),
        Entity.make("ledger", It, "账簿"),
        Entity.make("player", H, "玩家", agility=0.6, alertness=0.5),
        Entity.make("guard", H, "守卫", agility=0.5, alertness=1.0),
        Entity.make("captain", H, "船长", agility=0.4, alertness=0.6),
    ]


def _relations() -> list[Relation]:
    R = Relation
    return [
        R("path", Rel.CONNECTS, "harbor"), R("path", Rel.CONNECTS, "entrance"),
        R("door_main", Rel.CONNECTS, "entrance"), R("door_main", Rel.CONNECTS, "warehouse"),
        R("door_store", Rel.CONNECTS, "warehouse"), R("door_store", Rel.CONNECTS, "storeroom"),
        R("table", Rel.AT, "warehouse"),
        R("key", Rel.AT, "table"),
        R("ledger", Rel.AT, "storeroom"),
        R("player", Rel.AT, "warehouse"),
        R("guard", Rel.AT, "entrance"),
        R("captain", Rel.AT, "harbor"),
        R("captain", Rel.OWNS, "key"),
        R("captain", Rel.OWNS, "ledger"),
        R("key", Rel.MATCHES, "door_store"),
    ]


def _facts(*triples: tuple[str, Rel, str]) -> tuple[Fact, ...]:
    return tuple(Fact(Proposition.rel(s, r, o)) for s, r, o in triples)


def _layout() -> tuple[Fact, ...]:
    """在这里干活的人都熟悉的布局。"""
    return _facts(
        ("path", Rel.CONNECTS, "harbor"), ("path", Rel.CONNECTS, "entrance"),
        ("door_main", Rel.CONNECTS, "entrance"), ("door_main", Rel.CONNECTS, "warehouse"),
        ("door_store", Rel.CONNECTS, "warehouse"), ("door_store", Rel.CONNECTS, "storeroom"),
        ("table", Rel.AT, "warehouse"),
    )


def build_warehouse(seed: int = 7) -> Scenario:
    state = WorldState.build(seed, START, _entities(), _relations())

    def past(tick: int, facts: tuple[Fact, ...]):
        return replace(make_percept(state, Modality.SCENE, facts=facts), tick=tick)

    locked = (Fact(Proposition.attr("door_store", "locked", True)),)
    key_home = _facts(("key", Rel.AT, "table"), ("captain", Rel.OWNS, "key"), ("key", Rel.MATCHES, "door_store"))
    priors = {
        # 船长 08:00 亲眼看见钥匙在桌上，随后去了港口
        "captain": (past(at(1, 8, 0), _layout() + key_home + locked + _facts(("captain", Rel.AT, "harbor"))),),
        # 守卫清楚钥匙平时放在桌上、内仓锁着；他在入口站岗
        "guard": (past(at(1, 7, 30), _layout() + key_home + locked + _facts(("guard", Rel.AT, "entrance"))),),
        # 玩家此刻就在仓库里，看得见桌上的钥匙；但不知道仓库门锁没锁、钥匙配哪扇门
        "player": (replace(scene_percept(state, "player"), tick=START - 1),),
    }
    profiles = {
        "player": Profile("player", "player", "一个在港口打零工的外乡人", is_player=True),
        "guard": Profile(
            "guard", "guard", "尽职而谨慎的仓库守卫，宁可多查一遍，也不愿冤枉好人",
            goals=(Goal(GoalKind.PROTECT, "key", home="table"),),
        ),
        "captain": Profile(
            "captain", "captain", "精明的船长，钥匙和账簿从不离身太久",
            goals=(Goal(GoalKind.PROTECT, "key", home="table"), Goal(GoalKind.PROTECT, "ledger", home="storeroom")),
        ),
    }
    return Scenario("warehouse", state, profiles, priors)
