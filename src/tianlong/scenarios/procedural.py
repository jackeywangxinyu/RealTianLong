"""
[INPUT]: 依赖 core 的实体/关系/命题/感知类型，core/profiles 的 Goal / Profile，kernel/perception 的 make_percept / scene_percept，
         scenarios/base 的 Scenario
[OUTPUT]: 对外提供 random_scenario()（按种子生成随机小世界，含角色目标与初始认知）
[POS]: scenarios 的程序化内容；GNN 动态模型的数据与 RL 训练环境都从这里取样——训练分布覆盖布局、锁、藏匿、目标的组合，
       避免模型只记住“仓库钥匙”一个场景
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import random
from dataclasses import replace

from tianlong.core import Entity, Fact, Kind, Modality, Proposition, Rel, Relation, WorldState, at
from tianlong.core.profiles import Goal, GoalKind, Profile
from tianlong.kernel.perception import make_percept, scene_percept
from tianlong.scenarios.base import Scenario

_PLACE_NAMES = ["码头", "货栈", "账房", "酒肆", "后院", "地窖", "阁楼", "马厩"]
_ITEM_NAMES = ["铜钥匙", "账簿", "玉佩", "钱袋", "信函", "药瓶", "短刀", "印章"]
_PERSON_NAMES = ["阿福", "老周", "小翠", "王掌柜", "李镖头", "孙娘子"]


def random_scenario(seed: int, max_places: int = 5, max_items: int = 4, max_persons: int = 3) -> Scenario:
    rng = random.Random(seed)
    n_places = rng.randint(3, max_places)
    n_items = rng.randint(2, max_items)
    n_persons = rng.randint(2, max_persons)

    places = [f"p{i}" for i in range(n_places)]
    ents = [Entity.make(p, Kind.PLACE, _PLACE_NAMES[i]) for i, p in enumerate(places)]
    rels: list[Relation] = []

    # ---- 布局：一条链保证连通，再随机加一条捷径；随机锁上一扇门 ----
    edges = [(places[i], places[i + 1]) for i in range(n_places - 1)]
    if n_places > 3 and rng.random() < 0.5:
        a, b = rng.sample(range(n_places), 2)
        if abs(a - b) > 1:
            edges.append((places[a], places[b]))
    doors = []
    locked_door = rng.randrange(len(edges)) if rng.random() < 0.7 else None
    for i, (a, b) in enumerate(edges):
        d = f"d{i}"
        doors.append(d)
        ents.append(Entity.make(d, Kind.DOOR, f"{_name(ents, a)}与{_name(ents, b)}之间的门", locked=(i == locked_door)))
        rels += [Relation(d, Rel.CONNECTS, a), Relation(d, Rel.CONNECTS, b)]

    # ---- 台面 ----
    surfaces = []
    for p in places:
        if rng.random() < 0.6:
            s = f"s_{p}"
            surfaces.append(s)
            ents.append(Entity.make(s, Kind.SURFACE, f"{_name(ents, p)}的桌子"))
            rels.append(Relation(s, Rel.AT, p))

    # ---- 人：出生在锁门“外侧”的连通区域，保证开局可行动 ----
    persons = [f"h{i}" for i in range(n_persons)]
    for i, h in enumerate(persons):
        ents.append(Entity.make(h, Kind.PERSON, _PERSON_NAMES[i],
                                agility=round(rng.uniform(0.3, 0.8), 2), alertness=round(rng.uniform(0.3, 1.0), 2)))
        rels.append(Relation(h, Rel.AT, rng.choice(places)))

    # ---- 物品：第一件是匹配锁门的钥匙；其余随机放置，部分藏匿 ----
    items = [f"i{i}" for i in range(n_items)]
    for i, it in enumerate(items):
        is_key = i == 0 and locked_door is not None
        name = "铜钥匙" if is_key else _ITEM_NAMES[1 + (i % (len(_ITEM_NAMES) - 1))]
        holder = rng.choice(places + surfaces + (persons if rng.random() < 0.3 else []))
        hidden = (not is_key) and holder not in persons and rng.random() < 0.2   # 藏匿只发生在地点/台面
        ents.append(Entity.make(it, Kind.ITEM, name, small=rng.random() < 0.5, hidden=True if hidden else None))
        rels.append(Relation(it, Rel.AT, holder))
        if is_key:
            rels.append(Relation(it, Rel.MATCHES, doors[locked_door]))
        if rng.random() < 0.6:
            rels.append(Relation(rng.choice(persons), Rel.OWNS, it))

    state = WorldState.build(seed, at(1, 8, 0), ents, rels)
    return Scenario(f"proc-{seed}", state, _profiles(rng, state, persons, items), _priors(state, persons))


def _name(ents: list[Entity], eid: str) -> str:
    return next(e.name for e in ents if e.id == eid)


def _profiles(rng: random.Random, s: WorldState, persons: list[str], items: list[str]) -> dict[str, Profile]:
    """角色条件化目标：守护者、获取者、递送者各取所需，互相之间可能冲突。"""
    out = {}
    for h in persons:
        item = rng.choice(items)
        kind = rng.choice([GoalKind.PROTECT, GoalKind.ACQUIRE, GoalKind.DELIVER])
        if kind == GoalKind.PROTECT:
            goal = Goal(kind, item, home=s.target(item, Rel.AT))
        elif kind == GoalKind.DELIVER:
            goal = Goal(kind, item, recipient=rng.choice([p for p in persons if p != h]))
        else:
            goal = Goal(kind, item)
        out[h] = Profile(h, kind.value, f"{s.entity(h).name}，想要{kind.value} {s.entity(item).name}", (goal,))
    return out


def _priors(s: WorldState, persons: list[str]) -> dict[str, tuple]:
    """每个人都熟悉门的连接关系，并看得见自己所在之处；别处的物品需要亲自发现。"""
    layout = tuple(Fact(Proposition.of(r)) for r in s.sorted_relations() if r.type == Rel.CONNECTS)
    surfaces = tuple(Fact(Proposition.of(r)) for r in s.sorted_relations()
                     if r.type == Rel.AT and s.kind(r.src) == Kind.SURFACE)
    out = {}
    for h in persons:
        know = replace(make_percept(s, Modality.SCENE, facts=layout + surfaces), tick=s.clock - 1)
        out[h] = (know, replace(scene_percept(s, h), tick=s.clock - 1))
    return out
