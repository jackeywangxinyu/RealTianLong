"""
[INPUT]: 依赖 core/schema 的 Op / Manner / Kind，core/changes 的 Change，core/propositions 的 Fact，core/entities 的 Scalar
[OUTPUT]: 对外提供 Intent / Outcome / Event / PerceivedEvent / Modality / EntitySketch / Percept / Observation
[POS]: core 的因果链数据：意图 → 事件（真相，含变化）→ 观察（服务端溯源记录）→ 感知（角色可见的片面内容）
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from tianlong.core.changes import Change
from tianlong.core.entities import Scalar
from tianlong.core.propositions import Fact
from tianlong.core.schema import Kind, Manner, Op

# ============================================================
#  意图：角色想做什么（结构化，尚未发生）
#  based_on 记录决策依据的世界版本——过时意图会被权威写入器拒绝
# ============================================================


@dataclass(frozen=True, slots=True)
class Intent:
    id: str
    actor: str
    op: Op
    target: str | None = None
    obj: str | None = None          # 工具或被操作的物件（开锁的钥匙、递交的物品）
    manner: Manner = Manner.NORMAL
    topic: Fact | None = None       # 语义内容：tell 的命题（可以是谎言）、ask 的问题
    based_on: int = 0
    utterance: str | None = None    # 言语的表层文字（LLM/模板渲染）；只是修辞，事实内容以 topic 为准


# ============================================================
#  事件：实际发生了什么（真相）。只存在于世界日志，不直接交给角色
# ============================================================


class Outcome(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"    # 尝试了但没成功（门锁着、东西不在）——失败本身也是事实
    REJECTED = "rejected"  # 根本没进入世界：语法非法、过时版本


@dataclass(frozen=True, slots=True)
class Event:
    id: str
    tick: int
    intent: Intent
    place: str | None
    outcome: Outcome
    reason: str | None = None
    changes: tuple[Change, ...] = ()

    @property
    def op(self) -> Op:
        return self.intent.op

    @property
    def actor(self) -> str:
        return self.intent.actor


# ============================================================
#  感知：某个角色实际获得的、片面的信息
#  - PerceivedEvent 的字段可以缺失：听到响动的人不知道是谁、做了什么
#  - Percept 不携带来源事件 ID——角色不能顺着 ID 摸到真相
#  - Observation 才携带 source_event，仅供服务端溯源与调试
# ============================================================


class Modality(StrEnum):
    SELF = "self"      # 自己行动的结果
    SIGHT = "sight"    # 亲眼看到别人的行动
    SOUND = "sound"    # 只听到响动
    SPEECH = "speech"  # 听到别人说的话（事实是“说法”，可信度取决于说话者）
    SCENE = "scene"    # 环顾四周：看到的在场事物（附带负证据的范围）


@dataclass(frozen=True, slots=True)
class PerceivedEvent:
    kind: str                       # Op 值，或 "noise"
    place: str
    actor: str | None = None
    target: str | None = None
    obj: str | None = None
    outcome: Outcome | None = None
    topic: Fact | None = None
    reason: str | None = None       # 失败原因（门锁着、没找到……）：看得见失败的人也看得见原因
    utterance: str | None = None    # 听得见的人才有：说话者的原话


@dataclass(frozen=True, slots=True)
class EntitySketch:
    """角色得以认识某实体时获得的外观：种类、名字与肉眼可见的属性。"""

    id: str
    kind: Kind
    name: str
    attrs: tuple[tuple[str, Scalar], ...] = ()


@dataclass(frozen=True, slots=True)
class Percept:
    tick: int
    modality: Modality
    event: PerceivedEvent | None = None
    facts: tuple[Fact, ...] = ()
    scopes: tuple[str, ...] = ()           # 本次完整看清其直接内容的容纳者（负证据的适用范围）
    sketches: tuple[EntitySketch, ...] = ()
    informant: str | None = None           # SPEECH 的说话者：事实只是他的说法


@dataclass(frozen=True, slots=True)
class Observation:
    id: str
    observer: str
    source_event: str | None
    percept: Percept
