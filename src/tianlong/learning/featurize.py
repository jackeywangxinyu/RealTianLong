"""
[INPUT]: 依赖 numpy，cognition 的 GraphView / VIEW_ATTRS / EVENT_KIND，cognition 的 Candidate，core 的 Op / Manner / Modality / Kind / Rel / digest
[OUTPUT]: 对外提供特征词表常量（NODE_KINDS / REL_VOCAB / F_NODE / F_EDGE / N_OPS / N_MANNERS）、词表指纹 VOCAB 与 check_vocab() / StaleModel、
         GraphTensors、featurize()、ActionCode、encode_action()
[POS]: learning 的输入边界：只接受 GraphView——角色入口的张量在构造上就拿不到世界真相。
       关系类型（含极性与方向）编码进边特征，与可信度/时效/传闻一起交给支持 edge_dim 的卷积；
       numpy 是中立格式：动态模型转成 PyG Data，RL 环境把它填充成定长观测
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tianlong.cognition import EVENT_KIND, VIEW_ATTRS, Candidate, GraphView
from tianlong.core import Kind, Manner, Modality, Op, Rel, digest

# ============================================================
#  词表
# ============================================================

NODE_KINDS = (*(k.value for k in Kind), EVENT_KIND)
EVENT_KINDS = (*(o.value for o in Op), "noise")
MODALITIES = tuple(m.value for m in Modality)
_WORLD_RELS = tuple(r.value for r in Rel)
_EVENT_RELS = ("OCCURRED_AT", "BY", "ON", "WITH")

# 关系词表：(关系, 否定, 反向)。世界关系有正负极性；事件关系只有正。每条边都配一条反向边，让信息双向流动
REL_VOCAB: tuple[tuple[str, bool, bool], ...] = tuple(
    (r, neg, rev)
    for r in (*_WORLD_RELS, *_EVENT_RELS)
    for neg in ((False, True) if r in _WORLD_RELS else (False,))
    for rev in (False, True)
)
_REL_INDEX = {k: i for i, k in enumerate(REL_VOCAB)}

F_NODE = len(NODE_KINDS) + len(VIEW_ATTRS) + 1 + len(EVENT_KINDS) + len(MODALITIES)
F_EDGE = len(REL_VOCAB) + 3          # + 可信度、时效、传闻
AGE_SCALE = 120.0                    # 两小时以上的时效视为同等陈旧
N_OPS = len(Op)
N_MANNERS = len(Manner)
OP_INDEX = {o: i for i, o in enumerate(Op)}
MANNER_INDEX = {m: i for i, m in enumerate(Manner)}
HOLDER_KINDS = (Kind.PLACE.value, Kind.SURFACE.value, Kind.PERSON.value)
LOCATED_KINDS = (Kind.PERSON.value, Kind.ITEM.value, Kind.SURFACE.value)

# 词表指纹：检查点记下训练时的词表。新增行动或属性后旧模型的张量形状就对不上——
# 与其在 load_state_dict 里报一串维度错误，不如在加载时直说“请重训”
VOCAB = digest(NODE_KINDS, EVENT_KINDS, MODALITIES, REL_VOCAB, VIEW_ATTRS, tuple(o.value for o in Op),
               tuple(m.value for m in Manner))


class StaleModel(ValueError):
    """检查点的特征词表与当前代码不一致。"""


def check_vocab(ckpt: dict, path: object) -> None:
    if ckpt.get("vocab") != VOCAB:
        raise StaleModel(f"{path} 是用另一套特征词表训练的（行动或属性有增减），请按 README“训练与结果”重训")


@dataclass(frozen=True)
class GraphTensors:
    node_ids: tuple[str, ...]
    kinds: tuple[str, ...]
    x: np.ndarray            # float32 [N, F_NODE]
    edge_index: np.ndarray   # int64   [2, E]
    edge_attr: np.ndarray    # float32 [E, F_EDGE]

    def index_of(self, eid: str | None) -> int:
        if eid is None:
            return -1
        try:
            return self.node_ids.index(eid)
        except ValueError:
            return -1

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)


def featurize(view: GraphView) -> GraphTensors:
    ids = view.node_ids()
    index = {eid: i for i, eid in enumerate(ids)}
    x = np.zeros((len(ids), F_NODE), dtype=np.float32)
    k0, a0 = len(NODE_KINDS), len(NODE_KINDS) + len(VIEW_ATTRS)
    e0 = a0 + 1
    m0 = e0 + len(EVENT_KINDS)
    for i, n in enumerate(view.nodes):
        x[i, NODE_KINDS.index(n.kind)] = 1.0
        attrs = dict(n.attrs)
        for j, a in enumerate(VIEW_ATTRS):
            x[i, k0 + j] = attrs.get(a, 0.0)
        x[i, a0] = 1.0 if n.is_self else 0.0
        for key, val in n.attrs:
            if key.startswith("op:") and key[3:] in EVENT_KINDS:
                x[i, e0 + EVENT_KINDS.index(key[3:])] = val
            elif key.startswith("modality:") and key[9:] in MODALITIES:
                x[i, m0 + MODALITIES.index(key[9:])] = val

    src, dst, attr = [], [], []
    for e in view.edges:
        if e.src not in index or e.dst not in index:
            continue
        for rev in (False, True):
            feat = np.zeros(F_EDGE, dtype=np.float32)
            feat[_REL_INDEX[(e.rel, not e.holds, rev)]] = 1.0
            feat[-3] = e.confidence
            feat[-2] = min(e.age, AGE_SCALE) / AGE_SCALE
            feat[-1] = 1.0 if e.hearsay else 0.0
            a, b = (index[e.dst], index[e.src]) if rev else (index[e.src], index[e.dst])
            src.append(a)
            dst.append(b)
            attr.append(feat)
    edge_index = np.array([src, dst], dtype=np.int64) if src else np.zeros((2, 0), dtype=np.int64)
    edge_attr = np.stack(attr) if attr else np.zeros((0, F_EDGE), dtype=np.float32)
    return GraphTensors(ids, tuple(n.kind for n in view.nodes), x, edge_index, edge_attr)


# ============================================================
#  行动编码：操作 + 方式 + 目标/对象/行动者 在图中的位置（-1 表示无）
#  言语的命题内容不进入动态模型：它不改变物理世界，只改变听者的说法
# ============================================================


@dataclass(frozen=True, slots=True)
class ActionCode:
    op: int
    manner: int
    target: int
    obj: int
    actor: int


def encode_action(g: GraphTensors, actor: str, cand: Candidate) -> ActionCode:
    return ActionCode(OP_INDEX[cand.op], MANNER_INDEX[cand.manner], g.index_of(cand.target), g.index_of(cand.obj),
                      g.index_of(actor))
