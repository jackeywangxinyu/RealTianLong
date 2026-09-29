"""
[INPUT]: 依赖 core/memories 的 MemoryRecord
[OUTPUT]: 对外提供 MemoryView（长期记忆的结构化摘要：各实体被经历提到几次、各人的说法被亲眼证实/证伪几次）、MEMORY_FIELDS、
          reliability()（说法可靠度的拉普拉斯估计）
[POS]: memory 的策略入口：长期记忆以结构化摘要进入决策（脚本策略按说话者可靠度取舍矛盾的说法，学得的策略把它作为节点特征），
       而不是一段只有 LLM 读得懂的文字。它由角色自己的权威经历记录汇总而成——可重建、按时间过滤，从不改写现实或当前信念；
       训练环境与线上用同一个 records_for → from_records 定义，训练时看到什么，上线时就看到什么
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from tianlong.core.memories import MemoryRecord

MEMORY_FIELDS = ("mentions", "confirmed", "refuted")   # 进入策略观测的逐节点记忆特征，顺序即列序


@dataclass(frozen=True)
class MemoryView:
    mentions: Mapping[str, int] = field(default_factory=dict)    # 经历里提到某实体的次数
    confirmed: Mapping[str, int] = field(default_factory=dict)   # 某人的说法后来被自己亲眼证实的次数
    refuted: Mapping[str, int] = field(default_factory=dict)     # 某人的说法后来被自己亲眼证伪的次数
    records: int = 0

    @staticmethod
    def from_records(records: Iterable[MemoryRecord], now: int | None = None) -> MemoryView:
        return MemoryView().add(records, now)

    def add(self, records: Iterable[MemoryRecord], now: int | None = None) -> MemoryView:
        """并入新的经历记录（now 给出时只并入 known_at <= now 的：回忆不能来自未来）。"""
        m, c, r = Counter(self.mentions), Counter(self.confirmed), Counter(self.refuted)
        n = self.records
        for rec in records:
            if now is not None and rec.known_at > now:
                continue
            n += 1
            m.update(set(rec.subjects))
            if rec.informant and rec.verdict == "confirmed":
                c[rec.informant] += 1
            elif rec.informant and rec.verdict == "refuted":
                r[rec.informant] += 1
        return MemoryView(dict(m), dict(c), dict(r), n)

    def reliability(self, informant: str | None) -> float:
        """说话者可靠度：(证实 + 1) / (证实 + 证伪 + 2)；没有交道就是 0.5，自己亲眼所见不走这里。"""
        if informant is None:
            return 1.0
        c, r = self.confirmed.get(informant, 0), self.refuted.get(informant, 0)
        return (c + 1) / (c + r + 2)

    def features(self, eid: str) -> tuple[float, float, float]:
        return (float(self.mentions.get(eid, 0)), float(self.confirmed.get(eid, 0)), float(self.refuted.get(eid, 0)))
