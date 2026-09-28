"""
[INPUT]: 汇总 scenarios 各模块
[OUTPUT]: 对外提供 Scenario、build_warehouse
[POS]: scenarios 包入口；场景是纯内容层，只依赖 core 与 kernel 的感知构造器
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from tianlong.scenarios.base import Scenario
from tianlong.scenarios.warehouse import build_warehouse

__all__ = ["Scenario", "build_warehouse"]
