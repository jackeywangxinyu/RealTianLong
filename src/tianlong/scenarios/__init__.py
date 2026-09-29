"""
[INPUT]: 汇总 scenarios 各模块
[OUTPUT]: 对外提供 Scenario、Ending、build_warehouse、build_wuliang、SCENARIOS 注册表
[POS]: scenarios 包入口；场景是纯内容层，只依赖 core 与 kernel 的感知构造器
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from collections.abc import Callable

from tianlong.scenarios.base import Ending, Scenario
from tianlong.scenarios.tianlong import build_wuliang
from tianlong.scenarios.warehouse import build_warehouse

# 场景注册表：命令行 --world 按名取用
SCENARIOS: dict[str, Callable[[int], Scenario]] = {"warehouse": build_warehouse, "wuliang": build_wuliang}

__all__ = ["SCENARIOS", "Ending", "Scenario", "build_warehouse", "build_wuliang"]
