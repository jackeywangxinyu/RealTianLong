"""
[INPUT]: 汇总 kernel 各模块
[OUTPUT]: 对外提供 Kernel / StepResult / KERNEL_VERSION / ActionRule / default_rules / InvariantViolation / violations
[POS]: kernel 包入口；上层只经由此处使用世界规则内核
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from tianlong.kernel.invariants import InvariantViolation, violations
from tianlong.kernel.kernel import KERNEL_VERSION, Kernel, StepResult
from tianlong.kernel.rules import ActionRule, default_rules

__all__ = ["KERNEL_VERSION", "ActionRule", "InvariantViolation", "Kernel", "StepResult", "default_rules", "violations"]
