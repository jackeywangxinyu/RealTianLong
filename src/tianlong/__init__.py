"""
[INPUT]: 无
[OUTPUT]: 对外提供包版本号 __version__
[POS]: tianlong 包根；各子包依赖方向为 core ← kernel/cognition ← persistence/memory ← agents/learning/language ← runtime
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

__version__ = "0.1.0"
