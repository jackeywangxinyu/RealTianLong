"""
[INPUT]: 汇总 scenarios/tianlong 各幕
[OUTPUT]: 对外提供 build_wuliang
[POS]: scenarios/tianlong 包入口：天龙八部世界，按剧情区域逐幕构建（第一幕：无量山）
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from tianlong.scenarios.tianlong.wuliang import build_wuliang

__all__ = ["build_wuliang"]
