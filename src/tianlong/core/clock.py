"""
[INPUT]: 无外部依赖
[OUTPUT]: 对外提供 TICK_MINUTES、at()、clock_label()、is_night()、minutes_until_night()
[POS]: core 的时间语义；游戏时间是整数分钟，一个结算 tick = 1 分钟，被 kernel 推进、被 language 渲染
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

# 时间用“自第 1 日 00:00 起的分钟数”表示：整数可比较、可哈希、无时区歧义
TICK_MINUTES = 1
MINUTES_PER_DAY = 24 * 60


def at(day: int, hour: int, minute: int = 0) -> int:
    """第 day 日 hour:minute 对应的游戏时间（day 从 1 开始）。"""
    return (day - 1) * MINUTES_PER_DAY + hour * 60 + minute


NIGHT_FROM, NIGHT_UNTIL = 19, 5   # 戌时入夜，卯时破晓


def is_night(t: int) -> bool:
    hour = (t % MINUTES_PER_DAY) // 60
    return hour >= NIGHT_FROM or hour < NIGHT_UNTIL


def minutes_until_night(t: int) -> int:
    """距离下一次入夜还有几分钟；已是夜里则为 0。"""
    if is_night(t):
        return 0
    return NIGHT_FROM * 60 - (t % MINUTES_PER_DAY)


def clock_label(t: int) -> str:
    """渲染为“第1日 08:10”。"""
    day, rest = divmod(t, MINUTES_PER_DAY)
    return f"第{day + 1}日 {rest // 60:02d}:{rest % 60:02d}"
