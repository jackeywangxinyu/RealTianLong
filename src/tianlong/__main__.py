"""
[INPUT]: 依赖 runtime/cli 的 main
[OUTPUT]: `python -m tianlong` 入口
[POS]: tianlong 包的可执行入口，转交命令行前端
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import sys

from tianlong.runtime.cli import main

sys.exit(main())
