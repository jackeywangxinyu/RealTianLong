# language/
> L2 | 父级: /CLAUDE.md

开放语义与文字表达。LLM 只做两件事：把玩家的自由文本解析为结构化意图、把已经成立的事件按玩家可见范围写成文字。它永远不裁定事实；任何模型输出都要回到 kernel 结算。

成员清单
templates.py: 确定性文本层（无 LLM），render_fact/render_event/render_percept 角色视角措辞 + 失败原因表；memory 生成经历文本、narrator 无模型兜底共用
__init__.py: 包入口

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
