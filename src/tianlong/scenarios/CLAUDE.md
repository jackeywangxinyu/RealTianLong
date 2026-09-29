# scenarios/
> L2 | 父级: /CLAUDE.md

内容层。场景 = 初始世界 + 角色设定 + 初始认知；初始认知以"过去的感知"给出，于是信念从第一刻起只有一个来源——感知，没有直接注入信念的后门。

成员清单
base.py: Scenario 容器（世界 + 设定 + 初始认知 + 世界前提/文风/外观描写/别称/指令示例），player/npcs 便捷属性
tianlong/: 天龙八部世界，按剧情区域逐幕构建（见 tianlong/CLAUDE.md）
procedural.py: random_scenario() 按种子生成随机小世界（链式布局 + 捷径、随机锁门与匹配钥匙、台面、藏匿、角色条件化冲突目标、熟悉布局与所在处的初始认知），供 GNN 数据与 RL 环境取样；jianghu 概率叠加江湖层（身手、半数淬毒的兵刃、解药、秘籍、单向通道、寻仇与护人），用独立随机流，jianghu=0 与旧版逐字节相同
warehouse.py: 设计验收用例"仓库钥匙"：港口—仓库入口—仓库—内仓，钥匙属船长、在桌上、匹配锁着的仓库门；警觉守卫隔一道门必然听见正常拿取
__init__.py: 包入口与 SCENARIOS 注册表（warehouse / wuliang）

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
