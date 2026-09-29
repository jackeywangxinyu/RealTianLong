# kernel/
> L2 | 父级: /CLAUDE.md

世界规则内核。整个系统唯一允许"知道真相"并裁定事实的地方。Kernel.step() 是纯函数：同一版本 + 同一批意图 → 同一新版本 + 同一批事件与观察；无 IO、无全局随机（随机数由 world seed + tick + 事件派生）。GNN、LLM、多智能体都不能绕过它改写世界。

结算模型：同一版本观察 → 并行意图 → 准入（语法/版本/每人每 tick 一个）→ 按先手度排序逐个生效（冲突即由顺序裁定：后手者面对的已是被改变的世界）→ 不变量闸门 → 推进时钟 → 所有人环顾。

成员清单
kernel.py: Kernel/StepResult，结算主流程；先手度 = 规则基线 + 方式加成 + 敏捷 + 确定性抖动，与提交顺序无关；穴道被制者只能做规则声明 usable_when_subdued 的事
space.py: 空间与身体物理只读查询，AT 链上溯地点、门与邻接（不替角色挑路：没有 door_between）（暗门不入环顾、单向通道只能往一头走）、声音跳数、可见性（藏匿物需仔细查看，小物件被携带时藏在身上）、昼夜、身手（内力 + 最锋利兵刃，伤减、毒半）、点穴时限
perception.py: 感知物理，观察按行动实际留下的 Fragment（地点、公开视图、事实、响度）投影而非按行动目标；Witnessing 提供"行动者自知/片段目击/隔壁听声"积木，public_view() 默认不含内部失败原因与暗门 ID，SILENT_FAILURES 未触及世界的尝试不留痕迹；make_percept(vantage) 只给亲眼所见的实体外观，言语提到的、隔墙听见的、门那头的只闻其名（seen=False）；scene_percept() 每 tick 给出所见、在场者身体状态与负证据范围（暗门除外）；布尔属性统一规范为 (attr=True, 极性)；私密属性永不成为事实
resolution.py: Resolution 裁定值对象（结果/原因/变化/行动者获知/完整看清范围）
invariants.py: 不变量闸门，唯一位置、种类约束、门连两地、物品至多一主；违规即规则 bug，提交前爆炸
rules/: 每种行动一条规则的插件目录（见 rules/CLAUDE.md）
__init__.py: 包入口

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
