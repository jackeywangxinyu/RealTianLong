# cognition/
> L2 | 父级: /CLAUDE.md

角色心智。只依赖 core，绝不依赖 kernel——角色的心智里没有世界规则的真相。每个角色一份 BeliefStore，只由感知折叠而成：可以过时、可以错、可以自相矛盾。这里把"预测不是事实、回忆不是现状、相信不等于真实"实现为数据结构与修正规则。

修正规则：函数型槽位上新证据取代（衰减后）可信度不高于它的旧值，更可信的旧值保留（矛盾说法由此并存）；亲见=1.0、传闻=对说话者的信任；看清某容纳者却没看到原以为在那里的东西 → 负证据；随意环顾不否定"认为被藏起来"的东西，仔细查看才能。

成员清单
beliefs.py: Belief/Episode/BeliefChange/BeliefStore（entities/beliefs/trust 三张映射为 FrozenMap，认知只能经 revise() 形成新的一份），revise() 修正规则（亲见的草图才更新外观，只闻其名不抹掉已见外观），positives() 允许同槽多值并存，effective_confidence() 让易变事实（位置/锁）按 60 分钟半衰期变旧，信任度来自角色设定，episodes 保留近期经历作为 GNN 事件节点
navigation.py: believed_place() 沿认为的 AT 链找地点，routes_between() 认为连通两地的门（认为没锁的优先、确知单向反向的不算），route_to()/next_hop() 沿认为存在的门 BFS 给出下一站与所走的门——地图错了就会走错，不知道的暗门不在地图上
view.py: GraphView 统一图投影，world_view() 全知入口（仅环境动态学习）与 belief_view() 角色入口；节点带外观（只闻其名者外观未知而非否）与身体状态三态、边带可信度/时效/极性/传闻标记；learning 只接受 GraphView，隔离由类型边界保证
candidates.py: Candidate 结构化候选行动 + candidates()，MOVE 候选绑定认为存在的路线（认为锁着也照样可以去推），候选对象只来自认知图（按"以为"剪枝合理，按真相剪枝即泄密），确知钥匙不配/通道单向才剪掉；含动手、研读、施用、悄悄移动、搜走被制者之物；自知被制只剩开口与等待；截断时身体行动优先于组合爆炸的言语；WAIT 永居首位
__init__.py: 包入口

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
