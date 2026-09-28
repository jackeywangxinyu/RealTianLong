# kernel/rules/
> L2 | 父级: /src/tianlong/kernel/CLAUDE.md

行动规则插件。kernel 只认识 ActionRule 接口；新增行动 = 在 core/schema 加 Op 与 OpSignature + 在这里加一个子类并注册，kernel 与 perception 不改（OCP）。规则面对真实世界状态，产出 Resolution；感知方式通过覆写 perceive()/witness_places() 组合 Witnessing 积木。

成员清单
base.py: ActionRule 抽象基类（resolve/loudness/witness_places/perceive）+ 方式系数表（careful 更轻更慢，rough 更响更快）
movement.py: MoveRule 经门移动，推不开即获知"门锁着"，通过即获知"没锁"，出发地与目的地两边都是目击者；WaitRule 不产生感知
handling.py: TakeRule/PutRule/GiveRule 只改 AT 从不改 OWNS；careful 放置即藏匿（hidden），拿起时解除
locks.py: UnlockRule/LockRule 共享前置条件（手持、门在身边、钥匙匹配），钥匙不配即获知"不配"
senses.py: InspectRule 仔细查看地点/台面发现藏匿物、搜身发现藏在身上的小物件，并给出含藏匿物的"完整看清"范围
speech.py: TellRule/AskRule 言语不改物理世界，只给听者"说法"（可为谎言）；careful 即耳语，旁人只见交谈不闻内容
__init__.py: default_rules() 注册表，启动时校验每个 Op 都有规则

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
