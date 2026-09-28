# RealTianLong

图世界文字游戏引擎：**规则内核裁定事实，每个角色经由自己的认知图理解世界**；GNN 预测行动后果，强化学习训练角色如何选择，LangGraph 编排多个角色的决策流程，向量库提供可检索的经历，LLM 只负责开放语义与文字表达。

> 预测不是事实，回忆不是现状，相信不等于真实，多个 agent 达成一致也不等于事情已经发生。
> 这四条边界在本仓库里被实现成了数据结构、接口与测试。

## 架构

```
                 ┌──────────────────────────── runtime/session：一回合 ────────────────────────────┐
玩家文本 ──▶ language/parser（规则优先，Gemini 兜底，只能引用玩家认识的实体）──▶ 结构化意图 ─┐
                                                                                              │
各 NPC：agents/npc_graph（LangGraph）                                                          │
  observe → recall(Qdrant) → propose(候选) → predict(GNN) → decide(RL/脚本) → express → 意图 ─┤
          ▲ 只经由 AgentPort 读自己的认知与回忆                                                 │
          │                                                                                   ▼
cognition/BeliefStore ◀── 感知折叠 ◀── runtime/WorldAuthority（唯一写入者，幂等 + 版本锁）◀── kernel.step（纯函数裁定）
          │                                   │
          │                                   ├─▶ persistence（内存 / Neo4j）：世界、事件、观察、命题与信念、outbox
          │                                   └─▶ memory/indexer ─▶ Qdrant（派生数据，可重建）
          ▼
language/narrator：只把玩家本回合的感知写成文字
```

| 组件 | 选型 | 职责 | 刻意不承担 |
|---|---|---|---|
| 世界规则内核 | 纯 Python（`kernel/`） | 前置条件、冲突结算（先手度排序）、时间、效果、感知、不变量 | —— 唯一裁定事实处 |
| 图数据库 | Neo4j（`persistence/neo4j_store.py`） | 实体/关系/事件/观察；命题节点与 `BELIEVES` 边分离“内容”与“谁相信” | 判断角色该做什么 |
| 图神经网络 | PyTorch Geometric（`learning/`） | 行动条件化的动态预测：成败、位置变化（含“未知”）、属性 | 未经内核校验改写世界 |
| 多智能体框架 | LangGraph（`agents/`） | 单角色决策图 + `Send` 扇出并行编排 | 用“讨论结果”替代世界规则 |
| 向量数据库 | Qdrant（`memory/`） | 按语义检索本人有权回忆的经历（服务端强制过滤） | 裁定物品当前位置 |
| 强化学习 | RLlib（`learning/rl/`） | 模仿学习初始化 + PPO，所有角色共享参数、各自观测 | 游玩中临时重训 |
| LLM | Gemini（`language/`） | 输入解析、对白润色、叙述润色；失败即回退模板 | 裁定事实、越过认知边界 |

## 快速开始

```bash
uv venv .venv && . .venv/bin/activate
uv pip install -e ".[all,dev]"

# 离线游玩（模板叙述、脚本 NPC）
python -m tianlong --llm none --debug

# Gemini 叙述与对白（密钥只从环境变量读取，调用结果缓存在 .cache/llm）
export GEMINI_API_KEY=...        # 见 .env.example
python -m tianlong

# 用训练好的 GNN 预测器 / RL 策略驱动 NPC（先运行下方训练命令）
python -m tianlong --predictor gnn --policy learned

# Neo4j 持久化与跨进程存档（无 Docker 环境可用脚本起本地 Neo4j）
NEO4J_PASSWORD=******** scripts/neo4j_local.sh
export NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=********
python -m tianlong --store neo4j --save 我的存档
```

游戏内：`/beliefs` 查看你自己的认知，`/debug` 切换开发者视角（真相与 NPC 决策理由）。

## 设计验收用例：仓库钥匙

玩家在仓库，守卫在入口，船长在港口；钥匙在桌上、属于船长、匹配锁着的仓库门。

1. 玩家拿走钥匙 → 钥匙位置变化，**所有权不变，门仍锁着**。
2. 守卫隔一道门**只听到响动**，不知道是谁、做了什么；船长毫无感知，仍以为钥匙在桌上。
3. 守卫决定进去查看 → 环顾时发现**钥匙不在桌上**，但钥匙藏在玩家身上，**不知道谁拿走了它**。
4. 盘问无果后搜身，才知道钥匙在玩家身上 → 去港口报告船长 → 船长打听玩家下落 → 当面质问。

整条链路由 `tests/test_acceptance_warehouse.py` 与 `tests/test_agents.py` 逐段断言；NPC 的每一步都由各自的认知驱动自然涌现，没有剧本。

## 训练与结果

```bash
python -m tianlong.learning.train --view env   --worlds 500 --epochs 25   # 环境动态（全知）
python -m tianlong.learning.train --view agent --worlds 500 --epochs 25   # 角色视角（只看认知）
python -m tianlong.learning.rl.train --ppo-iterations 25                   # 模仿学习 + PPO + 评测
```

动态模型在**未见过的程序化世界**（按世界切分）上的结果，对照“什么都不变”基线：

| 指标 | 环境模型 | 角色模型 | 基线 |
|---|---|---|---|
| 变化的事实被正确预测（召回） | 100% | 92.0%（精确率 93.0%） | 0% |
| 未变化的事实保持不变 | 100% | 99.6% | 100% |
| 行动成败准确率 | 100% | 96.7% | 87.0% |
| 未知被错误确定化 | — | 2.0% | — |

角色模型低于环境模型是**正确的**：角色不知道门锁没锁、东西藏在哪，预测理应保留不确定性。

角色策略在 40 局**留出种子**的程序化世界上（每局 30 分钟游戏时间，2~3 个角色目标互相冲突）：

| 策略 | 平均回报 | 目标达成率 | 每局冤枉人次数 |
|---|---|---|---|
| 随机 | −0.768 | 22% | 9.9 |
| 脚本示范者 | **−0.234** | **46%** | 0.05 |
| 模仿学习 | −0.300 | 37% | 0 |
| 模仿学习 → PPO（25 轮 × 1500 步） | −0.283 | 38% | 0 |
| 同上，去掉世界模型预测特征 | −0.283 | 38% | 0 |

如实结论：RLlib 流水线端到端可用，“冤枉人”惩罚有效（随机策略每局 9.9 次 → 学得的策略 0 次）；但在这个算力预算下 PPO 只略优于模仿学习、**尚未超过脚本示范者**，且消融显示策略目前**没有利用**世界模型的预测特征。脚本策略的已知盲区是“不知道目标物品在哪时原地等待”，这正是 RL 应当学会的探索——模仿后的策略过于尖锐（示范准确率 99.5%）是首要嫌疑，见 `--entropy` 与 `--bc-smoothing`。

## 测试

```bash
pytest                                  # 默认套件（Neo4j 不可达则跳过相关用例）
NEO4J_URI=... NEO4J_PASSWORD=... pytest # 含内存/Neo4j 双后端契约测试
pytest -m slow                          # PPO 冒烟
```

| 边界 | 实现 | 可证伪断言 |
|---|---|---|
| 相信 ≠ 真实 | `cognition/beliefs.py` 槽位修正、传闻并存、负证据 | `test_cognition.py` |
| 没收到消息的人不能提前知道 | `AgentPort`、`belief_view`、`Percept` 不带溯源 ID | `test_isolation.py`、`test_agents.py`、`test_learning.py`（扰动真相，认知与张量逐字节不变） |
| 回忆 ≠ 现状 | `MemoryScope` 服务端强制过滤；工作记忆弥补索引延迟 | `test_memory.py` |
| 同意 ≠ 发生 | 智能体只产出意图，`WorldAuthority` 唯一提交 | `test_rl.py`、`test_agents.py` |
| 相同存档 + 行动 → 相同结果 | blake2b 派生 ID 与种子，纯函数内核 | `test_acceptance_warehouse.py`、`test_store_contract.py`（跨后端指纹一致） |
| 重试不二次结算 | 意图 ID 由 (世界, 分支, 角色, 版本) 派生 | `test_acceptance_warehouse.py`、`test_agents.py` |

项目地图见 [`CLAUDE.md`](CLAUDE.md)，每个模块目录下都有自己的 `CLAUDE.md`。
