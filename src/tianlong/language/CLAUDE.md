# language/
> L2 | 父级: /CLAUDE.md

开放语义与文字表达。LLM 只做两件事：把玩家的自由文本解析为结构化意图、把已经成立的事件按玩家可见范围写成文字。它永远不裁定事实；任何模型输出都要回到 kernel 结算。

成员清单
templates.py: 确定性文本层（无 LLM），render_fact/render_event/render_experience/render_percept 角色视角措辞（名称表即观察者的实体草图，按种类说"在桌上/在身上"）+ 失败原因/研读进境（未贯通者明说未贯通，免得润色时被夸成学成）/武侠状态措辞 + 自己移动只写“来到”（步态留给原话） + 事件后果（伤毒被制、学成、暗道与藏匿之物）；memory、speaker、narrator 共用
llm.py: LLMClient 协议 + GeminiClient（REST，密钥只从环境变量读）+ CachedLLM 磁盘缓存 + llm_from_env()；任何失败抛 LLMUnavailable，调用方必须回退模板
speaker.py: Speaker 协议 + TemplateSpeaker + LLMSpeaker，把结构化言语行动说成符合人设的一句话；事实以命题为准，原话只是修辞
parser.py: IntentParser 规则优先（按优先级尝试所有命中关键词的操作，含出手/研读/服下/跳崖/磕头）、规则失败才调 LLM（JSON Schema 约束、角色字段 required+nullable）；可引用实体只来自玩家认知图（场景别称只对已认识的实体生效），normalize() 把“朝门走”换算为门那边的地点；等待可带时长或“等到天黑”
narrator.py: Narrator 输入只有玩家本回合的感知，模板先写成事实清单（自称改为“你”，移动/查看后附所见，去重），LLM 只润色且不得添加清单外内容（程度照原样、发现不等于到手），失败即输出清单；玩家原话只作为意图与姿态交给 LLM（“跳下断崖”读来像跳，成败仍以清单为准），等待后的时辰排在所见之前；lore_keys() 只挑真正映入眼帘的人与物（门那头的地点不算，夜里取月下变体），外观描写只在初见时给出；世界前提与文风来自场景
__init__.py: 包入口

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
