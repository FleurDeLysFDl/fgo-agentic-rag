# FGO Agentic RAG

一个基于 *Fate/Grand Order* 游戏数据的 Agentic RAG（检索增强生成）系统——同时覆盖结构化的游戏机制事实（技能、宝具阶级、稀有度）和非结构化的中日文剧情语料（4000+ 从者资料页与剧情/任务脚本）。这个项目不是"跑通一个RAG demo"就结束了，而是刻意去挖那些只有把系统扔进真实多轮对话里才会暴露的失败模式——检索结果互相矛盾、reranker 在长文档上悄悄拖垮自己的延迟、反问循环收敛不了、需要枚举全部出现次数而不是找最相似的几个——每一个都是通过实际使用发现、从根因诊断、再修复的，详见下面的[工程日志](#工程日志实际踩过的坑与修复方式)。

## 为什么做这个

大多数 RAG 教程停在"embed文档、检索top-k、塞进prompt"这一步。这个项目往前走了一截，专门处理那些RAG系统真正投入对话使用后才会浮现的问题：检索出来的文档互相矛盾怎么办、cross-encoder reranker 在长文档上失控怎么办、反问式的自我纠错永远收敛不了怎么办、用户问"列出所有出现过的地方"这种topK检索天生答不了的问题怎么办。这些都不是纸上谈兵想出来的，是拿真实问题去跑、根因诊断、动手修复的。

## 架构

```
                      ┌─────────────────────────────────────────┐
                      │                数据管线                   │
                      │  Atlas Academy API ──▶ servants.db (SQL) │
                      │  fgo.wiki 爬取      ──▶ wiki_raw/*.json   │
                      │  Atlas 剧情脚本      ──▶ quest_raw/*.json │
                      │       │                                  │
                      │       ▼                                  │
                      │  BM25索引 + bge-m3 稠密索引 (Qdrant)      │
                      │  （长文档：先LLM摘要再embed）              │
                      └─────────────────────────────────────────┘
                                        │
                                        ▼
┌───────────────────────────── agent/graph.py（外层图）─────────────────────────────┐
│  resolve_question ──▶ decompose ──▶ solve_subquestions ──▶ synthesize            │
│  (对话记忆消解代词，   (多跳拆解 +      (逐个子问题：路由到        (把各子问题的     │
│   信息不足则反问)      query_type      子图 或 穷举关键词扫描)     回答综合成       │
│                        路由分类)                                  最终答案)       │
└─────────────────────────────────────┬─────────────────────────────────────────────┘
                                       │  query_type="standard"
                                       ▼
                    ┌───────────── agent/subgraph.py（Self-RAG子图）─────────────┐
                    │  route_question ──▶ structured_lookup / retrieve          │
                    │       │                       │                           │
                    │       ▼                       ▼                           │
                    │  servants.db             BM25 + bge-m3 + RRF融合          │
                    │  （精确事实）              + bge-reranker-v2-m3            │
                    │                           + 分数断层截取                   │
                    │                                │                           │
                    │                                ▼                           │
                    │                          check_conflict ──▶ generate      │
                    │                          （仅structured        │          │
                    │                            路由生效）           ▼          │
                    │                                          grade_generation │
                    │                                     （幻觉检测+答案       │
                    │                                      质量检测，          │
                    │                                      有限次重试）        │
                    └─────────────────────────────────────────────────────────┘
```

**两条检索路径**，按子问题动态选择：
- **结构化路径**（`servants.db`，SQLite）——精确的游戏机制事实：技能效果、宝具阶级/卡色、稀有度、获取途径。这是查库不是生成，没有幻觉风险。
- **向量检索路径**——BM25（关键词）+ bge-m3（稠密向量）→ 倒数排序融合（RRF）→ bge-reranker-v2-m3（cross-encoder重排）→ 一个动态截断：保留候选直到排序列表里分数下降最陡的那个断点为止，而不是写死的 top-k（详见工程日志）。

**自我纠错**（Self-RAG风格，用结构化LLM判断代替微调过的reflection token）：
- `check_conflict` —— 生成答案之前，先判断检索到的文档在这个问题上是否真的互相矛盾（比如两个从者变体宝具阶级不同），矛盾就反问用户消歧，而不是把两种说法混在一起给出模糊答案。
- `grade_generation` —— 检查生成的答案有没有幻觉（无资料支撑的论述）和是否切题，按需重试生成或重新检索（重试次数有上限）。

**对话记忆 + 反问机制**：`resolve_question` 结合对话记忆把"她的宝具是什么"这类依赖上下文的追问改写成自包含问题；记忆不足以消解时才反问，且反问轮次有上限，超过就强制给出最合理解读而不是无限循环。`agent/memory.py` 的 `ConversationMemory` 把对话持久化到 SQLite（重启不丢失），并且是**有界**的——只有最近几轮原文会被送进prompt，更早的内容会被压缩成一份滚动更新的摘要，而不是让prompt随对话变长无限膨胀。

## 技术栈

| 层次 | 选型 |
|---|---|
| 编排 | LangGraph（两层嵌套的 `StateGraph`：外层多跳规划，内层 Self-RAG 求解） |
| LLM | gpt-4o-mini，通过 OpenAI 兼容接口（`langchain-openai`） |
| 稠密向量 | `BAAI/bge-m3` |
| 重排模型 | `BAAI/bge-reranker-v2-m3`（cross-encoder） |
| 关键词检索 | `rank_bm25` + `jieba`（中文分词） |
| 向量库 | Qdrant，本地嵌入式模式（无需起服务） |
| 结构化存储 | SQLite（从者数据 + 对话记忆） |
| 前端 | Streamlit（多轮对话） |
| API | FastAPI（按 session_id 区分独立、持久化的对话） |
| 评测 | 自建 Recall@5 评测 + RAGAS（忠实度/相关性/精确率/召回率） |

## 快速开始

```bash
pip install -e ".[retrieval,agent,eval,dev,api,ui]"
cp .env.example .env   # 至少要填 LLM_API_KEY
```

构建语料和索引（可续跑，重复执行会跳过已缓存的步骤）：

```bash
python scripts/update_all.py          # Atlas API -> servants.db，fgo.wiki 爬取 -> wiki_raw/
python scripts/fetch_quest_scripts.py # 剧情/任务脚本文本 -> quest_raw/
python scripts/summarize_corpus.py    # 给长记录生成LLM摘要（用于embedding/rerank）
python scripts/build_bm25_index.py
python scripts/build_vector_index.py
```

运行：

```bash
streamlit run app.py                  # 聊天界面
uvicorn api:app --reload              # REST API（POST /chat，按session_id区分对话）
python -m agent.graph "阿尔托莉雅和贞德的宝具阶级哪个更高？"   # 命令行，单次查询
```

## 评测结果

25道手写评测题（[`eval/questions.json`](eval/questions.json)），涵盖单从者事实、跨从者对比、以及国服还未实装的JP限定剧情。完整的前后对比数据见 [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)。

| 指标 | 结果 |
|---|---|
| Recall@5（检索命中率） | **96%**（24/25） |
| 多跳查询平均LLM调用次数 | 25次（去掉逐文档打分前是54次） |

## 工程日志：实际踩过的坑与修复方式

以下都是拿真实query跑出来发现的，不是靠代码审查看出来的——每一条都有具体的前后对比数据。完整细节见 [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)。

- **Reranker延迟失控**：`max_length` 没设上限，导致cross-encoder遇到长剧情文本时会把候选文本编码到8192 token（是512 token正常情况的256倍算力），一次查询跑了20多分钟。修复：显式设 `max_length=512`，并且长文档改用其LLM生成的摘要来做重排（而不是截断的原文片段），把因截断损失的召回率补了回来。
- **逐文档打分是延迟的主要来源**：每个检索到的候选都要单独调一次LLM判断相关性（一次歧义查询能触发10+次调用）——整个删掉，换成一个不调LLM、免费的"分数断层截取"逻辑。
- **静默的top-5截断**：`retrieve()` 只请求reranker前5个结果，导致"断层截取"逻辑根本看不到、也救不回排第6名以后的真实命中——实测一个从者自己的资料页排第6（分数0.895，跟第5名分差只有0.012，压根不是断层），却被硬性的top-5卡在外面。现在改成请求reranker的全部候选池（reranker反正已经把它们全打过分了，不花额外算力），让断层截取逻辑自己判断该留几个。
- **无限反问循环**：开放式的叙事类问题没有天然的"足够具体"停止点，反问判断逻辑可能无限收窄下去（实测跑出过8轮，从"感情"问到"谁对谁的感情"都没给出答案）。修复：冲突检测只对structured路由（数值类事实）生效，并且给反问轮次设上限，超过就强制给出最合理解读而不是继续问。
- **Top-k检索答不了"列出所有X"**：语义/关键词top-k检索只能返回跟当前措辞最匹配的那一小撮文档——同一个"列出伊阿宋出场过的所有剧情"用两种不同问法，分别只召回6条和2条、且完全不重叠，而穷举扫描找到的真实出现次数是130条。修复：在问题拆解阶段给每个子问题标注 `query_type`，枚举类问题直接走穷举关键词扫描，不走相似度检索。
- **`route` 字段过期导致误判**：structured查库没命中、回退到向量检索重试时，`state["route"]` 没有跟着更新，还残留着旧值"structured"，导致 `check_conflict`（本该只对structured路由生效）错误地对向量检索出来的叙事文档做判断——实测出现过一次Lancer职阶技能问题被混进了Archer/Ruler/Alter等其他形态数据的情况。修复：让fallback到向量检索的分支显式把route更新为"vectorstore"。

## 已知局限

- 枚举类问题用的是精确子串扫描，不是语义匹配——只用代词指代、没提到实体原名的地方不会被统计进去。
- 相关性截取（`MIN_KEPT_DOCUMENTS` 下限，无上限）在分数曲线是渐进下降而非陡崖式下降时，偶尔会保留超出预期数量的候选；目前没有加上限。
- 还没有自动化测试套件。

## 项目结构

```
agent/              LangGraph外层图 + Self-RAG子图，LLM客户端，对话记忆，schema定义
retrieval.py        混合检索器（BM25 + 稠密向量 + RRF + 重排）
scripts/            数据管线（抓取、爬取、建索引）+ 评测脚本
app.py               Streamlit聊天前端
api.py                FastAPI接口（按session_id区分持久化对话）
eval/                手写评测题集 + 召回率结果
docs/BENCHMARKS.md  调试过程中的详细前后对比数据
```
