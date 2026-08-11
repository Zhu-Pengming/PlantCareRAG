# Retrieval evaluation v0.2

这一阶段的目的不是证明系统已经很好，而是用真实问法暴露知识库和 schema 的缺口。

## 数据集

- `data/main_queries.json`：40 条真实用户提问，覆盖现有 3 种植物。
- `data/abstain_queries.json`：10 条应拒答/降级的问题，包括库外植物和库内植物的缺失维度。
- `schemas/query.schema.json`：主集条目格式。
- `schemas/abstain_query.schema.json`：拒答集条目格式。

社区页面只提供 query 原文，不作为答案或 gold knowledge。`gold_entry_ids` 只能指向 `knowledge_base/data/entries/` 中有权威来源的条目。

### 语料边界

当前 query 评测集为英文语料，中文/跨语言用户输入不在当前 scope 内。40 条主集全部来自可回溯的英文 Reddit 帖子；结果只能解释英文新手问法上的表现，不代表中文用户性能。知识条目的事实 `content` 为中文，检索 metadata 含中英文，因此 rerank 报告会单独披露文档表示的语言错配。

## 主集字段

- `raw_text`：原样保留的帖子标题或正文提问，不修正语法、大小写和口语。
- `plant_id`：标注者根据帖子上下文确定的植物；无法唯一确定时为 `null`。
- `alias_used`：用户实际使用的实体名称；无显式实体时为 `null`。
- `expected_dimensions`：回答这个问题理论上需要检索的维度。
- `gold_entry_ids`：当前 KB 中应该召回的条目。可以为空。
- `status`：
  - `covered`：当前 KB 足以支持核心回答；
  - `partial`：只覆盖部分条件或原因；
  - `gap`：当前 KB 没有可支持核心回答的条目。
- `answer_shape`：用户期待的答案形式，用于区分“召回正确但答非所问”。
- `answer_shape_support`：只在完成答案形状审计时填写；`conditional_substitute` 表示用户要数字但 KB 只能给条件，`unsupported` 表示现有 gold 连所需回答形式都无法支持。
- `context_requirement`：问题是否依赖图片、帖子所属社区或会话上下文。
- `excluded_from_single_turn`：仅当 `context_requirement=thread_context` 时为 `true`。校验器强制两者完全一致，不能按模型成败挑样本排除。

`question_type` 与评测列固定映射为：A=`direct`、B=`alias_or_coreference`、C=`symptom`、D=`compound`、E=`judgment_or_premise`。当前总数严格为 A=11、B=6、C=6、D=9、E=8；31 条知识扩充后，40 条主集均已有至少一个 gold，但每列指标仍单独保存分母。

每个检索阶段保留两个预先定义的人群视图：

- 完整视图 N=40，包含上下文依赖样本，用于展示系统在原始数据上的全部行为；
- 单轮视图 N=30，一致排除全部 10 条 `thread_context`，D 列因此是 n=6，而不是只排除 `q:main:031` 后的 n=8。

单轮视图仍包含 7 条 `context_requirement=image`：图片属于同一帖子/turn 的上下文，因此这里不是“纯文本充分信息”评测。若未来要回答图像问题，仍需独立视觉输入评测。

## 指标口径

- `Hit@K`：有 gold 的 query 中，Top-K 是否至少命中一条 gold。
- `Recall@K`：Top-K 命中的 gold 条目数 / gold 条目总数。
- `Recall@|gold|`：每条 query 使用自身 gold 数量作为 K，再对 query 做宏平均。
- `MRR`：第一条 gold 的倒数排名。
- `Entity accuracy`：实体归一化结果是否等于标注 `plant_id`。
- `Coverage`：`covered / partial / gap` 的数据集占比；这是知识库覆盖度，不是检索性能。
- `Abstain accuracy`：拒答集中系统没有返回高置信知识的比例。

检索指标只在 `gold_entry_ids` 非空的 query 上计算；每个指标在 JSON 中都同时保存分子/累计值与 `n`。`gap` 不被偷偷当成检索失败，而是进入覆盖度和拒答分析。

拒答同时保留两种基线：

- BM25 分数阈值：弱对照；
- 结构化双门控：先做实体 linking，再判断已识别维度是否存在 `(plant_id, dimension)` 条目。维度 linking 当前是透明的关键词规则，不使用人工 `expected_dimensions`，因此不是 oracle 结果。

## 运行

```bash
python3 scripts/validate_evaluation.py
python3 scripts/create_evaluation_split.py
python3 scripts/evaluate_baseline.py --k 3
python3 scripts/evaluate_oracle.py --k 3
python3 scripts/evaluate_dimension_ablation.py --k 3
python3 scripts/evaluate_dimension_ablation.py --predictions evaluation/results/dimension_predictions_recall.json --output evaluation/results/dimension_ablation_recall.json --k 3
python3 -m venv .venv
.venv/bin/pip install -r requirements-rerank.txt
.venv/bin/python scripts/evaluate_reranker.py --k 3
.venv/bin/python scripts/evaluate_semantic_retrieval.py --split dev
.venv/bin/python scripts/evaluate_fusion_weight_sweep.py
```

当前 31 文档、`K=3` 的冻结结果保存在 [`results/baseline.json`](results/baseline.json)。人工 `expected_dimensions` 的 oracle 结果单独保存在 [`results/oracle_dimensions.json`](results/oracle_dimensions.json)；脚本只读取已有人工标签，不生成或覆盖标签。

保守版 LLM 原始预测保存在 [`results/dimension_predictions.json`](results/dimension_predictions.json)，对应消融为 [`results/dimension_ablation.json`](results/dimension_ablation.json)；召回优先版分别保存在 [`results/dimension_predictions_recall.json`](results/dimension_predictions_recall.json) 和 [`results/dimension_ablation_recall.json`](results/dimension_ablation_recall.json)。两版 prompt 都使用合成 few-shot 示例，不向模型提供评测集的 `expected_dimensions`。持续失败样本的排名与词面证据见 [`results/case_q_main_031.json`](results/case_q_main_031.json)。完整解释见 [`results/gap_analysis.md`](results/gap_analysis.md)。

Cross-encoder 结果保存在 [`results/rerank_ablation.json`](results/rerank_ablation.json)。实验使用 [`cross-encoder/ms-marco-MiniLM-L6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)，固定 revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`；对实体和维度过滤后的全部候选打分，不先截断 BM25 Top-N。结果文件同时记录模型、包版本、设备、逐条候选分数和两个评测视图，日常测试不需要再次下载模型。

英文 query 与中文知识正文的词面审计保存在 [`results/index_language_overlap.json`](results/index_language_overlap.json)。它使用与 BM25 完全相同的 tokenizer，把实际索引拆成 `content` 与 dimension/topics/conditions/retrieval metadata 两部分；这是描述性重叠检查，不是 BM25 分数归因或因果语言消融。

## 冻结的 dev/test 切分

模型实验开始前，40 条主集已按 `SHA-256(query_id)` 摘要升序做 label-blind 20/20 切分，manifest 位于 [`splits/main_sha256_20_20_v1.json`](splits/main_sha256_20_20_v1.json)。算法只读取 query ID，不读取文本、标签或既有指标，也不做事后分层和人工换样本。

dev/test 各含 5 条 thread-context 和 5 条 partial；植物分布不平衡，尤其虎尾兰为 4/11，因此模型实验除了 aggregate 还报告 per-plant 诊断。所有模型选择、chunk 和融合权重只看 dev；冻结后 test 已运行一次，不再重跑。

## Dev-only 跨语言语义召回

预注册配置见 [`experiments/semantic_retrieval_v1.json`](experiments/semantic_retrieval_v1.json)，结果见 [`results/semantic_retrieval_dev.json`](results/semantic_retrieval_dev.json)。实验固定一条知识一个 chunk，只索引中文 `content`；比较 BGE-M3 与 multilingual-E5-large，并以 content-only dev 宏 R@5 选型。模型选择阶段 test 未编码、未评测，CLI 需要显式 `--confirm-test` 且只允许一个模型才能进入 test。

Content-only BM25 同时报 forced Top-K 与 `score > 0` 两种口径：前者会用 entry ID tie-break 填入零分候选，后者才表示实际词面信号。dev 上正分 BM25 R@5 为 5.0%（1/45 pair），BGE-M3 为 71.3%（26/45），E5 为 69.7%（26/45）。按预注册规则选择 BGE-M3；固定等权 RRF `k=60` 将 metadata+content BM25 的 R@5 从 70.1% 提高到 75.1%，但 pair 命中仍为 26/45。

dev 中虎尾兰只有 4 条，一条等于 25pp；per-plant 结果只用于发现系统性失败，不用于模型优劣判断。词面重叠 4/91 与语义 Top-K pair 命中是不同指标，继续分开报告。

## 预注册融合扫描与一次冻结 test

融合扫描在运行前固定 BGE、`RRF k=60`、BM25 权重 1.0 和五个 BGE 权重；规则规定最佳 pair 命中不超过 28/45 就保留等权融合。权重 0.50 达到 28/45，触发停止条件，因此最终仍为 1.0/1.0。配置与结果见 [`experiments/fusion_weight_sweep_v1.json`](experiments/fusion_weight_sweep_v1.json) 和 [`results/fusion_weight_sweep_dev.json`](results/fusion_weight_sweep_dev.json)。

最终 test 配置见 [`experiments/semantic_retrieval_final_v1.json`](experiments/semantic_retrieval_final_v1.json)，一次性结果见 [`results/semantic_retrieval_test.json`](results/semantic_retrieval_test.json)：content-only BM25/BGE 的 R@5 为 15.0%/65.4%，pair 为 3/46 与 26/46；metadata+content BM25/等权 RRF 的 R@5 为 78.3%/77.1%，R@|gold| 为 48.3%/58.3%，pair 均为 32/46。正文激活已复现；简单融合没有扩大命中集合，且不同排序指标方向不一致。

运行记录 [`results/semantic_retrieval_test_run_v1.json`](results/semantic_retrieval_test_run_v1.json) 固定最终配置 SHA、结果 SHA、模型 revision 和唯一执行命令。不得依据 test 调参或覆盖结果。
