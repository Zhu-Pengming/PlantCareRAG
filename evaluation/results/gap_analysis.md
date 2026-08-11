# 评测闭环与缺口分析 v0.2

## 结论

知识库已在同一 3 种植物上从 13 条扩到 31 条，来源从 6 个增至 16 个。当前应冻结事实层，不再扩植物或追求更高的 BM25 总分。召回优先版 few-shot 维度分类器已吃到 91.6% 的 oracle 增益，把实体过滤后的 Recall@|gold| 从 57.3% 提到 71.0%；dimension-aware 仍是当前最大且已经实测成立的单一效果。

第一版 cross-encoder rerank 已完成。完整 N=40 上 Recall@|gold| 从 71.0% 升到 72.8%，但 Recall@5 从 87.6% 降到 86.8%；一致排除全部 10 条 thread-context 后，N=30 上分别为 79.1%→79.9% 和 95.2%→96.0%，其中 +0.8pp Recall@|gold| 只来自一条 query。结论是 **rerank 未在本规模上表现出稳健收益**；下一步应先验证语言匹配的文档表示，而不是在这 40 条上反复换模型。

扩充后，40 条主集全部有 gold；覆盖状态为 `covered` 30 条（75%）、`partial` 10 条（25%）、`gap` 0 条。gold 也从偏单一的“相关条目”改成能支撑多原因鉴别的集合，因此原始 BM25 的 Recall@|gold| 从 68.6% 降到 53.0%。这个下降是预期结果：知识和标注更完整后，词法检索无法同时覆盖多个原因的缺口被显露出来了。

## 当前 baseline

配置：31 个文档、40 条有 gold 的 query、`K=3`。每次返回量只占语料的 9.7%，不再是 13 文档阶段的 23.1%。

| 模式 | N | Hit@3 | Recall@3 | Recall@5 | Recall@\|gold\| | MRR |
|---|---:|---:|---:|---:|---:|---:|
| 原始 BM25 | 40 | 85.0% | 61.2% | 74.2% | 53.0% | 76.9% |
| 实体过滤 BM25 | 40 | 90.0% | 68.7% | 81.9% | 57.3% | 81.0% |

gold 数量分布为：1 条 14 个、2 条 13 个、3 条 4 个、4 条 7 个、5 条 1 个、6 条 1 个。由于 9 条 query 的 gold 超过 3 条，Recall@3 的理论宏平均上限现在是 93.4%，C、D、E 列的上限分别是 85.0%、83.3%、96.9%。因此 Recall@3 必须和 Recall@5、Recall@|gold| 一起解释。

### 完整与单轮人群

在排除 `q:main:031` 前先对全量数据执行一致性检查，发现共有 10/40 条 `context_requirement=thread_context`：`008`、`013`、`020`、`021`、`023`、`027`、`031`、`034`、`035`、`036`。其中 D 列不只有 `031`，还包括 `035`、`036`。

因此主报告固定保留两个视图：完整视图 N=40、统一排除上述 10 条的单轮视图 N=30。D 列分母分别为 9 和 6；不能报告“只排除 031”的 n=8，因为那会按已观察到的失败针对性剔除样本。数据 schema 与校验器强制 `excluded_from_single_turn=true` 当且仅当 `context_requirement=thread_context`。

这里的“单轮”只排除需要帖子线程/外部会话信息的样本；7 条同一帖子内的 image-context 样本仍保留，所以 N=30 不是纯文本信息充分集。完整表用于不隐藏原始数据行为，N=30 用于隔离当前单轮文本系统可以合理承担的部分。

### 扩充前后

| 冻结点 | 文档/有 gold query | 原始 R@\|gold\| | 实体 R@\|gold\| |
|---|---:|---:|---:|
| v0.1 | 13 / 35 | 68.6% | 69.5% |
| v0.2 | 31 / 40 | 53.0% | 57.3% |

这不是同一 gold 集上的模型回归：v0.2 为原有 query 补了多原因 gold，并让原先没有 gold 的问题进入检索分母。它说明 v0.1 的高分主要来自单 gold 与小语料，而不是检索已经解决了症状鉴别。从现在起所有模型和消融对比只使用 v0.2；v0.1 只作为“扩容如何改变任务难度”的历史注脚。

### 按问法列

当前所有列的总数就是检索分母。

| 列 | N | 原始 Hit@3 | 原始 R@3 | 原始 R@5 | 原始 R@\|gold\| | 实体 Hit@3 | 实体 R@3 | 实体 R@5 | 实体 R@\|gold\| |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A 直接属性 | 11 | 90.9% | 68.2% | 81.8% | 59.1% | 100.0% | 90.9% | 100.0% | 68.2% |
| B 别名/指代 | 6 | 66.7% | 58.3% | 100.0% | 41.7% | 66.7% | 58.3% | 100.0% | 41.7% |
| C 症状描述 | 6 | 83.3% | 42.8% | 46.1% | 46.1% | 83.3% | 42.8% | 54.4% | 46.1% |
| D 复合条件 | 9 | 77.8% | 43.5% | 51.9% | 46.3% | 88.9% | 49.1% | 58.3% | 54.6% |
| E 判断/前提 | 8 | 100.0% | 87.5% | 90.6% | 65.6% | 100.0% | 87.5% | 90.6% | 65.6% |

在 13 文档阶段，实体过滤在 D 列只改善覆盖完整度，Hit@3 不变；扩到 31 文档后，它同时把 D 列 Hit@3 从 77.8% 提到 88.9%、Recall@5 从 51.9% 提到 58.3%、Recall@|gold| 从 46.3% 提到 54.6%。这说明实体约束在候选干扰增加后开始同时帮助首命中和多原因覆盖，但 D 列只有 9 条，仍应作为局部证据报告。

## 人工维度 oracle

oracle 直接读取采集时人工标注的 `expected_dimensions`，不使用 LLM 重标，也不改写 gold。完整性检查通过：40 条 query 的所有 gold 维度都包含在对应的 `expected_dimensions` 中。

| 场景 | 实体 | 维度 | 全集 R@\|gold\| | D Hit@3 | D R@5 | D R@\|gold\| |
|---|---|---|---:|---:|---:|---:|
| 实体过滤 baseline | 当前 linker | 不过滤 | 57.3% | 8/9 | 58.3% | 54.6% |
| 维度 oracle | 当前 linker | 人工标签 | 72.2% | 8/9 | 66.7% | 54.6% |
| 实体 oracle | 人工 `plant_id` | 不过滤 | 60.9% | 9/9 | 73.1% | 64.8% |
| 完整路由 oracle | 人工 `plant_id` | 人工标签 | 78.9% | 9/9 | 81.5% | 64.8% |

oracle 结论分两层：

- aggregate 上，完美维度识别能把 57.3% 提到 72.2%，有 14.9 个百分点空间，而且测在当前全部 40 条 query 上。
- D 列上，维度 oracle 只改善较深的 Recall@5，Recall@|gold| 仍为 54.6%；这说明维度过滤已减少跨维度噪声，但同一维度内部仍需要 rerank。

D 列当前是 9 条，不是旧冻结点的 8 条；一条样本对应 11.1 个百分点。报告固定展示 `8/9`、`9/9` 等绝对数，不为这个小样本给出虚假的稳定性暗示。机器可读结果见 [`oracle_dimensions.json`](oracle_dimensions.json)。

## Few-shot 维度分类器与四档消融

两版均使用 `gpt-5.6-terra`、low reasoning effort；prompt 只包含维度定义、合成示例和 40 条 `query_id + raw_text`，不包含 `expected_dimensions` 或 gold，输出也未人工修改。

| 配置项 | 值 |
|---|---|
| 模型 | `gpt-5.6-terra` |
| reasoning effort | `low` |
| 官方配置页 | [OpenAI model page](https://developers.openai.com/api/docs/models/gpt-5.6-terra) |
| 访问日期 | 2026-08-11 |
| 保守版 | prompt v1.0；[`dimension_predictions.json`](dimension_predictions.json) |
| 召回优先版 | prompt v1.1；[`dimension_predictions_recall.json`](dimension_predictions_recall.json) |

召回优先版只迭代一次：明确“漏掉维度会永久删除 gold，多预测一个维度只会放宽候选集，拿不准时一并输出”，并增加 2 条隐含浇水的合成示例。预先停止条件是 R@5 必须超过 87%，否则停止并接受旧版；实际达到 87.6%，因此验收后停止调 prompt。

| 分类/检索指标 | 保守版 v1.0 | 召回优先版 v1.1 | oracle |
|---|---:|---:|---:|
| micro-Precision | 100.0% | 100.0% | — |
| micro-Recall | 87.5% | 92.2% | 100.0% |
| micro-F1 | 93.3% | 95.9% | — |
| 完全匹配 | 32/40 | 35/40 | 40/40 |
| 检索 R@5 | 84.7% | 87.6% | 90.1% |
| 检索 R@\|gold\| | 70.8% | 71.0% | 72.2% |

Precision 没有像预期一样下降；召回优先版在 64 个 gold 标签中命中 59 个，仍然只有 5 个漏标、没有多标：`watering` 2 次、`lighting` 2 次、`symptom` 1 次。`disease`、`pest`、`temperature`、`repotting` 各只有 1 个正样本，它们的 100% 不能解释为稳定能力。

采用召回优先版后，四档消融使用完全相同的 v0.2 数据与 N=40：

| 检索阶段 | Hit@3 | R@5 | R@\|gold\| | 相对上一档 |
|---|---:|---:|---:|---:|
| BM25 | 85.0% | 74.2% | 53.0% | — |
| + 实体过滤 | 90.0% | 81.9% | 57.3% | +4.4pp |
| + 预测维度 | 97.5% | 87.6% | 71.0% | +13.6pp |
| + oracle 维度 | 97.5% | 90.1% | 72.2% | +1.3pp |

预测维度取得 `(71.0 - 57.3) / (72.2 - 57.3) = 91.6%` 的可用 oracle 增益，分类器继续投入的上限已经很低。

D 列必须看绝对数：

| 检索阶段 | D Hit@3 | D R@5 | D R@\|gold\| |
|---|---:|---:|---:|
| BM25 | 7/9 | 51.9% | 46.3% |
| + 实体过滤 | 8/9 | 58.3% | 54.6% |
| + 预测维度 | 8/9 | 63.9% | 54.6% |
| + oracle 维度 | 8/9 | 66.7% | 54.6% |

保守版曾得到 57.4% D R@|gold|，是 `q:main:033` 漏标 `watering` 后候选集意外变窄造成的排序假提升；召回优先版补回该标签后，D R@|gold| 回到与 oracle 相同的 54.6%。维度过滤在 D 列的可靠作用体现在 R@5，而不是首命中或 Top-|gold|。完整召回版消融见 [`dimension_ablation_recall.json`](dimension_ablation_recall.json)。

## Cross-encoder rerank

使用 [`cross-encoder/ms-marco-MiniLM-L6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)，固定模型 revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`。对当前实体 linker 与预测/oracle 维度过滤后的**全部候选**打分，不先按 BM25 截断；cross-encoder 分数相同时才用 BM25 和 entry ID 稳定排序。运行环境为 CPU、`sentence-transformers 5.1.2`、`transformers 4.57.6`、`torch 2.8.0`。

为保留原有分类器上界，主表展示六档而不是删除 pre-rerank oracle：

| 完整视图 | N | Hit@3 | R@5 | R@\|gold\| |
|---|---:|---:|---:|---:|
| BM25 | 40 | 85.0% | 74.2% | 53.0% |
| + 实体过滤 | 40 | 90.0% | 81.9% | 57.3% |
| + 预测维度 | 40 | 97.5% | 87.6% | 71.0% |
| + oracle 维度 | 40 | 97.5% | 90.1% | 72.2% |
| + 预测维度 + rerank | 40 | 92.5% | 86.8% | 72.8% |
| + oracle 维度 + rerank | 40 | 92.5% | 89.8% | 75.8% |

| 单轮视图（排除全部 thread-context） | N | Hit@3 | R@5 | R@\|gold\| |
|---|---:|---:|---:|---:|
| BM25 | 30 | 86.7% | 78.4% | 55.1% |
| + 实体过滤 | 30 | 93.3% | 88.7% | 60.9% |
| + 预测维度 | 30 | 100.0% | 95.2% | 79.1% |
| + oracle 维度 | 30 | 100.0% | 96.8% | 79.1% |
| + 预测维度 + rerank | 30 | 96.7% | 96.0% | 79.9% |
| + oracle 维度 + rerank | 30 | 96.7% | 98.3% | 82.2% |

主要读数：

- 预测维度下，rerank 的净收益是完整集 +1.9pp、单轮集 +0.8pp R@|gold|；单轮提升只来自 `q:main:033` 一条 query 把一个 gold 推进 Top-|gold|，证据强度有限。
- oracle 维度 + rerank 在单轮集达到 82.2%，比 pre-rerank oracle 高 3.2pp，说明维度内排序仍有真实空间；预测维度版本只有 79.9%，因为漏掉的维度会在 rerank 前永久删除候选。
- 完整集 R@5 下降主要与 context-lost 样本有关：`q:main:027` 的实体为空，reranker 把错误植物的症状条目置顶；`q:main:034` 也有一条 gold 从第 5 降到第 6。统一排除 thread-context 后 R@5 不降反升。
- Hit@3 不是单调改善：单轮集从 30/30 降到 29/30，失败是仍保留的 image-context `q:main:025`；它的 gold 从第 3/5 移到第 4/5，R@5 不变，但首命中跌出 Top-3。这说明本实验不能只看宏平均 R@|gold|。

D 列继续报告绝对数：

| 阶段 | 完整 D Hit@3 | 完整 D R@5 | 完整 D R@\|gold\| | 单轮 D Hit@3 | 单轮 D R@5 | 单轮 D R@\|gold\| |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 7/9 | 51.9% | 46.3% | 5/6 | 69.4% | 61.1% |
| + 实体过滤 | 8/9 | 58.3% | 54.6% | 6/6 | 79.2% | 73.6% |
| + 预测维度 | 8/9 | 63.9% | 54.6% | 6/6 | 87.5% | 73.6% |
| + oracle 维度 | 8/9 | 66.7% | 54.6% | 6/6 | 91.7% | 73.6% |
| + 预测维度 + rerank | 8/9 | 66.7% | 57.4% | 6/6 | 91.7% | 77.8% |
| + oracle 维度 + rerank | 8/9 | 69.4% | 60.2% | 6/6 | 95.8% | 81.9% |

这版 cross-encoder 在 MS MARCO 英文 passage ranking 上训练，而 KB `content` 为中文。对实际索引做拆分审计后，91 个 query–gold 对中有 88 对与 metadata 重叠，只有 4 对与 `content` 重叠；84 对是 metadata-only，content-only 为 0。这说明当前真正工作的主通道是结构化 metadata，而中文正文缺少英文词面可达性；重叠统计仍不是因果语言消融，不能直接解释为“cross-encoder 架构无效”。下一步先用 content-only 多语语义召回测试英文 query 能否直接命中中文事实，再决定是否做 hybrid retrieval 和 rerank。完整结果见 [`rerank_ablation.json`](rerank_ablation.json) 和 [`index_language_overlap.json`](index_language_overlap.json)。

## 多语 embedding 激活正文通道：dev 发现，test 复现

模型选型仅使用哈希 dev 20 条、45 个 query–gold 对；test 未编码或评测。一条知识 entry 固定为一个 chunk，content-only 方法不读取实体、维度或其他 metadata。BGE-M3 不加 query instruction；multilingual-E5-large 按模型说明使用 `query:` / `passage:` 前缀，两者均归一化后做余弦相似度。

| Dev 方法 | Hit@3 | R@5 | R@\|gold\| | Pair@5 |
|---|---:|---:|---:|---:|
| Content-only BM25，正分候选 | 5.0% | 5.0% | 5.0% | 1/45 |
| Content-only BM25，强制 Top-5 | 10.0% | 22.5% | 11.3% | 7/45 |
| Content-only BGE-M3 | 80.0% | 71.3% | 52.6% | 26/45 |
| Content-only multilingual-E5-large | 75.0% | 69.7% | 39.7% | 26/45 |
| Metadata + content BM25 | 85.0% | 70.1% | 57.6% | 26/45 |
| BM25 + BGE-M3 固定 RRF | 85.0% | 75.1% | 59.7% | 26/45 |

正文通道在 dev 上被明确激活：BGE-M3 相对正分 content-only BM25 的 R@5 提高 66.3pp、pair 命中从 1/45 增至 26/45。BGE 只比 E5 高 1.7pp R@5，且 pair 命中相同；选择 BGE 是执行预注册规则，不是模型普遍优越性的证据。

BM25 与 BGE 各命中 26 个 pair，但只重合 19 个，各自独有 7 个，并集 33 个；固定 RRF 仍命中 26 个。这说明两个通道互补，但简单融合没有吃完 oracle union。RRF 的宏 R@5 提高 5.0pp，R@|gold| 提高 2.1pp，而 pair 总数不变。完整结果见 [`semantic_retrieval_dev.json`](semantic_retrieval_dev.json)，预注册配置见 [`../experiments/semantic_retrieval_v1.json`](../experiments/semantic_retrieval_v1.json)。

进入 test 前只允许一次五点加权 RRF 扫描，并预注册“最佳 pair ≤28/45 就保留等权”停止条件。最佳 BGE 权重 0.50 恰为 28/45，因此按规则接受原等权 RRF，不继续调参。见 [`fusion_weight_sweep_dev.json`](fusion_weight_sweep_dev.json) 与 [`../experiments/fusion_weight_sweep_v1.json`](../experiments/fusion_weight_sweep_v1.json)。

| 冻结 Test N=20 | Hit@3 | R@5 | R@\|gold\| | Pair@5 |
|---|---:|---:|---:|---:|
| Content-only BM25，正分候选 | 15.0% | 15.0% | 15.0% | 3/46 |
| Content-only BGE-M3 | 80.0% | 65.4% | 49.2% | 26/46 |
| Metadata + content BM25 | 85.0% | 78.3% | 48.3% | 32/46 |
| BM25 + BGE-M3 冻结等权 RRF | 95.0% | 77.1% | 58.3% | 32/46 |

Test 复现了正文激活：content-only R@5 提高 50.4pp。通道互补性弱于 dev：BM25/BGE 重合 24、各自独有 8/2、并集 34；RRF 仍为 32。融合把 Hit@3 和 R@|gold| 分别提高 10.0pp，却让宏 R@5 下降 1.25pp；因此结论是“简单 RRF 改变并部分改善前排排序，但未稳定扩大覆盖”，不是笼统的正向提升。最终配置、结果和一次运行记录分别见 [`../experiments/semantic_retrieval_final_v1.json`](../experiments/semantic_retrieval_final_v1.json)、[`semantic_retrieval_test.json`](semantic_retrieval_test.json) 和 [`semantic_retrieval_test_run_v1.json`](semantic_retrieval_test_run_v1.json)。

### 持续失败 case：`q:main:031`

这条 query 描述湿土、褐斑与黄叶，要求 `diagnosis_list`；标注为 `thread_context + partial`，当前文本没有出现 Pothos/绿萝，实体 linker 因此返回 `null`。

| 场景 | 4 条 gold 的排名（肥盐 / 过水 / 浇水 / 光照） |
|---|---|
| 未过滤全库 | 7 / 13 / 17 / 25 |
| + 预测维度 | 6 / 12 / 14 / 被漏标排除 |
| + oracle 维度 | 6 / 12 / 14 / 18 |
| + 人工实体 | 3 / 5 / 7 / 9 |
| + 人工实体和维度 | 3 / 5 / 6 / 7 |
| 预测维度 + cross-encoder | 6 / 10 / 9 / 被漏标排除 |
| oracle 维度 + cross-encoder | 6 / 10 / 9 / 16 |

它不是“所有 gold 都在 20 名以后”的纯 lexical miss：第一条 gold 在全库排第 7；一旦提供帖子中已知的植物实体，第一条 gold 到第 3，Hit@3 即可恢复。但多 gold 完整度确实存在 lexical gap：肥盐条目与 query 重叠 `leaf/leaves/yellow` 3 个 token，过水和浇水条目分别只重叠 `leaf`、`soil`，光照条目零重叠。知识正文为中文，部分条目的英文检索词又没有覆盖 `wet soil / brown spots / yellow leaves` 这组说法。

因此这条失败由两层组成：首命中失败主要是帖子上下文中的植物实体缺失，完整召回失败则同时包含英文词面覆盖不足。实际 cross-encoder 结果验证了边界：预测和 oracle 维度下第一条 gold 都只到第 6，Top-5 无 gold；模型没有信息判断用户说的是绿萝，分类器漏掉的光照候选也无法由 rerank 补回。`q:main:031` 继续保留在 N=40 完整视图和 case study，但与其余 9 条 thread-context 一起排除在 N=30 单轮视图之外。词面证据见 [`case_q_main_031.json`](case_q_main_031.json)，cross-encoder 排名见 [`rerank_ablation.json`](rerank_ablation.json)。

## 拒答双门控

结构门控继续采用两级判断：先做实体 linking，再检查识别出的每个维度是否存在 `(plant_id, dimension)` 条目。它不读取人工 `expected_dimensions`。

- 拒答集：10/10 正确拒答，并能对缺失维度给出解释。
- fully covered 主集：24/30 被接受，接受率 80.0%。
- 无上下文依赖的 fully covered 子集：16/16 被接受。
- 6 条误拒全部标注为 `thread_context`：其中 5 条当前文本没有可唯一链接的实体，1 条虽有实体但关键词维度分类失败。

因此这 6 条不应通过放松门控解决。但当前数据来自单轮 Reddit 帖子，它的缺失实体依赖原帖标题、配图或社区语境，并不存在可供继承的“上一轮对话”。现有数据只能评测外部帖子上下文注入，不能评测会话 `plant_id` 继承；后者必须先新建多轮评测集。BM25 固定阈值在当前拒答集只命中 1/10，进一步说明拒答不能被简化为相似度阈值。

## Answer shape 审计

首轮发现的 5 条答案形状缺口中，4 条浇水频率问题已经由“实用时间区间 + 土壤/季节条件”并列条目支持，且保留原来的条件性养护知识，没有把来源改写成无条件日历。

剩余 `q:main:019` 询问虎尾兰一次要浇多少水。当前 KB 能给出“浇透、避开叶丛中心、倒掉托盘积水”等动作，但没有可靠来源支持脱离盆径、基质和环境的固定毫升数，因此仍标为 `partial + conditional_substitute`。原始 BM25 的 Top-3 没召全它的 2 条 gold，实体过滤 BM25 召全；即使召全，生成层也应说明为什么不能给固定体积，而不是编一个数字。

## 已完成的知识补充

- 3 条 `pet_safety`：三种植物分别覆盖猫狗风险、相关毒性成分/机制和常见暴露症状；每条至少有两个来源引用。
- 8 条症状鉴别：补充龟背竹细菌性叶斑、炭疽与根茎腐烂，绿萝自然老叶、缺水与肥盐，虎尾兰过水/缺水迹象。
- 4 条实用浇水：三种植物的频率区间，以及虎尾兰的正确浇水动作。
- 3 条环境知识：绿萝湿度、温度与龟背竹高湿叶斑风险。

完整机器可读结果见 [`baseline.json`](baseline.json)。

## 仍未关闭的 10 条 partial

| query | 剩余边界 |
|---|---|
| `q:main:004` | 问题同时比较 split-leaf philodendron，后者不在当前实体范围。 |
| `q:main:010` | 请求“几乎无光”植物推荐，现有事实只能纠正光照前提。 |
| `q:main:019` | 缺少且不应臆造固定毫升数。 |
| `q:main:024` | 没有实际温度、持续时间等诊断条件。 |
| `q:main:025`–`027` | 症状鉴别仍缺虫害或更多排查证据；部分依赖图片/帖子上下文。 |
| `q:main:031` | 褐斑、黄叶和湿土的组合仍需要病害/根部检查信息。 |
| `q:main:037` | 同时要求换盆处理，而 `repotting` 仍是明确缺失维度。 |
| `q:main:039` | 同时询问不在库内的 peace lily。 |

## 数据集边界

当前 40 条主集全部是带可回溯链接的英文 Reddit 问法。中文/跨语言用户输入不在当前 scope 内，结果不代表中文用户性能。KB 事实正文为中文、检索 metadata 含中英文，这一文档表示限制已单独纳入 rerank 结果解释。中文别名继续保留供未来产品输入，但全部标为 `evaluation_status: untested`；歧义词 `千岁兰` 保持不可自动映射。

## 封版决定

1. 冻结 31 条知识、40 条 query、gold、10 条 thread-context 排除标记、BGE revision，以及 [`../splits/main_sha256_20_20_v1.json`](../splits/main_sha256_20_20_v1.json) 中的 dev/test 成员。
2. 维度 prompt、embedding 模型与 RRF 调参全部停止；test 不重跑、不用于方案修改。
3. 对外分别报告正文通道与混合系统，不把 dev 的 +5.0pp R@5 当作最终系统提升；冻结 test 的混合 R@5 实际为 -1.25pp。
4. 暂不实现“会话实体继承”。若要验证它，先建立真正带前序 turn 的多轮评测集；如果只加入 Reddit 标题/图片，则把模块命名为帖子上下文注入。
5. 本项目在“提出方法 → 验证 → 审计定位语言瓶颈 → 语义召回复现 → 保留融合负结果”处封版；后续研究使用新版本和新评测集。
