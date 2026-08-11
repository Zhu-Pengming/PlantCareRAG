# 植物知识库 v0.1

## 目标

这层只保存经过整理、可以追溯到来源的事实。向量、embedding、chunk 和模型生成结果都不属于事实层。

知识库采用三类对象：

- `plant`：植物实体与别名归一，例如“虎尾兰”和旧学名 `Sansevieria trifasciata` 都指向 `plant:dracaena_trifasciata`。
- `source`：来源登记表，记录页面、机构、访问日期和使用限制。
- `entry`：原子知识条目；一条只表达一个可独立检索和引用的结论。

## 目录

```text
knowledge_base/
├── data/
│   ├── plants.json
│   ├── entity_aliases.json
│   ├── sources.json
│   └── entries/
│       ├── dracaena_trifasciata.json
│       ├── epipremnum_aureum.json
│       └── monstera_deliciosa.json
└── schemas/
    ├── entry.schema.json
    ├── entity_alias.schema.json
    ├── plant.schema.json
    └── source.schema.json
```

## 建模规则

1. **实体先归一**：知识条目只能引用 `plants.json` 中存在的 `plant_id`。
2. **一条一个结论**：光照、浇水、症状原因分别建条目，不把整篇养护指南塞进一个 chunk。
3. **结论保留条件**：例如“耐弱光”不能改写成“弱光最好”；季节、室内环境、品种例外应写入 `conditions`。
4. **诊断使用关联语气**：来源只说“过度浇水可能导致烂根”时，条目不能写成“烂根一定由过度浇水导致”。
5. **每条可追溯**：至少一个 `source_ref`，定位到具体页面和段落/栏目。
6. **只存改写后的短事实**：不保存网页全文。当前来源统一按“链接 + 事实性改写”使用，版权状态仍标记为待复核。
7. **中文名是检索别名**：未经分类学来源核验的中文俗名不作为 accepted scientific name 的证据。
8. **版本显式记录**：所有对象必须包含 `schema_version`，结构升级时按版本迁移，不静默改变字段含义。
9. **别名验收与来源分开**：中文别名继续保留用于未来产品输入，但当前英文评测不覆盖它们，统一标记 `evaluation_status: untested`；`千岁兰` 仍保持歧义且禁止自动映射。

## v0.1 范围

当前范围为 3 种常见室内植物、31 条可追溯知识：

- 龟背竹（`Monstera deliciosa`）
- 绿萝（`Epipremnum aureum`）
- 虎尾兰（`Dracaena trifasciata`，旧名 `Sansevieria trifasciata`）

来源包括 GBIF、North Carolina Extension、ASPCA、RHS、University of Minnesota、University of Vermont、University of Wisconsin、University of Connecticut 与 Virginia Tech。它们不是最终大规模语料，而是后续采集和评审的模板。

## 条目维度

`taxonomy`, `plant_attribute`, `lighting`, `watering`, `humidity`, `temperature`, `soil`, `fertilizer`, `repotting`, `symptom`, `disease`, `pest`, `pet_safety`

维度是检索过滤字段，不是文件夹。一个用户问题可以触发多个维度，但每条事实只指定一个主维度。

## 验收门槛

进入可检索知识库前，每条数据必须满足：

- 校验脚本通过；
- 有植物实体、主维度、主题、适用条件和来源定位；
- 内容是短事实，不含无来源的固定浇水日历或确定性诊断；
- `review_status` 为 `approved`；
- 高风险内容（毒性、农药、疾病处置）至少经过第二来源或人工复核。

## 当前冻结点

先不增加植物种类，也不继续为数量补条目。31 条知识已经覆盖首轮 `answer_shape`、宠物安全、常见症状鉴别及湿度/温度/浇水动作缺口：

1. 3 种植物均有 `pet_safety`，且每条至少引用两个来源；
2. 高频症状已扩成多原因 gold，不再把黄叶或黑斑压成单一原因；
3. 4 条浇水频率问题已有“实用区间 + 环境条件”并列知识；固定毫升数仍不做无来源承诺；
4. 中文别名保留但标为 `untested`，不计入当前英文评测结论。

召回优先版 few-shot 预测维度已经将 Recall@|gold| 从实体过滤的 57.3% 提高到 71.0%，接近 72.2% oracle。下一步在这份冻结数据上测试维度内 rerank/query decomposition。当前单轮 Reddit 数据不能验证会话实体继承；若要做指代消解，需要另建多轮评测集。换盆、施肥、繁殖和支撑等维度继续由结构门控明确拒答，待真实 query 证明优先级后再补。
