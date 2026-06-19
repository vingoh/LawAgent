# 项目计划（Plans）

本目录存放 LawAgent 相关的实现计划，作为项目文档归档。

| 文件 | 说明 | 状态 |
|------|------|------|
| [build_index_plan_7b9cb35e.plan.md](build_index_plan_7b9cb35e.plan.md) | 初版：BM25 索引 + corpus + Macro F1 评估 | 已被后续版本取代 |
| [build_index_plan_c8033d89.plan.md](build_index_plan_c8033d89.plan.md) | 修订版：过滤规则、跨语言策略、评估明细 | 已实现 |
| [bm25_german_preprocessing_b22d6480.plan.md](bm25_german_preprocessing_b22d6480.plan.md) | BM25 德语预处理（停用词、stem、citation token） | 已实现 |
| [query_eval_decouple_plan.plan.md](query_eval_decouple_plan.plan.md) | Query / Eval 解耦：predictions CSV 接口 | 已实现 |
| [bm25_split_rrf.plan.md](bm25_split_rrf.plan.md) | BM25 court/law 分路召回 + 加权 RRF 融合 | 已实现 |
| [query_rewrite.plan.md](query_rewrite.plan.md) | LLM Query Rewrite：pipeline 第一步 + BM25 search_text | 已实现 |

新计划请创建在本目录，命名建议：`<主题>_<简短描述>.plan.md`。
