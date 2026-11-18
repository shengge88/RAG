# 上传说明

本仓库保留原有 `README.md`，并导入 `integrated_qa_system/` 源码。包含与排除项见根目录 `SNAPSHOT_MANIFEST.txt`。

## 未包含，需要自行准备

- `integrated_qa_system/config.ini`：原文件含数据库密码和客服电话，未纳入，也未在此重建。程序会读取该文件；请在本地自行创建，不要把真实密码提交到公开仓库。
- 模型权重与训练产物未上传，包括 `rag_qa/models/`（如 bge-m3、bge-reranker-large、bert-base-chinese）、`rag_qa/core/bert_results/`，以及 `rag_qa/core/bert_query_classifier/model.safetensors`。分类器目录只保留了小型配置和分词文件。
- `rag_qa/rag_assesment/.env` 未提交。触发 GitHub 推送保护的旧 key 已从上传副本删除。同目录只保留空模板 `.env.example`（`GOOGLE_API_KEY=`，无真实凭据）。`.gitignore` 会忽略 `.env`。

## 其他

- 源项目没有 `requirements.txt`，这里不补充，也不猜测依赖版本。
- 本地原项目文件未改动。`.idea/`、缓存和日志已排除。业务源码未改逻辑。
