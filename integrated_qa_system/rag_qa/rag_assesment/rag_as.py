# 导入pandas库，用于数据处理和保存CSV文件
import pandas as pd
from langchain_ollama import ChatOllama, OllamaEmbeddings
from dotenv import load_dotenv
load_dotenv('.env')
# 导入ragas库的evaluate函数，用于执行RAG评估
from ragas import evaluate
# 导入ragas的评估指标，包括忠实度、答案相关性、上下文相关性和上下文召回率
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
# 导入datasets库的Dataset类，用于构建RAGAS所需的数据格式
from datasets import Dataset
# 导入json库，用于加载JSON格式的评估数据集
import json,os
with open('rag_evaluate_data.json','r',encoding='utf-8') as f:
    data = json.load(f)
eval_data={
    "question": [item["question"] for item in data],
    # 提取每个数据条目的answer字段，组成答案列表
    "answer": [item["answer"] for item in data],
    # 提取每个数据条目的context字段，组成上下文列表（每个context为列表）
    "contexts": [item["context"] for item in data],
    # 提取每个数据条目的ground_truth字段，组成真实答案列表
    "ground_truth": [item["ground_truth"] for item in data]
}
dataset=Dataset.from_dict(eval_data)


llm = ChatOllama(
    model="qwen2.5:7b",
    base_url="http://localhost:11434",
    temperature=0,
)
embeddings = OllamaEmbeddings(
    model="mxbai-embed-large",
    base_url="http://localhost:11434",
)
result=evaluate(
    dataset=dataset,
    metrics=[faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,],
            llm=llm,
    embeddings=embeddings,
)
print("RAGAS评估结果：")
# 打印评估结果，包含各指标的分数
print(result)

