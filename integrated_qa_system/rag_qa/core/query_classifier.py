import json
import os
# 导入 PyTorch
import torch
# 导入日志
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
rag_qa_path = os.path.dirname(current_dir)
# print(f'rag_qa_path--》{rag_qa_path}')
core_path = os.path.join(rag_qa_path, 'core')
sys.path.insert(0, core_path)
sys.path.insert(0, rag_qa_path)
# 获取根目录文件所在的绝对位置
project_root = os.path.dirname(rag_qa_path)
sys.path.insert(0, project_root)
from base import logger
# 导入numpy
import numpy as np
# 导入 Transformers 库
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
# 导入train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

class QueryClassifier:
    def __init__(self,model_path='bert_query_classifier'):
        self.model_path = model_path
        bert_path = os.path.join(rag_qa_path, 'models', 'bert-base-chinese')
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.model=None
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'使用设备:{self.device}')
        self.label_map={'通用知识':0,'专业咨询':1}
        self.load_model()
    def load_model(self):
        if os.path.exists(self.model_path):
            self.model=BertForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            logger.info(f'加载模型:{self.model_path}')
        else:
            self.model=BertForSequenceClassification.from_pretrained('../models/bert-base-chinese')
            self.model.to(self.device)
            logger.info("初始化新 BERT 模型")
    def train_model(self,data_file='../classify_data/model_generic_5000.json'):
        if not os.path.exists(data_file):
            logger.error(f'数据集文件{data_file}不存在')
            raise FileNotFoundError(f'数据集文件{data_file}不存在')
        with open(data_file,'r',encoding='utf-8') as f:
            data=[json.loads(value) for value in f.readlines()]
        texts=[item['query'] for item in data]
        labels=[item['label'] for item in data]
        train_texts, val_texts, train_labels, val_labels=train_test_split(texts,labels,test_size=0.2,random_state=42)
        train_encodings,train_labels=self.preprocess_data(train_texts,train_labels)
        val_encodings,val_labels=self.preprocess_data(val_texts,val_labels)
        train_dataset=self.create_dataset(train_encodings,train_labels)
        val_dataset=self.create_dataset(val_encodings,val_labels)
        training_args=TrainingArguments(
            output_dir='./bert_results',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./bert_logs",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            save_total_limit=1,  # 只保存一个检查点，即最优的模型
            metric_for_best_model="eval_loss",
            fp16=False,
        )
        trainer=Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics
        )
        logger.info('开始训练bert模型...')
        trainer.train()
        self.save_model()
        self.evaluate_model(val_encodings,val_labels)
    def predict_category(self,query):
        if self.model is None:
            logger.error('模型未训练或加载')
            return '通用知识'
        encoding=self.tokenizer(query,truncation=True,padding=True,return_tensors='pt',max_length=128)
        encoding={key:value.to(self.device) for key,value in  encoding.items()}
        with torch.no_grad():
            output=self.model(**encoding)
            prediction=torch.argmax(output.logits,dim=1).item()
            return "专业咨询" if prediction == 1 else "通用知识"
    def evaluate_model(self,encodings,labels):
        dataset=self.create_dataset(encodings,labels)
        trainer=Trainer(model=self.model)
        predictions=trainer.predict(dataset)
        pred_labels=np.argmax(predictions.predictions, axis=-1)
        true_labels=labels
        logger.info("分类报告:")
        logger.info(classification_report(
            true_labels,
            pred_labels,
            target_names=["通用知识", "专业咨询"]
        ))
        logger.info("混淆矩阵:")
        logger.info(confusion_matrix(true_labels, pred_labels))
    def compute_metrics(self,eval_pred):
        logits,labels=eval_pred
        predictions=np.argmax(logits,axis=-1)
        accuracy=(predictions==labels).mean()
        return {"accuracy": accuracy}
    def save_model(self):
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)
        logger.info(f'模型保存至{self.model_path}')
    def preprocess_data(self,texts,labels):
        encodings=self.tokenizer(
            texts,
            truncation=True,
            max_length=128,
            return_tensors='pt',
            padding='max_length',
        )
        return encodings,[self.label_map[label] for label in labels]
    def create_dataset(self,encodings,labels):
        class Dataset(torch.utils.data.Dataset):
            def __init__(self,encodings,labels):
                self.encodings=encodings
                self.labels=labels
            def __getitem__(self, index):
                item={key:value[index] for key,value in self.encodings.items()}
                item['labels']=torch.tensor(self.labels[index])
                return item
            def __len__(self):
                return len(self.labels)
        return Dataset(encodings,labels)
