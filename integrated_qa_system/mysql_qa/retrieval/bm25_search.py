from rank_bm25 import BM25Okapi
# 导入数值计算库
import numpy as np
import os,sys
current_path=os.path.abspath(__file__)
model_path=os.path.abspath(os.path.dirname(current_path))
root_path=os.path.dirname(os.path.dirname(model_path))
new_root=os.path.join(root_path,'mysql_qa')
sys.path.insert(0,root_path)
sys.path.insert(0,new_root)
from base.logger import logger
# 导入文本预处理
from utils.preprocess import preprocess_text
class BM25Search:
    def __init__(self,redis_client,mysql_client):
        self.logger = logger
        # 初始化 Redis 客户端
        self.redis_client = redis_client
        # 初始化 MySQL 客户端
        self.mysql_client = mysql_client
        # 初始化 BM25 模型
        self.bm25 = None
        # 初始化问题列表
        self.questions = None
        # 初始化原始问题
        self.original_questions = None
        # 加载数据
        self._load_data()
    def _load_data(self):
        original_key = "qa_original_questions"
        tokenized_key = "qa_tokenized_questions"
        self.original_questions=self.redis_client.get_data(original_key)
        tokenized_questions=self.redis_client.get_data(tokenized_key)
        if not self.original_questions or not tokenized_questions:
            rows=self.mysql_client.fetch_question()
            if not rows:
                self.logger.warning('未加载到问题')
                return
            # 统一成字符串列表，避免元组/字符串混用
            self.original_questions=[q[0] for q in rows]
            tokenized_questions=[preprocess_text(q) for q in self.original_questions]
            self.redis_client.set_data(original_key,self.original_questions)
            self.redis_client.set_data(tokenized_key,tokenized_questions)
        self.questions=tokenized_questions
        self.bm25=BM25Okapi(self.questions)
        self.logger.info('BM25 模型初始化完成')
    def _softmax(self,scores):
        exp_scores=np.exp(scores-np.max(scores))
        return exp_scores / exp_scores.sum()
    def search(self,query,threshold=0.85):
        if not query or not isinstance(query,str):
            self.logger.error('无效查询')
            return None,False
        cached_answer=self.redis_client.get_answer(query)
        if cached_answer:
            return cached_answer,False
        try:
            if self.bm25 is None or not self.original_questions:
                self.logger.error('BM25 未初始化')
                return None,True
            query_tokens=preprocess_text(query)
            scores=self.bm25.get_scores(query_tokens)
            softmax_scores=self._softmax(scores)
            best_idx=int(softmax_scores.argmax())
            best_score=float(softmax_scores[best_idx])
            if best_score >= threshold:
                original_question=self.original_questions[best_idx]
                answer=self.mysql_client.fetch_answer(original_question)
                if answer:
                    self.redis_client.client.set(f'answer:{query}',answer)
                    self.logger.info(f"搜索成功，Softmax 相似度: {best_score:.3f}")
                    # 返回答案和 False
                    return answer, False
            self.logger.info(f"未找到可靠答案，最高 Softmax 相似度: {best_score:.3f}")
            # 返回 None 和 True
            return None, True
        except Exception as e:
            self.logger.error(f'搜索失败:{e}')
            return None,True
