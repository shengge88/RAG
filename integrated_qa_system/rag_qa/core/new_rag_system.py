import os,sys

from core.query_classifier import rag_qa_path

now_path=os.path.dirname(os.path.dirname(__file__))
root_path=os.path.dirname(now_path)
sys.path.insert(0,root_path)
from prompts import RAGPrompts
from vector_store import VectorStore
#   导入 time 模块，用于计算时间
import time
from base.logger import logger
from base.config import Config
from query_classifier import QueryClassifier  #   导入查询分类器
from strategy_selector import StrategySelector  #   导入
conf=Config()
class RAGSystem:

    def __init__(self,vector_store,llm):
        self.vector_store = vector_store
        #   设置大语言模型调用函数
        self.llm = llm
        #   获取 RAG 提示模板
        self.rag_prompt = RAGPrompts.rag_prompt()
        #   初始化查询分类器
        classifier_path=os.path.join(rag_qa_path,'core','bert_query_classifier')
        self.query_classifier = QueryClassifier(model_path=classifier_path)
        #   初始化策略选择器
        self.strategy_selector = StrategySelector()

    def _retrieve_with_subqueries(self,query,source_filter=None):
        logger.info(f"使用子查询策略进行检索 (查询: '{query}')")
        subquery_prompt_template=RAGPrompts.subquery_prompt()
        try:
            subqueries_text=self.llm(subquery_prompt_template.format(query=query)).strip()
            subqueries=[q.strip() for q in subqueries_text.split('\n') if q.strip()]
            logger.info(f'生成的子查询:{subqueries}')
            if not subqueries:
                logger.warning("未能生成有效的子查询")
                return []
            all_docs=[]
            for sub_q in subqueries:
                docs=self.vector_store.hybrid_search_with_rerank(sub_q,k=conf.RETRIEVAL_K,source_filter=source_filter)
                all_docs.extend(docs)
                logger.info(f"子查询 '{sub_q}' 检索到 {len(docs)} 个文档")
            unique_docs_dict={doc.page_content:doc for doc in all_docs}
            unique_docs=list(unique_docs_dict.values())
            logger.info(f"所有子查询共检索到 {len(all_docs)} 个文档, 去重后剩 {len(unique_docs)} 个")
            return unique_docs

        except Exception as e:
            logger.error(f"子查询策略执行失败: {e}")
            return []
    def _retrieve_with_backtracking(self,query,source_filter=None):
        logger.info(f"使用回溯问题策略进行检索(查询:'{query}')")
        backtrack_prompt_template=RAGPrompts.backtracking_prompt()
        try:
            simplified_query=self.llm(backtrack_prompt_template.format(query=query)).strip()
            logger.info(f"生成的回溯问题: '{simplified_query}'")
            return self.vector_store.hybrid_search_with_rerank(
                simplified_query,k=conf.RETRIEVAL_K,source_filter=source_filter
            )
        except Exception as e:
            logger.error(f"回溯问题策略执行失败: {e}")
            return []
    def _retrieve_with_hyde(self,query,source_filter=None):
        logger.info(f'使用HyDE策略进行检索(查询:{query})')
        hyde_prompt_template = RAGPrompts.hyde_prompt()  # 使用 template 后缀区分
        #   调用大语言模型生成假设答案
        try:
            hypo_answer = self.llm(hyde_prompt_template.format(query=query)).strip()
            logger.info(f"HyDE 生成的假设答案: '{hypo_answer}'")
            #   使用假设答案进行检索，并返回检索结果
            #   注意：HyDE 通常只用于生成检索向量，不一定需要 rerank 这一步，但这里复用了
            return self.vector_store.hybrid_search_with_rerank(
                hypo_answer, k=conf.RETRIEVAL_K,source_filter=source_filter  # 使用 K 而非 M
            )
        except Exception as e:
            logger.error(f"HyDE 策略执行失败: {e}")
            return []

    def retrieve_and_merge(self,query,source_filter=None,strategy=None):
        if not strategy:
            strategy=self.strategy_selector.select_strategy(query)
        ranked_sub_chunks=[]
        if strategy=='回溯问题检索':
            ranked_sub_chunks=self._retrieve_with_backtracking(query,source_filter=source_filter)
        elif strategy=='子查询检索':
            ranked_sub_chunks=self._retrieve_with_subqueries(query,source_filter=source_filter)
        elif strategy=='假设问题检索':
            ranked_sub_chunks = self._retrieve_with_hyde(query)
        else:
            logger.info(f"使用直接检索策略 (查询: '{query}')")
            ranked_sub_chunks=self.vector_store.hybrid_search_with_rerank(
                query=query,
                k=conf.RETRIEVAL_K,
                source_filter=source_filter,
            )
        logger.info(f"策略 '{strategy}' 检索到 {len(ranked_sub_chunks)} 个候选文档 (可能已是父文档)")
        final_context_docs=ranked_sub_chunks[:conf.CANDIDATE_M]
        logger.info(f"最终选取 {len(final_context_docs)} 个文档作为上下文")
        return final_context_docs

    def generate_answer(self, query, source_filter=None, history=None):
        #   记录查询开始时间
        start_time = time.time()
        logger.info(f"开始处理查询: '{query}', 学科过滤: {source_filter}")
        # 验证历史
        if history is not None and not isinstance(history, list):
            logger.warning(f'无效的历史格式：{type(history)},忽略历史')
            history = []
        elif history:
            history = history[-5:] # 严格只取出最近5轮对话
        # 构造历史的上下文：
        history_context = ''
        if history:
            history_context ="\n".join([f"Q:{h['question']}\nA:{h['answer']}" for h in history])
            logger.info(f'使用对话历史：{history_context[:50]}')

        #   判断查询类型
        query_category = self.query_classifier.predict_category(query)
        logger.info(f"查询分类结果：{query_category} (查询: '{query}')")
        #   如果查询属于“通用知识”类别，则直接使用 LLM 回答
        if query_category == "通用知识":
            logger.info("查询为通用知识，直接调用 LLM")
            context = ''
        else:
            logger.info("查询为专业咨询，执行 RAG 流程")
            #   选择检索策略
            strategy = self.strategy_selector.select_strategy(query)
            context_docs = self.retrieve_and_merge(query, source_filter=source_filter, strategy=strategy)
            if context_docs:
                context = "\n\n".join([doc.page_content for doc in context_docs])  # 使用换行符分隔文档
                logger.info(f"构建上下文完成，包含 {len(context_docs)} 个文档块")
                # logger.debug(f"上下文内容:\n{context[:500]}...") # Debug 日志可以打印部分上下文
            else:
                context = ""
                logger.info("未检索到相关文档，上下文为空")

        prompt_input = self.rag_prompt.format(context=context,
                                              question=query,
                                              history=history_context,
                                              phone=conf.CUSTOMER_SERVICE_PHONE)
        try:
            # 使用大模型获得输出结果：
            for token in self.llm(prompt_input):
                yield token
            process_time = time.time() - start_time
            logger.info(f'LLM查询处理完成（耗时：{process_time:.2f}s, 查询：{query})')
        except Exception as e:
            logger.error(f'调用LLM失败:{e}')
            yield f'抱歉，处理问题时出错，请你联系人工客服：{conf.CUSTOMER_SERVICE_PHONE}'

if __name__ == '__main__':
    # 示例：完整查询处理
    vector_store = VectorStore()
    llm=StrategySelector().call_dashscope

    query = "AI学科学费是多少？"
    rag_system = RAGSystem(vector_store, llm)
    answer = rag_system.generate_answer(query)
    print(answer)