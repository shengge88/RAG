import os.path
import sys
from langchain_ollama import ChatOllama
r_path=os.path.dirname(os.path.abspath(__file__))
root_path=os.path.dirname(r_path)
sys.path.insert(0,r_path)
sys.path.insert(0,root_path)
from mysql_qa import MYSQLClient, RedisClient, BM25Search
# 导入 RAG 系统组件，用于知识库检索和答案生成
from rag_qa import VectorStore, RAGSystem
# 导入配置和日志工具，用于系统配置和日志记录
from base import logger, Config
# 导入 OpenAI 客户端，用于调用 DashScope API
from openai import OpenAI
# 导入时间库，用于记录处理时间
import time
class IntegratedQASystem:
    def __init__(self):
        # 初始化日志工具，用于记录系统运行信息
        self.logger = logger
        # 初始化配置对象，加载系统参数
        self.config = Config()
        # 初始化 MySQL 客户端，用于数据库操作
        self.mysql_client = MYSQLClient()
        # 初始化 Redis 客户端，用于缓存管理
        self.redis_client = RedisClient()
        # 初始化 BM25 搜索模块，结合 MySQL 和 Redis
        self.bm25_search = BM25Search(self.redis_client, self.mysql_client)
        try:
            # 初始化 OpenAI 客户端，连接 DashScope API
            self.client = OpenAI(
                api_key=Config().DASHSCOPE_API_KEY,
                base_url=Config().DASHSCOPE_BASE_URL
            )
        except Exception as e:
            # 记录 OpenAI 初始化失败的错误日志
            self.logger.error(f"OpenAI 客户端初始化失败: {e}")
            # 抛出异常，终止初始化
            raise
        # 初始化向量存储，用于 RAG 系统的知识库管理
        self.vector_store = VectorStore()
        # 初始化 RAG 系统，传入向量存储和 DashScope API 调用函数
        self.rag_system = RAGSystem(self.vector_store, self.call_dashscope)

    def call_dashscope(self, prompt):
        # 定义调用 DashScope API 的方法，生成自然语言答案
        try:
            # 创建聊天完成请求，调用 DashScope API
            completion = self.client.chat.completions.create(
                model=self.config.LLM_MODEL,  # 使用配置中的语言模型
                messages=[
                    {"role": "system", "content": "你是一个有用的助手。"},  # 系统提示
                    {"role": "user", "content": prompt},  # 用户输入的提示
                ]
            )
            # 检查响应是否有效，返回答案内容
            return completion.choices[0].message.content if completion.choices else "错误：无效的 LLM 响应"
        except Exception as e:
            # 记录 API 调用失败的错误日志
            self.logger.error(f"LLM 调用失败: {e}")
            # 返回错误信息，便于调试
            return f"错误：LLM 调用失败 - {e}"
    def query(self,query,source_filter=None):
        start_time = time.time()
        self.logger.info(f'处理查询{query}')
        answer,need_rag=self.bm25_search.query(query,threshold=0.85)
        if answer:
            self.logger.info(f'MySQL答案:{answer}')
            processing_time = time.time() - start_time
            self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
            return answer
        elif need_rag:
            self.logger.info('无可靠MySQL答案,回退到RAG')
            answer=self.rag_system.generate_answer(query,source_filter=source_filter)
            self.logger.info(f"RAG 答案: {answer}")
            processing_time = time.time() - start_time
            self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
            return answer
        else:
            # 如果无答案，记录信息到日志
            self.logger.info("未找到答案")
            # 计算处理时间
            processing_time = time.time() - start_time
            # 记录处理时间到日志
            self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
            # 返回默认答案
            return "未找到答案"
def main():
    # 定义主函数，提供命令行交互界面
    qa_system = IntegratedQASystem()  # 初始化问答系统
    try:
        # 打印欢迎信息
        print("\n欢迎使用集成问答系统！")
        # 打印支持的学科类别
        print(f"支持的来源: {qa_system.config.VALID_SOURCES}")
        # 提示用户输入查询或退出
        print("输入查询进行问答，输入 'exit' 退出。")
        while True:
            # 获取用户输入的查询
            query = input("\n输入查询: ").strip()
            if query.lower() == "exit":
                # 如果用户输入 exit，记录退出日志
                logger.info("退出系统")
                # 打印退出信息
                print("再见！")
                # 退出循环
                break
            # 获取用户输入的学科过滤
            source_filter = input(f"输入来源过滤 ({'/'.join(qa_system.config.VALID_SOURCES)}) (按 Enter 跳过): ").strip()
            if source_filter and source_filter not in qa_system.config.VALID_SOURCES:
                # 如果学科过滤无效，记录警告日志
                logger.warning(f"无效来源 '{source_filter}'，忽略过滤")
                # 打印无效信息，忽略过滤
                print(f"无效来源 '{source_filter}'，继续无过滤。")
                source_filter = None
            # 执行查询，获取答案
            answer = qa_system.query(query, source_filter)
            # 打印答案
            print(f"\n答案: {answer}")
    except Exception as e:
        # 记录系统错误日志
        logger.error(f"系统错误: {e}")
        # 打印错误信息
        print(f"发生错误: {e}")
    finally:
        # 无论是否发生错误，关闭 MySQL 连接
        qa_system.mysql_client.close()

if __name__ == "__main__":
    # 如果脚本作为主程序运行，调用 main 函数
    main()



