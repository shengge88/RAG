import os.path
import sys
r_path=os.path.dirname(os.path.abspath(__file__))
root_path=os.path.dirname(r_path)
core_path=os.path.join(root_path,'core')
sys.path.insert(0,core_path)
sys.path.insert(0,r_path)
sys.path.insert(0,root_path)
# 导入 MySQL 和 Redis 客户端，管理数据库和缓存
from mysql_qa import MYSQLClient, RedisClient, BM25Search
# 导入 RAG 系统组件，用于知识库检索和答案生成
from rag_qa import VectorStore, RAGSystem
# 导入配置和日志工具，用于系统配置和日志记录
from base import logger, Config
# 导入 OpenAI 客户端，用于调用 DashScope API
from openai import OpenAI
# 导入时间库，用于记录处理时间
import time
# 导入 UUID 库，生成唯一会话 ID
import uuid
# 导入 pymysql 错误处理，用于数据库操作的异常捕获
import pymysql
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
            self.client = OpenAI(api_key=self.config.DASHSCOPE_API_KEY,
                                 base_url=self.config.DASHSCOPE_BASE_URL)
        except Exception as e:
            # 记录 OpenAI 初始化失败的错误日志
            self.logger.error(f"OpenAI 客户端初始化失败: {e}")
            # 抛出异常，终止初始化
            raise
        # 初始化向量存储，用于 RAG 系统的知识库管理
        self.vector_store = VectorStore()
        # 初始化 RAG 系统，传入向量存储和 DashScope API 调用函数
        self.rag_system = RAGSystem(self.vector_store, self.call_dashscope)
        # 初始化对话历史表，用于存储会话记录
        self.init_conversation_table()
    def init_conversation_table(self):
        try:
            self.mysql_client.cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    session_id VARCHAR(36) NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    INDEX idx_session_id (session_id)
                )
            """)
            self.mysql_client.connection.commit()
            self.logger.info('对话历史表初始化成功')
        except Exception as e:
            self.logger.error(f'初始化对话表历史失败{e}')
            raise
    def call_dashscope(self,prompt):
        try:
            completion=self.client.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=[
                    {
                        'role':'system','content':'你是一个有用的助手'
                    },{
                        'role':'user','content':prompt
                    }
                ],timeout=30,
                stream=True
            )
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    content=chunk.choices[0].delta.content
                    yield content
        except Exception as e:
            self.logger.error(f'LLM调用失败:{e}')
            return f'错误:LLM调用失败{e}'
    def _fetch_recent_history(self,session_id: str) -> list:
        try:
            self.mysql_client.cursor.execute("""
                SELECT question, answer
                FROM conversations
                WHERE session_id = %s
                ORDER BY timestamp DESC
                LIMIT %s
            """,(session_id,5))
            history=[{'question':row[0],'answer':row[1]} for row in  self.mysql_client.cursor.fetchall()]
            return history[::-1]
        except pymysql.MySQLError as e:
            # 记录查询失败的错误日志
            self.logger.error(f"获取对话历史失败: {e}")
            # 返回空列表
            return []
    def get_session_history(self,session_id:str) -> list:
        return self._fetch_recent_history(session_id)
    def update_session_history(self,session_id:str,question:str,answer:str) -> list:
        try:
            self.mysql_client.cursor.execute("""
            INSERT INTO conversations (session_id, question, answer, timestamp)
                VALUES (%s, %s, %s, NOW())
            """,(session_id,question,answer))
            history=self._fetch_recent_history(session_id)
            self.mysql_client.cursor.execute("""DELETE FROM conversations
             WHERE session_id = %s
            AND id NOT IN (SELECT id
            FROM (SELECT id
            FROM conversations
            WHERE session_id = %s
            ORDER BY timestamp DESC
            LIMIT %s) AS sub)
             """, (session_id, session_id, 5))
            self.mysql_client.connection.commit()
            self.logger.info(f'会话{session_id}历史更新成功')
            return history
        except Exception as e:
            self.logger.error(f"更新会话历史意外错误: {e}")
            self.mysql_client.connection.rollback()
            raise

    def query(self,query,source_filter=None,session_id=None):
        start_time = time.time()
        self.logger.info(f"处理查询'{query}'(会话ID{session_id})")
        history= self.get_session_history(session_id) if session_id else []
        answer,need_rag=self.bm25_search.search(query,threshold=0.85)
        if answer:
            self.logger.info(f'MySQL答案:{answer}')
            if session_id:
                self.update_session_history(session_id,query,answer)
            processing_time = time.time() - start_time
            self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
            # 一次性返回答案，标记为完整
            yield answer, True
        elif need_rag:
            logger.info('无可靠MySQL答案,回退到RAG')
            collected_answer=''
            for token in self.rag_system.generate_answer(query,source_filter=source_filter,history=history):
                collected_answer+=token
                yield token,False
            if session_id:
                self.update_session_history(session_id,query,collected_answer)
                # 计算处理时间
                processing_time = time.time() - start_time
                # 记录处理时间到日志
                self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
                # 返回空字符串，标记流结束
                yield "", True
        else:
            # 如果无答案，记录信息到日志
            self.logger.info("未找到答案")
            # 计算处理时间
            processing_time = time.time() - start_time
            # 记录处理时间到日志
            self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
            # 一次性返回默认答案，标记为完整
            yield "未找到答案", True
def main():
    qa_system = IntegratedQASystem()
    session_id = str(uuid.uuid4())
    print("\n欢迎使用集成问答系统！")
    print(f"会话ID: {session_id}")
    print(f"支持的学科类别：{qa_system.config.VALID_SOURCES}")
    print("输入查询进行问答，输入 'exit' 退出。")
    try:
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
            source_filter = input(f"请输入学科类别 ({'/'.join(qa_system.config.VALID_SOURCES)}) (直接回车默认不过滤): ").strip()
            if source_filter and source_filter not in qa_system.config.VALID_SOURCES:
                # 如果学科过滤无效，记录警告日志
                logger.warning(f"无效的学科类别 '{source_filter}'，将不过滤")
                # 设置为空，忽略过滤
                source_filter = None
            # 打印答案提示
            print("\n答案: ", end="", flush=True)
            # 初始化累积答案的字符串
            answer = ""
            # 迭代 query 方法的生成器
            for token, is_complete in qa_system.query(query, source_filter=source_filter, session_id=session_id):
                if token:
                    # 仅当 token 非空时打印
                    print(token, end="", flush=True)
                    # 累积答案
                    answer += token
                if is_complete:
                    print()
                    break
            # 打印对话历史
            history = qa_system.get_session_history(session_id)
            print("\n最近对话历史:")
            for idx, entry in enumerate(history, 1):
                # 按顺序打印历史记录
                print(f"{idx}. 问: {entry['question']}\n   答: {entry['answer']}")
    except Exception as e:
        # 记录系统错误日志
        logger.error(f"系统错误: {e}")
        # 打印错误信息
        print(f"发生错误: {e}")
    finally:
        # 关闭 MySQL 连接
        qa_system.mysql_client.close()

if __name__ == "__main__":
    # 如果脚本作为主程序运行，调用 main 函数
    main()
