from db.mysql_client import MYSQLClient
# 导入 Redis 客户端
from cache.redis_client import RedisClient
# 导入 BM25 搜索
from retrieval.bm25_search import BM25Search
# 导入日志
from base.logger import logger
# 导入时间库
import time
class MySQLSystem:
    def __init__(self):
        self.logger=logger
        self.mysql_client=MYSQLClient()
        self.redis_client=RedisClient()
        self.bm25_search=BM25Search(self.redis_client,self.mysql_client)
    def query(self,query):
        start_time = time.time()
        self.logger.info(f'处理查询{query}')
        answer,_ =self.bm25_search.search(query)
        if answer:
            self.logger.info(f'MYSQL答案{answer}')
        else:
            self.logger.info('SQL中未找到答案,需要调用RAG系统')
            answer='SQL未找到答案'
        processing_time = time.time() - start_time
        self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
        # 返回答案
        return answer
def main():
    mysql_system = MySQLSystem()
    try:
        print('\n欢迎使用MySQL问答系统')
        print("输入查询进行问答,输入'exit'退出")
        while True:
            query=input('\n输入查询').strip()
            if query.lower()=='exit':
                logger.info('退出MySQL系统')
                break
            answer=mysql_system.query(query)
            print(f'\n答案:{answer}')

    except Exception as e:
        logger.error(f'系统错误{e}')
    finally:
        mysql_system.mysql_client.close()
if __name__ == '__main__':
    main()