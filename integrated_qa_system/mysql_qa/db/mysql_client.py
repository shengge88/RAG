import pymysql
import pandas as pd
import os,sys
current_path=os.path.abspath(__file__)
model_path=os.path.abspath(os.path.dirname(current_path))
root_path=os.path.dirname(os.path.dirname(model_path))
sys.path.insert(0,root_path)
from base.config import Config
from base.logger import logger
class MYSQLClient:
    def __init__(self):
        self.logger=logger
        cfg = Config()
        try:
            self.connection=pymysql.connect(
                host=cfg.MYSQL_HOST,
                user=cfg.MYSQL_USER,
                password=cfg.MYSQL_PASSWORD,
                database=cfg.MYSQL_DATABASE
            )
            self.cursor=self.connection.cursor()
            self.logger.info('MYSQL连接成功')
        except Exception as e:
            self.logger.error(f'MYSQL连接失败{e}')
            raise
    def create_table(self):
        create_table_query='''
            CREATE TABLE IF NOT EXISTS jpkb (
                id INT AUTO_INCREMENT PRIMARY KEY,
                subject_name VARCHAR(20),
                question VARCHAR(1000),
                answer VARCHAR(1000)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        '''
        try:
            self.cursor.execute(create_table_query)
            self.connection.commit()
            self.logger.info('表创建成功')
        except Exception as e:
            self.logger.error(f'表创建失败{e}')
            raise
    def insert_data(self,csv_path):
        try:
            data=pd.read_csv(csv_path)
            for _,row in data.iterrows():
                insert_query="INSERT INTO jpkb (subject_name, question, answer) VALUES (%s, %s, %s)"
                self.cursor.execute(insert_query,(row['学科名称'],row['问题'],row['答案']))
            self.connection.commit()
            self.logger.info('数据插入成功')
        except Exception as e:
            self.logger.error(f'查询失败:{e}')
            self.connection.rollback()
            raise
    def fetch_question(self):
        try:
            self.cursor.execute("SELECT question FROM jpkb")
            result=self.cursor.fetchall()
            self.logger.info('成功获取问题')
            return result
        except Exception as e:
            self.logger.error(f'查询失败{e}')
            return []
    def fetch_answer(self,question):
        try:
            self.cursor.execute("SELECT answer FROM jpkb WHERE question=%s", (question,))
            result=self.cursor.fetchone()
            print(f'result-->{result}')
            print(f'result[0]-->{result[0]}')
            return result[0] if result else None
        except Exception as e:
            self.logger.error(f'获取答案失败:{e}')
            return None
    def close(self):
        try:
            self.connection.close()
            self.logger.info('MySQL连接已关闭')
        except Exception as e:
            self.logger.error(f'关闭连接失败:{e}')
if __name__ == '__main__':

    client=MYSQLClient()
    #client.create_table()
    #client.insert_data('../data/JP学科知识问答.csv')
    #client.fetch_question()
    client.fetch_answer(question='用上下文管理器实现函数运行时间的计算?')
