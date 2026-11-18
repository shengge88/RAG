import os,sys
current_path=os.path.abspath(__file__)
model_path=os.path.abspath(os.path.dirname(current_path))
root_path=os.path.dirname(os.path.dirname(model_path))
sys.path.insert(0,root_path)
from base.logger import logger
import jieba

def preprocess_text(text):
    logger.info('开始处理文本')
    try:
        return jieba.lcut(text.lower())
    except Exception as e:
        logger.error(f'文本处理失败:{e}')
        return []