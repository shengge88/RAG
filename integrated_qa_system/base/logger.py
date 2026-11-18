import logging
import os

try:
    from base.config import Config
except ImportError:
    from config import Config

current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
root_path = os.path.dirname(current_dir)
log_path = os.path.join(root_path, 'logs', 'app.log')


def setup_logging(log_file=None):
    if log_file is None:
        log_file = Config().LOG_FILE
    if not os.path.isabs(log_file):
        log_file = os.path.join(root_path, log_file)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger('EduRAG')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger


logger = setup_logging()
