import os
import sys

model_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(model_path)
if model_path not in sys.path:
    sys.path.insert(0, model_path)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from base.config import Config
from base.logger import logger

__all__ = ['Config', 'logger', 'model_path', 'root_path']
