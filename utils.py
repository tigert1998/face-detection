import sys
import logging
from typing import Optional


def get_logger(name, filename: Optional[str]):
    root = logging.getLogger(name)
    root.setLevel(logging.INFO)
    if filename is None:
        handler = logging.StreamHandler(sys.stdout)
    else:
        handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(filename)s:%(lineno)s] [%(levelname)s] %(message)s"
    )
    handler.setFormatter(formatter)
    root.handlers = [handler]
    return root
