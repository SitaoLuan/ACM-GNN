from abc import ABC, abstractmethod
from datetime import datetime
import logging
from pathlib import Path
from sys import stdout


class BaseLogger(ABC):
    def __init__(self, name, log_dir="./logs/", log_level="INFO"):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_name = log_dir + datetime.now().strftime(f"{name}file_%Y_%m_%d_%H_%M.log")
        self.logger = logging.getLogger(log_name)
        self.stream_handler = logging.StreamHandler(stdout)
        self.file_handler = logging.FileHandler(log_name)
        self._set_logger(log_level)

    def _set_logger(self, log_level):
        log_format = logging.Formatter(
            "%(asctime)s: %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        log_level = getattr(logging, log_level)
        self.logger.setLevel(log_level)

        self.stream_handler.setFormatter(log_format)
        self.logger.addHandler(self.stream_handler)

        self.file_handler.setFormatter(log_format)
        self.logger.addHandler(self.file_handler)

    def log_init(self, msg):
        msg = f"Init - {msg}"
        self.logger.info(msg)

    @abstractmethod
    def log_best_result(self, **kwargs):
        pass

    @abstractmethod
    def log_run(self, **kwargs):
        pass

    @abstractmethod
    def log_param_tune(self, **kwargs):
        pass

    def log_time(self, msg):
        msg = f"Total time elapsed - {msg}"
        self.logger.info(msg)




