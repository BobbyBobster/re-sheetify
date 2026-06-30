import logging
import sys
from pythonjsonlogger.json import JsonFormatter

from sheets.params import *


class GCPJsonFormatter(JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)

        log_record["severity"] = log_record.pop("levelname", "INFO")

        log_record["logging.googleapis.com/sourceLocation"] = {
            "file": record.pathname,
            "line": record.lineno,
            "function": record.funcName,
        }


def setup_logging():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)

    if TRAINING_ENV == "cloud":
        formatter = GCPJsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s"
        )
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
        )

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
