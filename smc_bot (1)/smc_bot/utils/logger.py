"""utils/logger.py"""
import logging
import os
from datetime import datetime


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(getattr(logging, level, logging.INFO))

    fmt = logging.Formatter("%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
                             datefmt="%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.addHandler(ch)

    os.makedirs("logs", exist_ok=True)
    fh = logging.FileHandler(
        f"logs/bot_{datetime.now():%Y%m%d}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return log
