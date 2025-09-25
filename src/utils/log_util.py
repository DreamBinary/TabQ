# -*- coding:utf-8 -*-
# @FileName : log_util.py
# @Time : 2025/1/30 16:36
# @Author : fiv
import logging
import sys
import time
from functools import wraps


def init_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s-%(levelname)s-%(filename)s:%(lineno)d -->> %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    logging.getLogger("PIL").setLevel(logging.WARNING)
    return logging.getLogger()


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        cost_time = end_time - start_time
        start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
        end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
        cost_time_str = time.strftime("%H:%M:%S", time.gmtime(cost_time))
        print(
            f"-->> Function `{func.__name__}` cost time {cost_time_str} from {start_time_str} to {end_time_str}"
        )
        return result

    return wrapper


def timer_async(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))

        result = await func(*args, **kwargs)

        end_time = time.time()
        end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
        cost_time = end_time - start_time
        cost_time_str = time.strftime("%H:%M:%S", time.gmtime(cost_time))

        print(
            f"-->> Function `{func.__name__}` cost time {cost_time_str} from {start_time_str} to {end_time_str}"
        )
        return result

    return wrapper
