import inspect
import logging
import sys
from functools import wraps

from loguru import logger


class InterceptHandler(logging.Handler):
    """
    標準のloggingメッセージをloguruに流すためのクラス
    以下のコードをそのまま流用しました。
    https://github.com/Delgan/loguru?tab=readme-ov-file#entirely-compatible-with-standard-logging
    """

    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists.
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def configure_loguru(*, is_verbose: bool) -> None:
    """
    loguruの設定を行います。

    Args:
        is_verbose: 詳細なログを出力するか
    """

    logger.remove()
    logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO, force=True)

    if is_verbose:
        # srcモジュール用: DEBUG以上
        level_per_module = {
            "": "INFO",
            "src": "DEBUG",
            "__main__": "DEBUG",
        }
    else:
        level_per_module = {
            "": "INFO",
        }

    logger.add(sys.stderr, diagnose=False, filter=level_per_module)  # type: ignore[arg-type]
    logger.add(".log/annofab-coco-tools.log", rotation="1 day", diagnose=False, filter=level_per_module)  # type: ignore[arg-type]


def log_exception():  # noqa: ANN201
    """
    例外発生時に指定したメッセージをログに出力するデコレータ。
    例外は再スローされる。
    """

    def decorator(func):  # noqa: ANN001, ANN202
        @wraps(func)
        def wrapper(*args, **kwargs):  # noqa: ANN202, ANN003, ANN002
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(e)
                raise

        return wrapper

    return decorator
