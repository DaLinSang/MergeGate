"""
日志配置模块
=============

职责：
    统一配置 Python 标准库 logging 的输出格式，让所有模块的日志长得一样。

在系统中的位置：
    基础设施层（core/）。由 app/main.py 在创建应用时调用一次，全局生效。

为什么需要它：
    如果不统一配置，每个模块用 logging.getLogger(__name__) 拿到的 logger
    会各自继承默认格式（没有时间、没有级别），排查问题时很难定位。
"""

import logging


def configure_logging() -> None:
    """配置根 logger 的输出级别和格式。

    只需要在应用启动时调用一次。之后的 logging.getLogger(__name__)
    都会自动继承这套配置。

    格式说明：
        %(asctime)s   → 时间，如 2026-10-22 17:30:00,123
        %(levelname)s → 级别，如 INFO / WARNING / ERROR
        %(name)s      → logger 名字，通常是模块路径，如 app.services.scheduler
        %(message)s   → 日志正文

    为什么用 INFO 级别：
        开发阶段看 INFO 足够；DEBUG 会刷屏，WARNING 会漏掉关键流程信息。
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
