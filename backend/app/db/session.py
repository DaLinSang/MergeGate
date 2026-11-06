"""
数据库连接模块
===============

职责：
    1. 定义 ORM 基类 Base（所有表模型的父类）
    2. 创建连接池 engine
    3. 创建会话工厂 SessionLocal
    4. 提供 FastAPI 依赖 get_db()，为每个请求分配/回收会话

在系统中的位置：
    数据层（db/），是"最底层"——只被上层 import，自己不 import 任何业务代码。

三个核心概念（Java 开发者对照）：
    engine       ≈ DataSource（连接池）
    SessionLocal ≈ SqlSessionFactory（会话工厂）
    Base         ≈ 所有 @Entity 的共同父类
    Session      ≈ SqlSession / EntityManager（工作单元）

为什么 Session 就是"工作单元"：
    一个 Session 内部维护着"已加载对象"的身份映射和"待提交的变更"。
    db.add(obj) / obj.field = x 只是把变更记在 Session 里，
    只有 db.commit() 才真正写库，db.rollback() 可以全部撤销。
"""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings


class Base(DeclarativeBase):
    """所有表模型的基类。

    继承它之后，SQLAlchemy 会自动把子类注册到 Base.metadata 里，
    于是 main.py 里的 Base.metadata.create_all() 才能一次性建出全部 32 张表。

    等价于 Java 里所有 @Entity 类共同实现的某个标记接口，
    只不过这里 SQLAlchemy 用它来做"表注册表"。
    """

    pass


# ────────────────────────────────────────────────────────────────
# 连接池（全局唯一）
# ────────────────────────────────────────────────────────────────
# create_engine 不会立刻连接数据库，而是创建一个"连接池管理者"。
# pool_pre_ping=True：每次从池里取连接前先 ping 一下，
#   避免拿到已被数据库/网络断开的死连接（长时间空闲后很常见）。
engine = create_engine(settings.database_url, pool_pre_ping=True)

# ────────────────────────────────────────────────────────────────
# 会话工厂
# ────────────────────────────────────────────────────────────────
# autoflush=False：执行查询前不自动把待写入的变更刷到数据库。
#   这样"先 add 再 query"不会意外触发 INSERT，行为更可预测。
# autocommit=False：必须显式 commit() 才生效（现代 SQLAlchemy 的默认推荐）。
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db() -> Generator[Session, None, None]:
    """FastAPI 依赖：为每个请求提供一个数据库会话，请求结束后自动关闭。

    用法：
        @router.get("/xxx")
        async def handler(db: Session = Depends(get_db)):
            ...

    yield 的作用（关键）：
        这不是普通 return。yield 把 db 交出去之后，函数【暂停】在这里；
        等请求处理完毕，FastAPI 会回来继续执行 finally 里的 db.close()。

        等价于 Java 的 try-with-resources / Spring 的 @Transactional 代理，
        但这里是显式写出来的。

    为什么要每个请求一个 Session 而不是全局一个：
        1. Session 不是线程安全的，全局共享会串数据
        2. 请求结束自动 close，避免连接泄漏
        3. 事务边界清晰：一个请求 = 一个事务
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        # 无论请求成功、抛异常还是被取消，都会执行到这里
        db.close()
