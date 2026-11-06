"""
定时同步任务（Webhook 的兜底）
============================

【业务定位】
    后台常驻一个异步循环，每隔 N 秒把所有已连接仓库同步一遍，
    作为 Webhook 之外的"兜底刷新"。

【数据流向】
    上游（谁调它）：
        app/main.py 的 startup 事件
            → asyncio.create_task(periodic_repo_sync_loop())
        app/main.py 的 shutdown 事件
            → task.cancel()
    下游（它调谁）：
        services/repos_sync.sync_repository_by_id()
            └ 内部会自己开 Session、查仓库、调 sync_repository()、关 Session
        repositories 表（只查 ID）

【核心规则】
    1. **单个仓库失败不能中断循环** —— 否则一个 token 过期的仓库会让所有仓库都停摆。
    2. **查 ID 和同步用不同的 Session** —— 同步可能几分钟，长事务会锁表。
    3. **间隔有 30 秒下限** —— 防止配置写错把 GitHub 额度刷爆。

为什么需要它（既然已经有 Webhook 了）：
    Webhook 可能丢（网络抖动、GitHub 重试耗尽、内网无法被外网访问）。
    定时任务是兜底：即使 Webhook 全丢了，数据最多延迟一个周期也会更新。

与其它三个同步入口的关系：
    连接仓库 ──┐
    手动刷新 ──┤
    定时任务 ──┼──> 都最终调用同一个 sync_repository()
    Webhook ───┘
    ——"入口可以增加，同步规则必须收口"。

【本文件函数地图】
    periodic_repo_sync_loop()  ★ 唯一函数：无限循环，三步一轮
        ① 只查仓库 ID（短 Session）
        ② 逐个仓库同步（每个用自己的 Session）
        ③ 睡到下一轮
"""

import asyncio
import logging

from app.core.config import settings
from app.db.models import Repository
from app.db.session import SessionLocal
from app.schemas.repos import RepoSyncRequest
from app.services.repos_sync import sync_repository_by_id

logger = logging.getLogger(__name__)


async def periodic_repo_sync_loop() -> None:
    """无限循环：周期性同步所有已连接的 GitHub 仓库。

    【调用链】追这个函数会进入：
        1. SessionLocal()                      → 开短 Session 查仓库 ID
        2. sync_repository_by_id()             → 逐个仓库同步（repos_sync）
        3. asyncio.sleep()                     → 睡到下一轮
    【被谁调用】app/main.py 的 startup 事件（asyncio.create_task）

    【数据流向】
        入：settings.auto_sync_enabled / auto_sync_interval_seconds / auto_sync_limit
        出：无返回值（永不正常返回）—— 结果体现在数据库里

    这个函数永远不会正常返回——它只在应用关闭时被 cancel()。

    生命周期：
        main.py startup → asyncio.create_task(本函数)
        main.py shutdown → task.cancel()

    【为什么永不返回】
        它是"常驻后台任务"。返回就意味着循环结束了，同步也就停了。
        唯一退出的方式是：
            ① 被 cancel（应用关闭）
            ② settings.auto_sync_enabled 变成 False（下一轮 while 判断时自然退出）
    """

    # 启动后先等 5 秒，让应用把其它初始化（建表、Milvus 探测）跑完，
    # 避免一启动就抢数据库连接。
    await asyncio.sleep(5)

    # while 条件用配置项，是为了支持"运行时关闭"：
    # 只要把 auto_sync_enabled 改成 False，下一轮循环就会自然退出。
    while settings.auto_sync_enabled:

        # ── 第一步：只查出所有仓库的 ID ──
        # 为什么要单独开一个 Session、查完立刻关闭：
        #   下面的同步可能耗时几分钟。如果拿着这个 Session 一直不关，
        #   会形成长事务，锁住表，阻塞其他请求。
        #   所以这里"只取 ID"，取完马上释放连接。
        db = SessionLocal()
        try:
            repos = db.query(Repository).filter(Repository.provider.in_(["github", "github_compatible"])).all()
            repo_ids = [repo.id for repo in repos]
        finally:
            db.close()

        # ── 第二步：逐个仓库同步，每个仓库用自己的 Session ──
        for repo_id in repo_ids:
            try:
                # sync_repository_by_id 内部会新建 Session、查 Repository、
                # 调核心同步函数、最后关闭 Session。
                await sync_repository_by_id(repo_id, RepoSyncRequest(limit=settings.auto_sync_limit))
            except Exception as exc:
                # 关键设计：单个仓库失败只记日志，不中断循环。
                # 否则一个 token 过期的仓库会让所有其他仓库都停止同步。
                logger.warning("Periodic repo sync failed for %s: %s", repo_id, exc)

        # ── 第三步：睡到下一轮 ──
        # max(..., 30) 是保护：即使有人把间隔配成 1 秒，
        # 也不会疯狂刷 GitHub API（无 token 时额度只有 60 次/小时）。
        await asyncio.sleep(max(settings.auto_sync_interval_seconds, 30))
