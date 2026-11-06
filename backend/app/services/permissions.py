"""
权限与审计
=========

【业务定位】
    极简的角色权限系统 + 审计日志写入。
    课程项目级实现：没有登录态、没有密码，靠"取第一个用户"确定当前身份。

【数据流向】
    上游（谁调它）：
        api/routes/*.py → require_permission("draft:approve") 作为 FastAPI 依赖
        api/routes/action_drafts.py → write_audit_log() 记录敏感操作
    下游（它写谁）：
        users 表       ← ensure_demo_user()
        audit_logs 表  ← write_audit_log()

【核心规则】
    1. 角色是"包含关系"：owner ⊃ maintainer ⊃ developer ⊃ viewer。
       每个角色直接列出自己的权限集合（不用继承计算，简单直观）。
    2. 未知角色一律按 viewer 处理（最小权限原则）。
    3. 敏感操作必须写审计日志 —— 出了问题要能查"谁在什么时候干了什么"。

【本文件函数地图】
    ensure_demo_user()      取/建当前用户
    has_permission()        判断某用户有没有某权限
    require_permission()    ★ 生成 FastAPI 依赖（路由上直接用）
    write_audit_log()       写审计日志
"""

from collections.abc import Callable

from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.models import AuditLog, User
from app.db.session import get_db


# 角色 → 权限集合。
#
# 【为什么每个角色都完整列一遍，而不是"继承上一个角色再加"】
#   继承需要额外代码计算，而且排查"为什么这个角色有这个权限"时要跨好几层。
#   直接列出来，一眼能看全，改动也不会影响其他角色。
#
# 【权限命名约定：资源:动作】
#   repo:read        读仓库数据
#   repo:sync        触发同步（会消耗 GitHub 额度）
#   agent:run        跑 Agent（会消耗 LLM 费用）
#   draft:create     生成动作草稿（AI 想改 GitHub）
#   draft:approve    ★ 批准草稿（真正写 GitHub）—— 这是最危险的权限
#   eval:read        看评测结果
#   settings:manage  改系统配置
ROLE_PERMISSIONS: dict[str, set[str]] = {
    "owner": {"repo:read", "repo:sync", "agent:run", "draft:create", "draft:approve", "eval:read", "settings:manage"},
    "maintainer": {"repo:read", "repo:sync", "agent:run", "draft:create", "draft:approve", "eval:read"},
    "developer": {"repo:read", "agent:run", "draft:create", "eval:read"},
    "viewer": {"repo:read", "eval:read"},
}


def ensure_demo_user(db: Session) -> User:
    """取当前用户（没有就创建一个 owner）。

    【调用链】被 require_permission() 的依赖函数调用
    【下游】users 表

    【数据流向】
        入：db
        出：User（保证有 role）

    【三个分支】
        ① 库里没有任何用户 → 建一个 demo-owner
        ② 有用户但 role 为空（老数据）→ 补成 owner
        ③ 正常 → 直接返回

    【为什么取 created_at 最早的那个】
        单用户模式下，第一个创建的就是"主人"。
        多用户场景这里要换成"从 token/session 解析"—— 本项目不做。
    """
    user = db.query(User).order_by(User.created_at.asc()).first()
    if user is None:
        user = User(name="demo-owner", role="owner")
        db.add(user)
        db.commit()
        db.refresh(user)
    # 兼容老数据：role 字段是后加的，历史行可能是 NULL
    if not getattr(user, "role", None):
        user.role = "owner"
        db.commit()
        db.refresh(user)
    return user


def has_permission(user: User, permission: str) -> bool:
    """判断用户是否有某权限。

    【调用链】被 require_permission() 的依赖函数调用

    【为什么要 .lower() 和默认 "viewer"】
        role 可能来自数据库（大小写不规范）或为 None。
        统一小写 + 默认 viewer，保证"异常情况走最小权限"而不是报错放行。
    """
    return permission in ROLE_PERMISSIONS.get((user.role or "viewer").lower(), set())


def require_permission(permission: str) -> Callable:
    """生成一个 FastAPI 依赖，用于在路由上做权限校验。

    【调用链】追这个函数会进入：
        1. ensure_demo_user()    → 取当前用户
        2. has_permission()      → 判断权限
        3. HTTPException(403)    → 没有权限时抛
    【被谁调用】api/routes/*.py 的路由装饰器参数，例如：
        @router.post("/drafts/{id}/approve")
        async def approve(user: User = Depends(require_permission("draft:approve"))): ...

    【为什么返回一个函数（工厂模式）】
        FastAPI 的 Depends 需要一个可调用对象。
        权限名要在"路由定义时"就绑定，所以用闭包把 permission 捕获进去。

    【为什么用 403 而不是 401】
        401 是"没登录"，403 是"登录了但没权限"。
        本项目没有登录态，语义上更接近 403。
    """
    def dependency(db: Session = Depends(get_db)) -> User:
        user = ensure_demo_user(db)
        if not has_permission(user, permission):
            raise HTTPException(status_code=403, detail=f"需要权限：{permission}")
        return user

    return dependency


def write_audit_log(
    db: Session,
    *,
    user: User | None,
    repo_id=None,
    action: str,
    target_type: str | None = None,
    target_id: str | None = None,
    status: str = "success",
    request_json: dict | None = None,
    result_json: dict | None = None,
) -> AuditLog:
    """写一条审计日志。

    【调用链】被 api/routes/action_drafts.py 等敏感接口调用
    【下游】audit_logs 表（只 flush，由外层事务提交）

    【数据流向】
        入：user（可空，系统操作时为 None）、action（如 "draft.confirm"）、
            target_type / target_id（操作对象）、status、请求与结果快照
        出：AuditLog

    【为什么只 flush 不 commit】
        审计日志必须和业务操作在【同一个事务】里：
        业务回滚了，日志也该一起回滚 —— 否则会出现"日志说执行了，实际没执行"。
    """
    row = AuditLog(
        user_id=user.id if user else None,
        repo_id=repo_id,
        action=action,
        target_type=target_type,
        target_id=target_id,
        status=status,
        request_json=request_json or {},
        result_json=result_json or {},
    )
    db.add(row)
    db.flush()
    return row
