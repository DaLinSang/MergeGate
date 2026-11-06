"""
Token 加密模块
===============

职责：
    对敏感凭证（主要是 GitHub Personal Access Token）做对称加密和解密。

在系统中的位置：
    基础设施层（core/）。被两处使用：
        1. api/routes/repos.py   → 连接仓库时，加密 token 后存入数据库
        2. services/repos_sync.py → 同步前，从数据库取出密文解密，再调 GitHub API

为什么需要加密：
    GitHub Token 等同于账号权限。如果明文存在 PostgreSQL 里，
    任何能查库的人（或误导出的备份文件）都能直接拿到。
    加密后即使数据库泄露，没有密钥也无法还原。

安全边界（重要）：
    这是"课程项目级"的加密，不是生产级方案：
        - 密钥由固定字符串派生（见 _derive_demo_key），不是随机生成后妥善保管
        - 密钥写在 .env 的 TOKEN_ENCRYPTION_KEY，默认值是 "change-me-in-local-dev"
    上生产必须改成：随机生成密钥 + 用 KMS 托管 + 支持密钥轮换。
"""

from cryptography.fernet import Fernet, InvalidToken


def _derive_demo_key(secret: str) -> bytes:
    """把任意长度的字符串密钥，派生成 Fernet 要求的 32 字节 base64 密钥。

    Fernet 的硬性要求：
        密钥必须是 32 字节，且经过 urlsafe base64 编码。

    这里的做法：
        用 SHA-256 把任意长度输入压成固定 32 字节，再做 base64 编码。

    为什么这样做（而不是直接用配置里的字符串）：
        配置里的 "change-me-in-local-dev" 只有 21 字节，长度不够，Fernet 会直接报错。

    为什么不适合生产：
        SHA-256 是确定性的——同样的输入永远得到同样的密钥。
        这意味着密钥强度完全取决于原始字符串的强度，且无法做盐值隔离。
    """
    # Fernet 需要 urlsafe base64 编码的 32 字节密钥。
    # 这里的确定性本地开发派生方式适合课程项目，不适合生产环境。
    # 局部 import：这两个模块只在本函数用到，放在顶层会增加无关模块的加载开销
    import base64
    import hashlib

    # digest() 得到 32 字节二进制摘要；urlsafe_b64encode 转成 Fernet 要的格式
    return base64.urlsafe_b64encode(hashlib.sha256(secret.encode("utf-8")).digest())


def encrypt_token(token: str, secret: str) -> str:
    """加密明文 token，返回可以直接存数据库的字符串。

    参数：
        token  → 明文 token，如 "ghp_xxxxxxxx"
        secret → 加密密钥，来自 settings.token_encryption_key

    返回：
        Fernet 密文（base64 字符串），可安全存入 TEXT 字段。

    注意：
        Fernet 每次加密都会随机生成 IV（初始化向量），
        所以同一个 token 加密两次，得到的密文是不一样的——这是正常且必要的，
        否则攻击者可以通过"密文是否相同"推测出两个仓库用了同一个 token。
    """
    return Fernet(_derive_demo_key(secret)).encrypt(token.encode("utf-8")).decode("utf-8")


def decrypt_token(encrypted: str, secret: str) -> str | None:
    """解密数据库里存的密文，还原成明文 token。

    参数：
        encrypted → 加密后的密文
        secret    → 与加密时相同的密钥

    返回：
        解密成功的明文 token；解密失败返回 None。

    为什么失败返回 None 而不是抛异常：
        解密失败最常见的原因是「换了 TOKEN_ENCRYPTION_KEY」——
        此时旧密文全部无法解开。这种情况下程序不应该崩溃，
        而是应该降级为「没有 token」，让同步以匿名身份继续跑（只能读公开仓库）。
        调用方 repos_sync 会把 None 当作"无凭证"处理。
    """
    try:
        return Fernet(_derive_demo_key(secret)).decrypt(encrypted.encode("utf-8")).decode("utf-8")
    except InvalidToken:
        # 密钥不匹配，或密文被篡改/损坏 → 当作"没有凭证"
        return None
