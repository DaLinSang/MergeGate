/**
 * 跨页面复用的基础类型
 * ====================
 *
 * 【业务定位】只放"多个页面都要用"的实体类型，页面专属类型就近定义在页面文件里。
 * 【数据流向】后端 app/schemas/repos.py::RepoResponse → 这里 → app/repos/page.tsx 等
 * 【要点】字段名必须和后端 Pydantic 模型严格一致（不做驼峰转换），否则运行期拿到 undefined。
 */

export type Repository = {
  id: string;                             // 内部 ID（UUID，后端主键）
  owner: string;                          // 账号/组织名
  name: string;                           // 仓库名
  full_name: string;                      // owner/name，连接时的去重键（upsert 按它查）
  provider?: string;                      // 平台类型：github / github_compatible
  api_base_url?: string;                  // 企业版 GitHub 的 API 地址，公有云为空
  clone_url?: string | null;              // clone 地址
  local_path?: string | null;             // 本地代码目录（代码图谱和工作区读取都靠它）
  checkout_mode?: string;                 // managed（系统克隆）/ local（直接用本地目录）
  description?: string;                   // 仓库描述
  default_branch?: string;                // 默认分支（PR 快照要切回它）
  last_sync_at?: string | null;           // 上次同步时间，前端据此显示"数据新鲜度"
};
