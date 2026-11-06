import Link from "next/link";

/**
 * 顶部导航
 * ========
 *
 * 【业务定位】全站唯一的导航入口，按"用户做事的顺序"排：知识 → RAG → 仓库 → 草稿 → 运行 → 报告。
 *
 * 【要点】
 *    1. 导航项是硬编码的，新增页面要记得回来加一行，否则只能靠 URL 直接访问
 *    2. 当前导航里没有 /chat、/issues、/pull-requests、/ci —— 这几个入口在首页卡片里
 */
const navItems = [
  { href: "/", label: "Workspace" },          // 首页看板（app/page.tsx，全站最大的文件）
  { href: "/knowledge", label: "项目知识" },    // 已入库文档列表 + 切片查看
  { href: "/rag", label: "RAG 工作室" },       // 检索调试台：改配置、看六阶段耗时
  { href: "/repos", label: "Repos" },         // 连接仓库 / 触发同步（数据流 1 的入口）
  { href: "/action-drafts", label: "Drafts" }, // AI 写 GitHub 前的草稿确认（数据流 7 的闸门）
  { href: "/workflow-runs", label: "Runs" },   // 多 Agent 编排的执行记录
  { href: "/workspaces", label: "Reports" },   // 注意：label 是 Reports，实际指向 /workspaces
  { href: "/code-graph", label: "Code Graph" }, // 代码图谱（数据流 2 的产物）
  { href: "/evals", label: "Evals" }          // 评测中心（RAGAS + 硬规则）
];

/** 全站外壳：一条顶栏 + 内容区。layout.tsx 里包住所有页面。 */
export function AppShell({ children }: { children: React.ReactNode }) {
  return (
    <main className="min-h-screen">
      <div className="border-b border-line bg-white/85 px-4 py-3">
        <div className="flex flex-wrap items-center gap-2">
          <div className="mr-2 text-sm font-semibold text-ink">DevFlow AI</div>
          {navItems.map((item) => (
            <Link key={item.href} href={item.href} className="rounded-md px-3 py-1.5 text-sm text-slate-600 transition hover:bg-teal-50 hover:text-teal-800">
              {item.label}
            </Link>
          ))}
        </div>
      </div>
      <div className="w-full p-3">{children}</div>
    </main>
  );
}
