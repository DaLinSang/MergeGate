import type { Metadata } from "next";
import "./globals.css";
import { AppShell } from "@/components/layout/app-shell";

// 浏览器标签页信息。description 用英文，和后端 OpenAPI 的描述保持一致
export const metadata: Metadata = {
  title: "DevFlow AI",
  description: "PR / Issue intelligent collaboration agent for engineering teams"
};

/**
 * Next.js App Router 的根布局：所有页面都套在这一层里。
 *
 * 【数据流向】
 *    各页面（app 下每个 page.tsx）作为 children 注入 → 外面包一层 AppShell（顶栏导航）
 *
 * 【要点】
 *    lang="zh-CN" 不能省：formatDate 用的是 Intl.DateTimeFormat("zh-CN")，
 *    页面语言标记错了会影响字体回退和屏幕阅读器朗读。
 */
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body>
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
