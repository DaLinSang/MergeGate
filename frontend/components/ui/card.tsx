import { clsx } from "clsx";

/**
 * 通用 UI 原子组件
 * ================
 *
 * 【业务定位】全站复用的小组件。没有业务逻辑，只有样式约定，改这里会影响所有页面。
 *
 * 【要点】
 *    1. 一律用 clsx 合并 className，调用方传的 className 在后面 —— 后写的赢，能覆盖默认样式
 *    2. GhostButton 默认 w-full + text-left：它是"列表项按钮"，不是普通按钮
 */

/** 卡片容器。页面里每个功能区块都用它，保证圆角 / 阴影 / 背景一致。 */
export function Card({ className, children }: { className?: string; children: React.ReactNode }) {
  return (
    <section className={clsx("rounded-lg border border-line/80 bg-white/90 p-5 shadow-sm shadow-slate-200/60 backdrop-blur", className)}>
      {children}
    </section>
  );
}

/** 页面标题区：小标签 + 主标题 + 说明文字。每个页面顶部一个。 */
export function PageTitle({ title, description }: { title: string; description: string }) {
  return (
    <div className="mb-6 flex flex-col gap-2 border-b border-line/80 pb-5">
      <div className="text-xs font-medium uppercase tracking-[0.18em] text-accent">DevFlow AI</div>
      <h1 className="text-2xl font-semibold tracking-normal text-ink md:text-3xl">{title}</h1>
      <p className="max-w-3xl text-sm leading-6 text-slate-600">{description}</p>
    </div>
  );
}

/** 主操作按钮（提交 / 连接 / 开始检索）。disabled 时自动降透明度并禁用指针。 */
export function PrimaryButton({ children, className, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      className={clsx(
        "inline-flex items-center justify-center rounded-md bg-accent px-4 py-2 text-sm font-medium text-white shadow-sm shadow-teal-900/10 transition hover:bg-teal-700 disabled:cursor-not-allowed disabled:opacity-60",
        className
      )}
      {...props}
    >
      {children}
    </button>
  );
}

/** 次级按钮：整行宽、左对齐，用作列表里的可选条目。 */
export function GhostButton({ children, className, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      className={clsx(
        "w-full rounded-md border border-line bg-white px-3 py-2 text-left text-sm text-slate-700 transition hover:border-teal-200 hover:bg-teal-50/70 hover:text-teal-900",
        className
      )}
      {...props}
    >
      {children}
    </button>
  );
}
