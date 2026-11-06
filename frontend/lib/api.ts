/**
 * 前端 → 后端的唯一 HTTP 出口
 * ===========================
 *
 * 【业务定位】
 *    七条数据流在前端的第一站。所有页面 / 组件都不直接调 fetch，一律走这里。
 *    好处：baseURL、错误格式、SSE 分帧这三件容易写错的事，只在一个地方处理。
 *
 * 【数据流向】
 *    入：app 下各页面与 components 组件传入 path（如 "/api/repos"，后端统一挂 /api 前缀）+ body
 *    出：JSON 反序列化 → 交给页面渲染（类型见 lib/rag.ts）
 *    异常：非 2xx 一律抛 Error，message 优先取后端的 detail
 *
 * 【要点】
 *    1. 一律 cache: "no-store" —— 后端数据会被同步任务随时改写，走 Next 缓存会看到旧值
 *    2. apiUpload 不能手写 Content-Type —— 浏览器要自己补 multipart 的 boundary
 *    3. apiPostStream 只认 "\n\n" 分帧（SSE 规范）；最后一段可能不完整，必须留在 buffer 里
 */

// 后端地址：构建时注入 NEXT_PUBLIC_API_BASE_URL，没配就退到本地 8000
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

/** GET 请求，返回 JSON。用于所有只读列表 / 详情页的首屏加载。 */
export async function apiGet<T>(path: string): Promise<T> {
  // no-store：禁止 Next 在服务端缓存响应，否则同步完 GitHub 后页面还是旧数据
  const response = await fetch(`${API_BASE_URL}${path}`, { cache: "no-store" });
  if (!response.ok) await throwApiError(response);
  return response.json();
}

/** POST + JSON body。用于创建类操作（连接仓库、发起对话、触发评测）。 */
export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!response.ok) await throwApiError(response);
  return response.json();
}

/** PATCH + JSON body。用于部分更新（改知识库配置、改草稿状态）。 */
export async function apiPatch<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!response.ok) await throwApiError(response);
  return response.json();
}

/**
 * POST + FormData。用于文件上传（RAG 入库的文档）。
 *
 * 【为什么不能像 apiPost 那样写 Content-Type】
 *    手写 multipart/form-data 会丢掉 boundary 参数，后端解析直接 400。
 *    交给浏览器自动生成，它会把 boundary 一起写进 header。
 */
export async function apiUpload<T>(path: string, body: FormData): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    body
  });
  if (!response.ok) await throwApiError(response);
  return response.json();
}

/**
 * POST + 读 SSE 流。对话页（app/chat）唯一的流式通道。
 *
 * 【调用链】追这个函数会看到：
 *    1. reader.read()        → 按 chunk 拿字节流，一个 chunk 不保证是一个完整事件
 *    2. decoder.decode()     → 增量解码（stream: true），中文不会被从字节中间切断
 *    3. buffer.split("\n\n") → SSE 用空行分帧，所以只能按 "\n\n" 切
 *    4. onEvent(event, data) → 回调给页面，按事件名分别处理 token / tool / done
 *
 * 【数据流向】
 *    入：path（如 "/api/chat/stream"）+ body + AbortSignal（用户点"停止生成"）
 *    出：不返回数据，全部通过 onEvent 回调推送；函数结束 = 流关闭
 *
 * 【关键规则】
 *    1. 只按 "\n\n" 切帧：按 "\n" 切会把一条事件的 event/data 两行拆散
 *    2. 最后一段必须留在 buffer：它可能是半条事件，下一轮 chunk 到了再拼
 *    3. data 可能跨多行（SSE 允许），所以要 filter + join("\n")
 *    4. 空 data 直接跳过：后端会发心跳 / 注释行，JSON.parse("") 会炸
 */
export async function apiPostStream(
  path: string,
  body: unknown,
  onEvent: (event: string, data: unknown) => void,
  options: { signal?: AbortSignal } = {}
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: options.signal
  });
  if (!response.ok) await throwApiError(response);
  if (!response.body) throw new Error("Streaming response body is not available.");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  // buffer 跨 chunk 累积：一个 SSE 事件可能被拆到两个网络包里
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    // stream: true 让多字节的中文字符等下一个 chunk 补齐再解码
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    // pop() 取出最后一段（可能是半条），留给下一轮拼接
    buffer = events.pop() ?? "";
    for (const rawEvent of events) {
      const lines = rawEvent.split("\n");
      // 没有 event: 行时按 SSE 默认事件名 "message" 处理
      const event = lines.find((line) => line.startsWith("event: "))?.slice(7) ?? "message";
      // data: 可能有多行，按规范用 "\n" 拼回去
      const dataText = lines
        .filter((line) => line.startsWith("data: "))
        .map((line) => line.slice(6))
        .join("\n");
      // 空 data 跳过：心跳帧 / 注释帧没有数据体
      if (dataText) onEvent(event, JSON.parse(dataText));
    }
  }
}

/** DELETE 请求。用于删除仓库、删文档、取消草稿。 */
export async function apiDelete<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, { method: "DELETE" });
  if (!response.ok) await throwApiError(response);
  return response.json();
}

/**
 * 把非 2xx 响应转成带可读 message 的 Error。
 *
 * 【为什么单独抽出来】
 *    FastAPI 的报错有三种形态，前端要统一成一句话给用户看：
 *      1. HTTPException        → {"detail": "字符串"}
 *      2. Pydantic 校验失败 422 → {"detail": [{"msg": "...", "loc": [...]}]}
 *      3. 其它（网关 / 代理）   → 纯文本或 HTML
 *    直接 String(payload) 会让 422 显示成 "[object Object]"。
 */
async function throwApiError(response: Response): Promise<never> {
  const fallback = `${response.status} ${response.statusText}`.trim();
  const contentType = response.headers.get("content-type") ?? "";
  if (contentType.includes("application/json")) {
    let message: string | null = null;
    try {
      const payload = await response.json();
      message = readApiErrorMessage(payload);
    } catch {
      // body 不是合法 JSON（少见，但网关会返回 JSON 类型的错误页）→ 退回状态码
      message = null;
    }
    throw new Error(message || fallback);
  }
  const text = (await response.text()).trim();
  throw new Error(text || fallback);
}

/**
 * 从后端错误体里挖出人能读的那句话。
 *
 * 【覆盖的三种形态】
 *    1. 纯字符串                → 直接用
 *    2. detail 是字符串         → HTTPException，直接用
 *    3. detail 是数组           → Pydantic 422，逐条取 msg 拼成 "; "
 *    4. message 字段            → 兼容非 FastAPI 风格的中间层
 *    都没有 → 返回 null，由上层退回 "状态码 + 状态文本"
 */
function readApiErrorMessage(payload: unknown): string | null {
  if (typeof payload === "string") return payload;
  if (!payload || typeof payload !== "object") return null;
  const detail = (payload as { detail?: unknown }).detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    // 422 的 detail 是数组，每项形如 {loc: [...], msg: "field required", type: "..."}
    const messages = detail
      .map((item) => {
        if (item && typeof item === "object" && "msg" in item) return String((item as { msg: unknown }).msg);
        return typeof item === "string" ? item : null;
      })
      .filter(Boolean);
    return messages.length ? messages.join("; ") : null;
  }
  const message = (payload as { message?: unknown }).message;
  return typeof message === "string" ? message : null;
}
