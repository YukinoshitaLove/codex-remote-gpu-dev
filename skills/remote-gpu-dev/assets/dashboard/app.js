"use strict";

const byId = (id) => document.getElementById(id);
const clamp = (value) => Math.max(0, Math.min(100, Number(value) || 0));
const text = (value, fallback = "—") => value === null || value === undefined || value === "" ? fallback : String(value);
const TENSORBOARD_STATES = {
  starting: { label: "启动中", className: "offline", detail: "TensorBoard sidecar 正在启动并等待健康检查。" },
  live: { label: "在线", className: "live", detail: "TensorBoard 已连接。" },
  offline: { label: "离线", className: "offline", detail: "TensorBoard 当前不可连接；训练与工单状态不受影响。" },
  failed: { label: "失败", className: "failed", detail: "TensorBoard 启动或代理失败；训练与工单状态不受影响。" },
  stopped: { label: "已停止", className: "stopped", detail: "TensorBoard 已停止，可保留 event 文件供稍后重新启动。" },
  cleanup_pending: { label: "待清理", className: "failed", detail: "TensorBoard 精确停止尚未完成；GPU 工单状态不由此自动改变。" },
};
const TENSORBOARD_TICKET_RE = /^GPU-[\p{L}\p{N}_-]+$/u;

let selectedTensorboardTicketId = null;
let selectedTensorboardItem = null;
let loadedTensorboardPath = null;
let tensorboardControlPending = false;
let tensorboardControlNotice = { ticketId: null, kind: "", message: "" };
let suppressedTensorboardTicketId = null;
const tensorboardViewerOrigin = `${window.location.protocol}//localhost${window.location.port ? `:${window.location.port}` : ""}`;

function node(tag, className, content) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (content !== undefined) element.textContent = text(content);
  return element;
}

function formatBytes(value) {
  if (!Number.isFinite(Number(value))) return "—";
  const gib = Number(value) / (1024 ** 3);
  return `${gib.toFixed(gib >= 10 ? 1 : 2)} GiB`;
}

function formatPercent(value) {
  return Number.isFinite(Number(value)) ? `${Math.round(Number(value))}%` : "—";
}

function formatPower(value) {
  return Number.isFinite(Number(value)) ? `${(Number(value) / 1000).toFixed(0)} W` : "—";
}

function formatTime(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? text(value) : date.toLocaleString("zh-CN", { hour12: false });
}

function gpuHolders(ticket) {
  const result = new Map();
  for (const item of ticket?.active || []) {
    for (const gpuId of item.assigned_gpus || []) {
      if (!result.has(gpuId)) result.set(gpuId, []);
      result.get(gpuId).push(item);
    }
  }
  return result;
}

function metric(label, value) {
  const box = node("div", "metric");
  box.append(node("label", "", label), node("strong", "", value));
  return box;
}

function tensorboardMetadata(item) {
  return item?.tensorboard && typeof item.tensorboard === "object" ? item.tensorboard : null;
}

function tensorboardState(item) {
  const metadata = tensorboardMetadata(item);
  const key = typeof metadata?.status === "string" ? metadata.status.toLowerCase() : "unknown";
  return { key, ...(TENSORBOARD_STATES[key] || { label: "未知", className: "unknown", detail: "TensorBoard 状态暂不可用。" }) };
}

function tensorboardPath(item) {
  const metadata = tensorboardMetadata(item);
  const ticketId = typeof item?.id === "string" ? item.id : "";
  const suffixLength = Array.from(ticketId.slice(4)).length;
  if (!metadata || !TENSORBOARD_TICKET_RE.test(ticketId) || suffixLength < 1 || suffixLength > 156) return null;
  const expected = `/tb/${encodeURIComponent(ticketId)}/`;
  const advertised = typeof metadata.path_prefix === "string" ? metadata.path_prefix : "";
  const normalized = advertised.endsWith("/") ? advertised : `${advertised}/`;
  return normalized === expected ? `${tensorboardViewerOrigin}${expected}` : null;
}

function tensorboardButton(item) {
  const state = tensorboardState(item);
  const button = node("button", `tensorboard-button ${state.className}`);
  button.type = "button";
  button.setAttribute("aria-controls", "tensorboard-panel");
  button.setAttribute("aria-expanded", String(selectedTensorboardTicketId === item.id));
  button.append(node("span", "tensorboard-button-dot"), node("span", "", `TensorBoard · ${state.label}`));
  button.addEventListener("click", () => {
    selectedTensorboardTicketId = item.id;
    selectedTensorboardItem = item;
    if (tensorboardControlNotice.ticketId !== item.id) {
      tensorboardControlNotice = { ticketId: null, kind: "", message: "" };
    }
    for (const candidate of document.querySelectorAll(".tensorboard-button")) {
      candidate.setAttribute("aria-expanded", String(candidate === button));
    }
    renderTensorboardPanel(item);
    byId("tensorboard-panel").scrollIntoView({ behavior: "smooth", block: "start" });
  });
  return button;
}

function renderGpuCard(gpu, holders, remoteConnected) {
  const card = node("article", "gpu-card");
  const head = node("div", "gpu-head");
  const identity = node("div");
  identity.append(node("div", "gpu-id", `GPU ${gpu.index}`), node("div", "gpu-name", gpu.name));
  const assignments = holders.get(gpu.index) || [];
  const processCount = (gpu.processes || []).length;
  let state = "空闲";
  let stateClass = "state-pill";
  if (!remoteConnected) {
    state = "指标离线";
    stateClass += " unknown";
  } else if (assignments.length) {
    state = "已派单";
    stateClass += " busy";
  } else if (processCount) {
    state = "未登记进程";
    stateClass += " foreign";
  }
  head.append(identity, node("span", stateClass, state));
  card.append(head);

  const hero = node("div", "metric-hero");
  hero.append(node("strong", "", formatPercent(gpu.utilization)), node("span", "", "GPU UTIL"));
  const utilBar = node("div", "bar");
  const utilFill = node("span");
  utilFill.style.width = `${clamp(gpu.utilization)}%`;
  utilBar.append(utilFill);
  card.append(hero, utilBar);

  const memoryLabel = `${formatBytes(gpu.memory_used_bytes)} / ${formatBytes(gpu.memory_total_bytes)}`;
  const memoryBar = node("div", "bar");
  const memoryFill = node("span");
  memoryFill.style.width = `${clamp(gpu.memory_percent)}%`;
  memoryBar.append(memoryFill);
  card.append(node("div", "muted", `显存 ${memoryLabel}`), memoryBar);

  const metrics = node("div", "metric-row");
  metrics.append(
    metric("温度", Number.isFinite(Number(gpu.temperature_c)) ? `${gpu.temperature_c}°C` : "—"),
    metric("功耗", formatPower(gpu.power_mw)),
    metric("进程", processCount)
  );
  card.append(metrics);

  const allocation = node("div", "allocation");
  if (assignments.length) {
    allocation.textContent = assignments.map((item) => `${item.project} · ${item.owner} · ${item.status}`).join(" / ");
  } else {
    allocation.textContent = "账本无持有态工单";
  }
  card.append(allocation);
  if (processCount) {
    const processLine = node("div", "processes");
    processLine.textContent = gpu.processes.slice(0, 8).map((item) => `PID ${item.pid} · ${formatBytes(item.gpu_memory_bytes)}`).join("  |  ");
    card.append(processLine);
  }
  return card;
}

function renderTicket(item, queued = false) {
  const row = node("div", "ticket");
  if (selectedTensorboardTicketId === item.id) row.classList.add("selected");
  const body = node("div");
  body.append(node("div", "ticket-title", `${item.project} · ${item.status}`));
  body.append(node("div", "ticket-purpose", item.purpose));
  body.append(node("div", "ticket-meta", `${item.owner} · 更新 ${formatTime(item.updated_at)}`));
  const gpuLabel = queued ? `请求 ${text(item.requested_gpus)} GPU` : `GPU ${(item.assigned_gpus || []).join(",")}`;
  const actions = node("div", "ticket-actions");
  actions.append(node("div", "ticket-gpus", gpuLabel));
  if (!queued && tensorboardMetadata(item)) actions.append(tensorboardButton(item));
  row.append(body, actions);
  return row;
}

function renderTicketList(target, items, queued = false) {
  target.replaceChildren();
  if (!items.length) {
    target.append(node("div", "empty", queued ? "当前没有排队工单" : "当前没有持有态工单"));
    return;
  }
  for (const item of items) target.append(renderTicket(item, queued));
}

function renderRecent(target, items) {
  target.replaceChildren();
  if (!items.length) {
    target.append(node("div", "empty", "暂无终态工单"));
    return;
  }
  for (const item of items) {
    const row = node("div", `recent ${["failed", "cancelled", "expired"].includes(item.status) ? item.status : ""}`);
    if (selectedTensorboardTicketId === item.id) row.classList.add("selected");
    const body = node("div", "recent-body");
    body.append(node("div", "ticket-title", `${item.project} · ${item.status}`));
    body.append(node("div", "ticket-meta", `${formatTime(item.updated_at)} · ${text(item.result, "无结果摘要")}`));
    row.append(body);
    if (tensorboardMetadata(item)) row.append(tensorboardButton(item));
    target.append(row);
  }
}

function unloadTensorboardFrame() {
  const frame = byId("tensorboard-frame");
  frame.hidden = true;
  if (loadedTensorboardPath !== null) {
    frame.removeAttribute("src");
    loadedTensorboardPath = null;
  }
}

function renderTensorboardPanel(item) {
  const panel = byId("tensorboard-panel");
  if (!item || !tensorboardMetadata(item)) {
    selectedTensorboardTicketId = null;
    selectedTensorboardItem = null;
    panel.hidden = true;
    unloadTensorboardFrame();
    return;
  }

  selectedTensorboardItem = item;
  const state = tensorboardState(item);
  const status = byId("tensorboard-status");
  status.className = `tensorboard-status ${state.className}`;
  status.textContent = state.label;
  byId("tensorboard-title").textContent = `${text(item.project, "未命名实验")} · TensorBoard`;
  byId("tensorboard-meta").textContent = `${text(item.id, "未知工单")} · ${text(item.owner, "未知负责人")}`;
  panel.hidden = false;

  const lifecycle = byId("tensorboard-lifecycle");
  const controlAction = tensorboardControlAction(item);
  lifecycle.hidden = controlAction === null;
  lifecycle.disabled = tensorboardControlPending;
  lifecycle.className = `lifecycle-button ${controlAction === "close" ? "stop" : "start"}`;
  lifecycle.textContent = tensorboardControlPending
    ? "处理中…"
    : controlAction === "close"
      ? state.key === "live"
        ? "停止 TensorBoard"
        : "重试清理 TensorBoard"
      : "启动 TensorBoard";

  const controlMessage = byId("tensorboard-control-message");
  const notice = tensorboardControlNotice.ticketId === item.id ? tensorboardControlNotice : null;
  controlMessage.className = `tensorboard-control-message ${notice?.kind || ""}`;
  controlMessage.textContent = notice?.message || "";

  const frame = byId("tensorboard-frame");
  const empty = byId("tensorboard-empty");
  const path = tensorboardPath(item);
  if (suppressedTensorboardTicketId === item.id && state.key === "stopped") {
    suppressedTensorboardTicketId = null;
  }
  if (state.key === "live" && path && suppressedTensorboardTicketId !== item.id) {
    empty.hidden = true;
    frame.hidden = false;
    if (loadedTensorboardPath !== path) {
      frame.src = path;
      loadedTensorboardPath = path;
    }
    return;
  }

  unloadTensorboardFrame();
  empty.hidden = false;
  byId("tensorboard-empty-title").textContent = suppressedTensorboardTicketId === item.id
    ? "正在停止 TensorBoard"
    : state.key === "live"
      ? "TensorBoard 路径不可用"
      : `TensorBoard ${state.label}`;
  const lastError = tensorboardMetadata(item).last_error;
  const stateDetail = suppressedTensorboardTicketId === item.id
    ? "等待工单快照确认停止；event 文件不会被删除。"
    : state.detail;
  byId("tensorboard-empty-detail").textContent = lastError ? `${stateDetail} ${text(lastError)}` : stateDetail;
}

function closeTensorboardPanel() {
  selectedTensorboardTicketId = null;
  selectedTensorboardItem = null;
  byId("tensorboard-panel").hidden = true;
  unloadTensorboardFrame();
}

function tensorboardControlAction(item) {
  const state = tensorboardState(item).key;
  if (["starting", "live", "failed", "cleanup_pending"].includes(state)) return "close";
  if (state === "stopped") return "open";
  return null;
}

async function controlSelectedTensorboard() {
  const item = selectedTensorboardItem;
  const action = tensorboardControlAction(item);
  if (!item || !action || tensorboardControlPending) return;

  const ticketId = item.id;
  tensorboardControlPending = true;
  tensorboardControlNotice = {
    ticketId,
    kind: "pending",
    message: action === "open" ? "正在启动 TensorBoard…" : "正在精确停止 TensorBoard…",
  };
  if (action === "close") {
    suppressedTensorboardTicketId = ticketId;
    unloadTensorboardFrame();
  }
  renderTensorboardPanel(item);

  try {
    const response = await fetch(`/api/tensorboard/${action}`, {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticket_id: ticketId }),
    });
    let payload = null;
    try {
      payload = await response.json();
    } catch (_error) {
      payload = null;
    }
    if (!response.ok || !payload?.ok) {
      throw new Error(text(payload?.error, `HTTP ${response.status}`));
    }
    tensorboardControlNotice = {
      ticketId,
      kind: "success",
      message: action === "open"
        ? "TensorBoard 已启动，等待实时视图刷新。"
        : "TensorBoard 已停止；event 文件仍然保留。",
    };
    await refresh();
  } catch (error) {
    if (action === "close" && suppressedTensorboardTicketId === ticketId) {
      suppressedTensorboardTicketId = null;
    }
    tensorboardControlNotice = {
      ticketId,
      kind: "error",
      message: `操作失败：${text(error?.message, "未知错误")}`,
    };
  } finally {
    tensorboardControlPending = false;
    if (selectedTensorboardItem?.id === ticketId) {
      renderTensorboardPanel(selectedTensorboardItem);
    }
  }
}

function render(payload) {
  const ticketState = payload.ticket || {};
  const ticket = ticketState.snapshot || {};
  const remote = payload.remote || {};
  const sample = remote.sample || {};
  const bothLive = Boolean(ticketState.connected && remote.connected);
  byId("live-dot").className = `dot ${bothLive ? "live" : "error"}`;
  byId("connection-label").textContent = bothLive ? "实时连接" : "数据不完整";
  byId("last-update").textContent = `刷新 ${formatTime(payload.generated_at)}`;
  byId("server-label").textContent = `${text(payload.profile?.name)} · ${text(ticket.server)} · ${text(remote.hello?.hostname)}`;
  byId("collector-label").textContent = `nvitop ${text(remote.hello?.nvitop_version)} · 延迟 ${text(remote.age_seconds)}s`;

  const warnings = byId("warnings");
  warnings.replaceChildren();
  if (ticketState.error) warnings.append(node("div", "warning error", ticketState.error));
  if (remote.error) warnings.append(node("div", "warning warn", remote.error));
  for (const item of payload.warnings || []) warnings.append(node("div", `warning ${item.level || "info"}`, item.message));

  const holders = gpuHolders(ticket);
  const gpuGrid = byId("gpu-grid");
  gpuGrid.replaceChildren();
  const gpus = [...(sample.gpus || [])].sort((a, b) => a.index - b.index);
  if (!gpus.length) gpuGrid.append(node("div", "empty", "等待远端 GPU 快照"));
  for (const gpu of gpus) gpuGrid.append(renderGpuCard(gpu, holders, Boolean(remote.connected)));

  const active = ticket.active || [];
  const queued = ticket.queued || [];
  byId("active-count").textContent = String(active.length);
  byId("queue-count").textContent = String(queued.length);
  renderTicketList(byId("active-tickets"), active, false);
  renderTicketList(byId("queued-tickets"), queued, true);
  const history = ticket.history || ticket.recent || [];
  renderRecent(byId("recent-tickets"), history);

  if (selectedTensorboardTicketId !== null) {
    const selected = [...active, ...history].find((item) => item.id === selectedTensorboardTicketId);
    renderTensorboardPanel(selected);
  }
}

async function refresh() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    render(await response.json());
  } catch (_error) {
    byId("live-dot").className = "dot error";
    byId("connection-label").textContent = "本地服务断开";
  }
}

byId("tensorboard-close").addEventListener("click", closeTensorboardPanel);
byId("tensorboard-lifecycle").addEventListener("click", controlSelectedTensorboard);
refresh();
setInterval(refresh, 1500);
