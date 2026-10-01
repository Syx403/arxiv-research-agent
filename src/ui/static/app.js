"use strict";

const $ = (selector) => document.querySelector(selector);
const state = {session: null, meta: null, sessions: [], submitting: false, pending: null,
  reasoning: null, refreshVersion: 0, selectionVersion: 0};
const labels = {needs_input: "待补充需求",complete: "核验完成", partial: "部分完成", incomplete: "未完成", running: "研究中"};
const money = (value, digits = 4) => `$${Number(value || 0).toFixed(digits)}`;

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function text(element, value) {
  if (element.textContent !== value) element.textContent = value;
}

function updatePart(element, value, build) {
  const fingerprint = JSON.stringify(value) ?? "null";
  if (element.dataset.fingerprint === fingerprint) return false;
  element.replaceChildren(...build());
  element.dataset.fingerprint = fingerprint;
  return true;
}

async function api(path, options = {}) {
  const response = await fetch(`/api${path}`, {
    ...options,
    headers: {"Content-Type": "application/json", "X-ARA-Client": "local-ui", ...options.headers},
  });
  let result;
  try { result = await response.json(); } catch { throw new Error("本地服务未返回有效结果，请检查服务是否仍在运行。"); }
  if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "请求未完成，请检查输入后重试。");
  return result;
}

function notice(message = "") {
  text($("#notice"), message);
  $("#notice").hidden = !message;
}

function renderMeta() {
  if (!state.meta) return;
  const budget = state.meta.budget;
  text($("#budget-used"), money(budget.committed_usd, 3));
  text($("#budget-limit"), `/ ${money(budget.limit_usd, 2)}`);
  $("#budget-meter").max = budget.limit_usd;
  $("#budget-meter").value = budget.committed_usd;
  text($("#budget-note"), `余额 ${money(Math.max(0, budget.limit_usd - budget.committed_usd), 3)} · 请求 ${budget.requests} / ${budget.max_requests}` + (budget.uncertain_or_reserved_usd > 0 ? `\n含 ${money(budget.uncertain_or_reserved_usd)} 预留或待确认费用` : ""));
  if (!state.reasoning) {
    state.reasoning = {main: state.meta.profiles.main.effort, fast: state.meta.profiles.fast.effort};
    for (const role of ["main", "fast"]) $(`#${role}-effort`).value = state.reasoning[role];
  }
  text($("#corpus-label"), "arXiv 实时搜索 · 正文按需复用");
  renderReasoning();
  controls();
}

function renderSessions() {
  const container = $("#sessions");
  const ids = new Set(state.sessions.map((session) => session.id));
  for (const child of [...container.children]) {
    if (child.dataset.sessionId ? !ids.has(child.dataset.sessionId) : ids.size > 0) child.remove();
  }
  if (!state.sessions.length && !container.querySelector(".history-empty")) container.append(node("p", "muted history-empty", "你的研究会保存在这里。"));
  for (const [index, session] of state.sessions.entries()) {
    let button = [...container.children].find((child) => child.dataset.sessionId === session.id);
    if (!button) {
      button = node("button", "session");
      button.dataset.sessionId = session.id;
      button.addEventListener("click", () => selectSession(session.id).catch((error) => notice(error.message)));
    }
    text(button, session.title);
    button.classList.toggle("active", session.id === state.session?.id);
    button.title = session.title;
    if (session.id === state.session?.id) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
    if (container.children[index] !== button) container.insertBefore(button, container.children[index] || null);
  }
}

function renderReasoning() {
  if (!state.reasoning) return;
  text($("#thinking-summary"), `主任务 ${state.reasoning.main} · 轻量任务 ${state.reasoning.fast}`);
  text($("#model-label"), `下一轮 · Flash · ${state.reasoning.main} / ${state.reasoning.fast}`);
}

function details(label, className) {
  const element = node("details", className);
  element.append(node("summary", "", label));
  return element;
}

function answerText(text, sources, turnId) {
  const paragraph = node("div", "answer-text");
  const tokens = /(\[([^\]\n]+)#(\d+)\]|\*\*([^*]+)\*\*|`([^`]+)`)/g;
  let offset = 0;
  for (const match of text.matchAll(tokens)) {
    paragraph.append(document.createTextNode(text.slice(offset, match.index)));
    if (match[2]) {
      const index = sources.findIndex((source) => source.paper_id === match[2] && source.chunk_id === Number(match[3]));
      if (index >= 0) {
        const paperIndex = [...new Set(sources.map((source) => source.paper_id))].indexOf(match[2]);
        const link = node("a", "", String(paperIndex + 1));
        link.href = `#source-${turnId}-${index}`;
        link.setAttribute("aria-label", `查看论文 ${paperIndex + 1} 的证据片段 ${match[3]}`);
        link.addEventListener("click", (event) => {
          event.preventDefault();
          const card = document.getElementById(`source-${turnId}-${index}`);
          for (let parent = card.parentElement; parent; parent = parent.parentElement) {
            if (parent.tagName === "DETAILS") parent.open = true;
          }
          card.scrollIntoView({block: "nearest", behavior: "smooth"});
          card.focus({preventScroll: true});
        });
        paragraph.append(link);
      } else paragraph.append(document.createTextNode(match[0]));
    } else if (match[4]) paragraph.append(node("strong", "", match[4]));
    else paragraph.append(node("code", "", match[5]));
    offset = match.index + match[0].length;
  }
  paragraph.append(document.createTextNode(text.slice(offset)));
  return paragraph;
}

function sourcePanel(turn) {
  const papers = new Map();
  for (const [index, source] of turn.sources.entries()) {
    if (!papers.has(source.paper_id)) papers.set(source.paper_id, []);
    papers.get(source.paper_id).push({source, index});
  }
  const panel = details(`引用论文 · ${papers.size} 篇 · ${turn.sources.length} 个证据片段`, "sources");
  let paperIndex = 0;
  for (const entries of papers.values()) {
    const first = entries[0].source;
    const card = node("article", "source-card");
    const title = node(first.url ? "a" : "div", "source-title", `${++paperIndex}. ${first.title}`);
    if (first.url) { title.href = first.url; title.target = "_blank"; title.rel = "noopener noreferrer"; }
    card.append(title, node("div", "source-meta", `${first.paper_id}${first.published_at ? ` · 发表于 ${first.published_at}` : ""} · 本次回答引用的论文`));
    const excerpts = details(`查看 ${entries.length} 个原文片段`, "source-excerpts");
    for (const {source, index} of entries) {
      const excerpt = node("section", "source-excerpt");
      excerpt.id = `source-${turn.id}-${index}`;
      excerpt.tabIndex = -1;
      excerpt.append(node("div", "source-meta", `片段 ${source.chunk_id} · 本次核验使用的原文`), node("blockquote", "", source.text));
      excerpts.append(excerpt);
    }
    card.append(excerpts);
    panel.append(card);
  }
  return panel;
}

function paperCards(papers) {
  return papers.flatMap((paper, index) => {
    const card = node("article", "source-card paper-card");
    const title = node("a", "source-title", `${index + 1}. ${paper.title}`);
    // Links are supplied by the server from validated arXiv IDs, never model text.
    title.href = paper.url; title.target = "_blank"; title.rel = "noopener noreferrer";
    const fit = {direct: "直接匹配", partial: "相关参考 · 不计入直接匹配数量", unassessed: "候选 · 未筛选", user_selected: "指定阅读"}[paper.fit] || "候选";
    const date = paper.published_at ? `${paper.date_confirmed ? "首次发表" : "搜索页日期，待确认"} ${paper.published_at.slice(0, 10)}` : "日期待确认";
    card.append(title, node("div", "source-meta", `${paper.paper_id} · ${date} · ${fit}`));
    if (paper.authors?.length) card.append(node("p", "source-meta", `arXiv 作者：${paper.authors.slice(0, 3).map((a) => a.trim()).join("、")}${paper.authors.length > 3 ? ` 等 ${paper.authors.length} 位` : ""}`));
    if (paper.summary) card.append(node("p", "", paper.summary));
    const basis = {conceptual: "摘要介绍概念方案，实施与实验依据需查阅原文。", empirical: "摘要报告了实验；具体条件和结论仍以原文为准。", not_reported: ""}[paper.evidence_basis];
    if (basis) card.append(node("p", "muted", basis));
    if (paper.why_relevant) card.append(node("p", "", `与你需求的关系：${paper.why_relevant}`));
    if (paper.uncertainty) card.append(node("p", "muted", `待确认：${paper.uncertainty}`));
    const evidence = details("查看摘要依据 · 非全文审阅", "source-excerpts");
    (paper.evidence?.length ? paper.evidence : [paper.abstract || "arXiv 未提供摘要。"])
      .forEach((quote) => evidence.append(node("blockquote", "", quote)));
    card.append(evidence);
    if (paper.fit === "partial" && (index === 0 || papers[index - 1].fit !== "partial")) {
      return [node("p", "muted", "相关参考 · 以下论文尚未直接满足全部需求"), card];
    }
    return [card];
  });
}

function inspection() {
  const panel = details("查看本轮过程", "inspection");
  const content = node("div", "inspector");
  for (const name of ["question", "profile", "discovery", "review", "subquestions", "steps", "gaps", "checks", "run-id"]) {
    const slot = node("div"); slot.dataset.part = name; content.append(slot);
  }
  panel.append(content);
  return panel;
}

function updateInspection(panel, turn) {
  const part = (name) => panel.querySelector(`[data-part="${name}"]`);
  updatePart(part("review"), turn.claim_review, () => {
    const review = turn.claim_review;
    if (!review?.targeted_followup) return [];
    const sections = [...new Set((review.checked_sources || []).map((s) => `${s.paper_id} · ${s.section || "未标注章节"}`))];
    const outcome = {supported: "找到支持该说法的原文", contradicted: "找到与该说法相反的原文证据", not_established: "在本次查阅的解析材料中未找到支持", insufficient: "尚不足以完成判断"}[review.outcome];
    return [node("h3", "", "针对论断的原文补查"),
      node("p", "", review.all_available_material_checked
        ? `补查材料覆盖指定版本当前解析的全部 ${review.checked_sources.length} 个片段。结论限于这些材料，解析仍可能遗漏内容。`
        : `本次只检查了 ${review.checked_sources?.length || 0} 个片段；不能据此断言其他章节没有相关结论。`),
      node("p", "", review.status === "complete" ? `${outcome || "补查完成"}。具体结论及依据见上方核验后的回答。` : "补查尚未完成，不能据此判断论文是否支持该说法。"),
      node("p", "muted", sections.join("；"))];
  });
  updatePart(part("question"), [turn.research_question, turn.context_messages], () => [
    node("h3", "", "实际研究的问题"), node("p", "", turn.research_question || "尚未完成问题理解"),
    node("p", "muted", `参考了 ${turn.context_messages || 0} 条历史消息；历史对话用于理解问题，论文原文用于支撑结论。`),
  ]);
  updatePart(part("profile"), turn.profiles, () => [node("p", "muted", turn.profiles
    ? `本轮模型：${turn.profiles.main.model} · 主任务 ${turn.profiles.main.effort} / 轻量任务 ${turn.profiles.fast.effort}`
    : "此历史运行未记录思考配置。")]);
  updatePart(part("discovery"), [turn.research_policy, turn.external_discovery, turn.paper_search, turn.cache_hits], () => {
    if (turn.paper_search?.as_of) {
      const report = turn.paper_search;
      const items = [node("h3", "", report.reading_target ? "论文获取与正文阅读" : "arXiv 检索与筛选"),
        node("p", "", `已执行 ${report.queries?.length || 0} 个查询 · ${report.rounds} 轮 · ${report.candidate_count} 篇去重候选 · ${turn.cache_hits || 0} 次缓存命中`),
        node("p", "muted", `查询时间 ${report.as_of}。${report.reading_target === "user_selected" ? "按你指定的论文直接读取正文，无须再次推荐筛选。" : "论文搜索先使用摘要；深入分析、实验或实现问题会读取正文。"}`)];
      const readLabels = {reading: "正在获取与索引正文", full_text: "正文已获取并索引", cached_full_text: "复用已索引正文", read_failed: "正文获取或索引失败"};
      for (const item of report.read_details || []) items.push(node("p", "", `${item.paper_id} · ${readLabels[item.status] || item.status}${item.chunks_inserted ? ` · ${item.chunks_inserted} 个正文片段` : ""}`));
      if (report.date_from) items.push(node("p", "muted", `${report.date_mode === "required" ? "限定首次发表日期" : "优先近期研究"}：${report.date_from} 至 ${report.date_to}`));
      if (report.criteria?.length) items.push(node("p", "", `筛选要点：${report.criteria.join("；")}`));
      const queries = node("ul");
      for (const query of report.queries || []) queries.append(node("li", "", `${query.terms.join(" AND ") || query.id}${query.scope === "title" ? "（标题检索）" : ""} · ${query.sort === "recent" ? "按时间" : "按相关性"} · ${query.offset ? `起点 ${query.offset} · ` : ""}${query.status || "查询中"}${query.results !== undefined ? ` · ${query.results} 篇` : ""}`));
      items.push(queries);
      if (report.reason) items.push(node("p", "", report.reason));
      if (report.stop_reason) items.push(node("p", "muted", `停止原因：${report.stop_reason} · 已评估 ${report.assessments || 0}/${report.limits?.assessments || 3} 次候选`));
      if (report.decisions?.length) {
        const decisions = node("details");
        decisions.append(node("summary", "", "查看路由决策"));
        for (const decision of report.decisions) decisions.append(node("p", "muted", `${decision.from} → ${decision.action} · ${decision.reason}`));
        items.push(decisions);
      }
      const errorLabels = {invalid_json: "输出不是合法 JSON", schema_invalid: "输出字段不符合约定", incomplete_output: "输出不完整", unknown_paper_id: "返回了候选之外的论文 ID", duplicate_paper_id: "返回了重复论文 ID", invalid_sentence_id: "摘要证据编号无效", invalid_overview_binding: "概述引用不属于选定论文", unsupported_authorship_classification: "生成了未经证实的作者归属判断"};
      if (report.errors?.length) items.push(node("p", "muted", "未完成步骤：" + report.errors.map((e) => `${e.stage} (${errorLabels[e.code] || e.code || e.http_status || e.error})`).join("；")));
      return items;
    }
    const report = turn.external_discovery;
    const mode = {auto: "自动 · 按需联网", local: "仅本地论文库", online: "本轮联网发现"}[turn.research_policy?.mode] || "arXiv 实时搜索";
    if (!report?.attempted) return [node("p", "muted", `论文来源：${mode}。本轮尚未触发外部获取。`)];
    const status = {complete: "已完成有界检索", partial: "部分获取成功", timeout: "达到外部获取时限", unavailable: "外部获取失败", planning_failed: "搜索计划解析失败，尚未查询论文", no_results: "本次查询无结果", screening_failed: "部分候选未能完成判断", no_relevant_candidates: "本批候选未通过相关性筛选", read_failed: "全文读取失败", budget_exhausted: "达到预算上限"}[report.status] || report.status;
    const dateScope = report.date_from ? (report.date_mode === "preferred" ? `优先 ${report.date_from} 至 ${report.date_to} 的近期研究，同时保留相关基础论文。` : `论文日期限定 ${report.date_from} 至 ${report.date_to}。`) : "";
    const content = [node("h3", "", "外部论文发现"), node("p", "", `${mode} · ${status} · ${report.elapsed_s || 0} 秒`), node("p", "muted", `查询时间 ${report.as_of}。${dateScope}每轮最多读取 ${report.limits?.papers || 3} 篇；这是候选样本，不代表穷尽文献或热度排名。`)];
    const queries = node("ul");
    (report.queries || []).forEach((query) => queries.append(node("li", "", `${query.provider}${query.sort ? ` · ${query.sort === "recent" ? "按时间" : "按相关性"}` : ""}: ${query.query || query.id}`)));
    content.push(queries);
    const reads = {metadata_only: "仅标题 / 摘要", full_text: "全文已读取并索引", cached_full_text: "复用已索引全文", reading: "正在读取", read_failed: "全文读取失败", interrupted: "全文读取中断"};
    const screens = {relevant: "通过筛选", needs_full_text: "最新官方报告，优先核对原文", not_relevant: "相关性不足", unknown: "判断未完成", outside_date_window: "不在日期范围内", pending: "待筛选"};
    for (const paper of report.candidates || []) {
      const card = node("div", "source-card");
      card.append(node("strong", "", paper.title), node("p", "source-meta", `${paper.paper_id} · ${paper.published_at || "日期未知"} · ${screens[paper.screen_status] || paper.screen_status} · ${reads[paper.read_status] || paper.read_status}`));
      if (/^arxiv:\d{4}\.\d{4,5}$/.test(paper.paper_id)) {
        const link = node("a", "", "查看论文页面");
        link.href = `https://arxiv.org/abs/${paper.paper_id.slice(6)}`;
        link.target = "_blank"; link.rel = "noopener noreferrer";
        card.append(link);
      }
      if (/^report:[a-f0-9]{32}$/.test(paper.paper_id) && /^https:\/\/(huggingface\.co\/deepseek-ai\/|fe-static\.deepseek\.com\/)/.test(paper.url || "")) {
        const link = node("a", "", "查看官方技术报告");
        link.href = paper.url; link.target = "_blank"; link.rel = "noopener noreferrer";
        card.append(link);
      }
      content.push(card);
    }
    if (report.errors?.length) content.push(node("p", "muted", "本轮中断或降级：" + report.errors.map((error) => `${error.stage} (${error.http_status || error.error})`).join("；")));
    return content;
  });
  updatePart(part("subquestions"), turn.subquestions, () => {
  if (turn.subquestions?.length) {
    const list = node("ul");
    turn.subquestions.forEach((question) => list.append(node("li", "", question)));
    return [node("h3", "", "检索子问题"), list];
  }
  return [];
  });
  if (turn.steps.length && !part("steps").children.length) {
    part("steps").append(node("h3", "", "执行阶段 · 各阶段耗时"), node("ol", "steps"));
  }
  turn.steps.forEach((step, index) => {
    const list = part("steps").querySelector("ol");
    if (!list.children[index]) list.append(node("li"));
    const timing = step.duration_s !== undefined ? `${step.duration_s.toFixed(1)}s` : turn.status === "running" && index === turn.steps.length - 1 ? "进行中" : `第 ${step.elapsed_s.toFixed(1)}s 开始`;
    text(list.children[index], `${step.stage} · ${timing}`);
  });
  updatePart(part("gaps"), turn.missing_aspects, () => {
  if (turn.missing_aspects?.length) {
    const list = node("ul");
    turn.missing_aspects.forEach((gap) => list.append(node("li", "", gap)));
    return [node("h3", "", "仍缺少的证据"), list];
  }
  return [];
  });
  updatePart(part("checks"), [turn.checks, turn.generation_retries, turn.delivery_from_prior_draft], () => {
    const items = [];
  if (turn.checks?.length) {
    items.push(node("h3", "", `${turn.delivery_from_prior_draft ? "保留较早版本的已核验内容" : "交付所用草稿的核验"} · 重新生成 ${turn.generation_retries || 0} 次`));
    for (const check of turn.checks) {
      const item = node("div", "check" + (check.supports ? "" : " rejected"));
      const omitted = {duplicate: "重复说明 · 未纳入回答", unrequested: "超出所问范围 · 未纳入回答"}[check.scope_status];
      item.append(node("div", "check-label", `${omitted || (check.status === "error" ? "判断未完成" : check.supports ? "支持" : "未通过")} · ${check.paper_id}#${check.chunk_id}`), node("p", "", check.claim), node("p", "muted", check.rationale));
      items.push(item);
    }
  }
  return items;
  });
  updatePart(part("run-id"), [turn.id, turn.error_type], () => [node("p", "muted", `运行编号：${turn.id}`),
    ...(turn.error_type ? [node("p", "muted", `错误类型：${turn.error_type}`)] : [])]);
}

function renderTurn(turn) {
  const element = node("article", "turn new-turn");
  element.id = `turn-${turn.id}`;
  element.append(node("div", "user-message", turn.question));
  const heading = node("div", "answer-heading");
  const mark = node("img"); mark.src = "/static/mark.svg"; mark.alt = "";
  heading.append(mark, node("strong", "", "ARA"), node("span", `badge ${turn.status}`, labels[turn.status] || turn.status));
  element.append(heading);
  element.append(node("div", "turn-body"), node("p", "retrieval-progress"));
  const summary = node("div", "run-summary");
  summary.append(node("span", "elapsed"), node("span", "cost"), node("span", "uncertain"));
  element.append(summary, node("div", "papers-slot"), node("div", "sources-slot"), inspection());
  return element;
}

function updateTurn(element, turn) {
  const badge = element.querySelector(".badge");
  badge.className = `badge ${turn.status}`; text(badge, turn.result_kind === "papers" && turn.status === "complete" ? "筛选完成" : labels[turn.status] || turn.status);
  const body = element.querySelector(".turn-body");
  const changed = updatePart(body, [turn.status, turn.answer, turn.status === "running" ? null : turn.stage], () => {
    if (turn.status === "running") return [node("p", "pending-text")];
    return [answerText(turn.answer, turn.sources, turn.id), ...(turn.status !== "complete" ? [node("p", "run-note", turn.stage)] : [])];
  });
  if (turn.status === "running") text(body.querySelector(".pending-text"), `${turn.stage}… ${turn.result_kind === "papers" ? "已取得的候选会先显示，筛选后更新简析。" : "完成核验后显示回答。"}`);
  const progress = turn.retrieval_progress;
  const progressText = progress ? `检索问题 ${progress.subq_index + 1} · 已保留 ${progress.accepted_chunks} 个片段，来自 ${progress.papers} 篇论文` + (progress.retry ? ` · 补查 ${progress.retry} 次` : "") + (progress.stage === "no_new_evidence" ? " · 无新增证据，停止补查" : "") : "";
  text(element.querySelector(".retrieval-progress"), progressText);
  element.querySelector(".retrieval-progress").hidden = !progressText || turn.status !== "running";
  text(element.querySelector(".elapsed"), `${Number(turn.elapsed_s).toFixed(1)} 秒`);
  text(element.querySelector(".cost"), turn.cost_incomplete ? "本轮费用记录未完成，累计费用见共享预算" : `${turn.cost.requests} 次请求 · ${money(turn.cost.estimated_usd)}`);
  text(element.querySelector(".uncertain"), turn.cost.uncertain_usd > 0 ? `另有 ${money(turn.cost.uncertain_usd)} 预留 / 待确认` : "");
  updatePart(element.querySelector(".sources-slot"), turn.sources, () => turn.sources.length ? [sourcePanel(turn)] : []);
  const papersChanged = updatePart(element.querySelector(".papers-slot"), turn.paper_results, () => paperCards(turn.paper_results || []));
  updateInspection(element.querySelector(".inspection"), turn);
  return changed || papersChanged;
}

function renderConversation(reset = false) {
  const scroller = $("#scroll-area");
  const nearBottom = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 100;
  const turns = state.session?.turns || [];
  $("#welcome").hidden = turns.length > 0;
  text($("#conversation-title"), state.session?.title || "新的研究");
  if (reset) $("#messages").replaceChildren();
  let contentChanged = false;
  for (const turn of turns) {
    let element = document.getElementById(`turn-${turn.id}`);
    if (!element) { element = renderTurn(turn); $("#messages").append(element); contentChanged = true; }
    contentChanged = updateTurn(element, turn) || contentChanged;
  }
  if (reset || (nearBottom && contentChanged && !window.getSelection()?.toString())) scroller.scrollTop = scroller.scrollHeight;
  controls();
}

function controls() {
  const running = state.session?.turns.at(-1)?.status === "running";
  const busyElsewhere = state.meta?.active_session && state.meta.active_session !== state.session?.id;
  $("#send").disabled = state.submitting || running || Boolean(busyElsewhere) || !$("#question").value.trim();
  $("#new-chat").disabled = state.submitting;
  document.querySelectorAll(".session").forEach((button) => { button.disabled = state.submitting; });
  $("#stop").hidden = !running;
  $("#activity").classList.toggle("running", Boolean(running));
  text($("#activity"), running ? `${state.session.turns.at(-1).stage} · 你可以展开本轮过程查看进展` : busyElsewhere ? "另一段会话正在研究，可从左侧研究记录打开。" : "");
  for (const role of ["main", "fast"]) $(`#${role}-effort`).disabled = state.submitting || Boolean(state.pending);
}

async function refresh() {
  const version = ++state.refreshVersion;
  const selected = state.session?.id;
  const [meta, sessions, session] = await Promise.all([
    api("/status"), api("/sessions"), selected ? api(`/sessions/${selected}`) : Promise.resolve(null),
  ]);
  if (version !== state.refreshVersion) return;
  state.meta = meta; state.sessions = sessions;
  if (state.connectionLost) { notice(); state.connectionLost = false; }
  if (selected && state.session?.id === selected) state.session = session;
  renderMeta(); renderSessions(); renderConversation();
}

async function selectSession(id) {
  const version = ++state.selectionVersion;
  const session = await api(`/sessions/${id}`);
  if (version !== state.selectionVersion) return;
  state.session = session; state.pending = null;
  localStorage.setItem("ara-session", id);
  $("#question").value = "";
  $("#sidebar").classList.remove("open"); $("#menu").setAttribute("aria-expanded", "false");
  notice(); renderSessions(); renderConversation(true);
}

$("#new-chat").addEventListener("click", () => {
  state.selectionVersion++;
  state.session = null; state.pending = null; localStorage.removeItem("ara-session");
  $("#question").value = ""; $("#sidebar").classList.remove("open"); $("#menu").setAttribute("aria-expanded", "false");
  notice(); renderSessions(); renderConversation(true); $("#question").focus();
});
$("#menu").addEventListener("click", () => $("#menu").setAttribute("aria-expanded", String($("#sidebar").classList.toggle("open"))));
document.querySelectorAll(".starter").forEach((button) => button.addEventListener("click", () => {
  $("#question").value = button.dataset.question; controls(); $("#question").focus();
}));
$("#question").addEventListener("input", controls);
$("#question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing && event.keyCode !== 229) {
    event.preventDefault(); if (!$("#send").disabled) $("#composer").requestSubmit();
  }
});
$("#composer").addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = $("#question").value.trim();
  if (!question || $("#send").disabled) return;
  state.submitting = true; notice(); controls();
  try {
    if (!state.session) {
      state.session = await api("/sessions", {method: "POST"});
      localStorage.setItem("ara-session", state.session.id);
    }
    if (!state.pending || state.pending.question !== question || state.pending.session !== state.session.id) {
      state.pending = {question, request_id: crypto.randomUUID(), session: state.session.id, reasoning: {...state.reasoning}};
    }
    await api(`/sessions/${state.session.id}/messages`, {method: "POST", body: JSON.stringify({question, request_id: state.pending.request_id, reasoning: state.pending.reasoning})});
    state.pending = null; $("#question").value = "";
    await refresh(); $("#scroll-area").scrollTop = $("#scroll-area").scrollHeight;
  } catch (error) { notice(`${error.message} 若发送状态不明确，稍等片刻再刷新查看。`); }
  finally { state.submitting = false; controls(); }
});
$("#stop").addEventListener("click", async () => {
  $("#stop").disabled = true;
  try { await api(`/sessions/${state.session.id}/stop`, {method: "POST"}); await refresh(); }
  catch (error) { notice(error.message); }
  finally { $("#stop").disabled = false; }
});

async function poll() {
  try { await refresh(); }
  catch { state.connectionLost = true; notice("暂时连不上本地服务。恢复连接后会继续显示已有进度，不会自动重新发送问题。"); }
  setTimeout(poll, state.meta?.active_session ? 1200 : 6000);
}

async function initialize() {
  try {
    try {
      const saved = JSON.parse(localStorage.getItem("ara-reasoning"));
      if (saved && [saved.main, saved.fast].every((effort) => ["low", "high", "max"].includes(effort))) {
        state.reasoning = {main: saved.main, fast: saved.fast};
        for (const role of ["main", "fast"]) $(`#${role}-effort`).value = saved[role];
      }
    } catch { /* Invalid preferences fall back to server defaults. */ }
    const id = localStorage.getItem("ara-session");
    if (id) {
      try { await selectSession(id); } catch { localStorage.removeItem("ara-session"); }
    }
    await refresh();
  } catch (error) { notice(error.message); }
  setTimeout(poll, 1200);
}
for (const role of ["main", "fast"]) $(`#${role}-effort`).addEventListener("change", () => {
  state.reasoning = {main: $("#main-effort").value, fast: $("#fast-effort").value};
  localStorage.setItem("ara-reasoning", JSON.stringify(state.reasoning));
  renderReasoning();
});
initialize();
