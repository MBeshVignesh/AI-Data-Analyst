let currentSessionId = null;
let currentController = null;
let activeAssistant = null;
let stopRequested = false;
let allDatasets = [];
let datasetSourceMap = {};

const chatEl = document.getElementById("chat");
const sendBtn = document.getElementById("sendBtn");
const messageInput = document.getElementById("messageInput");
const sidebarFileInput = document.getElementById("sidebarFileInput");
const sidebarUploadBtn = document.getElementById("sidebarUploadBtn");
const sidebarUploadActions = document.getElementById("sidebarUploadActions");
const sidebarUploadFileName = document.getElementById("sidebarUploadFileName");
const sidebarUploadTickBtn = document.getElementById("sidebarUploadTickBtn");
const sidebarUploadCrossBtn = document.getElementById("sidebarUploadCrossBtn");
const sessionSelect = document.getElementById("sessionSelect");
const newSessionBtn = document.getElementById("newSessionBtn");
const renameInput = document.getElementById("renameInput");
const renameBtn = document.getElementById("renameBtn");
const deleteBtn = document.getElementById("deleteBtn");
const datasetSelect = document.getElementById("datasetSelect");
const datasetPreview = document.getElementById("datasetPreview");
const refreshBtn = document.getElementById("refreshBtn");
const sourceFilterBoxes = Array.from(document.querySelectorAll("input.source-filter"));

function getSourceFilter() {
  return sourceFilterBoxes.filter((b) => b.checked).map((b) => b.value);
}

function addMessage(role, content) {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  div.textContent = content;
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
  return div;
}

function addAssistantShell() {
  const div = document.createElement("div");
  div.className = "message assistant";
  const content = document.createElement("div");
  content.className = "assistant-content";
  const code = document.createElement("div");
  code.className = "assistant-code hidden";
  const plotWrap = document.createElement("div");
  plotWrap.className = "assistant-plot hidden";
  const plot = document.createElement("div");
  plot.className = "plot";
  plotWrap.appendChild(plot);
  div.appendChild(content);
  div.appendChild(code);
  div.appendChild(plotWrap);
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
  return { wrapper: div, content, code, plotWrap, plot, plotProgress: null, plotTimer: null };
}

function addAssistantMessageMarkdown(content) {
  const shell = addAssistantShell();
  shell.content.innerHTML = renderMarkdown(content || "");
  return shell;
}

function maybeShowPlotPlaceholder(codeText) {
  if (!activeAssistant) return;
  if (activeAssistant.plotWrap && !activeAssistant.plotWrap.classList.contains("hidden")) return;
  if (/\\bpx\\.|\\bsns\\.|\\bplt\\.|plotly/i.test(codeText || "")) {
    activeAssistant.plotWrap.classList.remove("hidden");
    startPlotProgress(activeAssistant);
  }
}

function startPlotProgress(shell) {
  if (!shell) return;
  if (!shell.plotProgress) {
    shell.plotProgress = document.createElement("div");
    shell.plotProgress.className = "plot-progress";
    shell.plotWrap.appendChild(shell.plotProgress);
  }
  let percent = 0;
  shell.plotProgress.textContent = `Rendering chart ${percent}%`;
  if (shell.plotTimer) clearInterval(shell.plotTimer);
  shell.plotTimer = setInterval(() => {
    percent = Math.min(90, percent + 5);
    shell.plotProgress.textContent = `Rendering chart ${percent}%`;
  }, 400);
}

function finishPlotProgress(shell, text) {
  if (!shell || !shell.plotProgress) return;
  if (shell.plotTimer) {
    clearInterval(shell.plotTimer);
    shell.plotTimer = null;
  }
  shell.plotProgress.textContent = text || "Rendering chart 100%";
  setTimeout(() => {
    if (shell.plotProgress) {
      shell.plotProgress.remove();
      shell.plotProgress = null;
    }
  }, 600);
}

function renderPlotlyWithRetry(plotEl, payload, shell, attempt = 0) {
  const maxAttempts = 3;
  const delay = 600;
  if (typeof Plotly === "undefined") {
    if (attempt < maxAttempts) {
      setTimeout(() => renderPlotlyWithRetry(plotEl, payload, shell, attempt + 1), delay);
      return;
    }
    plotEl.textContent = "Plotly is not available in the browser.";
    if (shell) finishPlotProgress(shell, "Rendering chart 100%");
    return;
  }
  try {
    const result = Plotly.newPlot(plotEl, payload.data, payload.layout || {}, { responsive: true });
    if (result && typeof result.then === "function") {
      result
        .then(() => {
          if (shell) finishPlotProgress(shell, "Rendering chart 100%");
        })
        .catch(() => {
          if (attempt < maxAttempts) {
            setTimeout(() => renderPlotlyWithRetry(plotEl, payload, shell, attempt + 1), delay);
          } else {
            plotEl.textContent = "Plot render failed. Please refresh.";
            if (shell) finishPlotProgress(shell, "Rendering chart 100%");
          }
        });
    } else if (shell) {
      finishPlotProgress(shell, "Rendering chart 100%");
    }
  } catch (err) {
    if (attempt < maxAttempts) {
      setTimeout(() => renderPlotlyWithRetry(plotEl, payload, shell, attempt + 1), delay);
    } else {
      plotEl.textContent = "Plot render failed. Please refresh.";
      if (shell) finishPlotProgress(shell, "Rendering chart 100%");
    }
  }
}

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function renderMarkdown(text) {
  const parts = text.split("```");
  let html = "";
  for (let i = 0; i < parts.length; i++) {
    const part = parts[i];
    if (i % 2 === 0) {
      html += escapeHtml(part).replace(/\n/g, "<br/>");
    } else {
      const firstNewline = part.indexOf("\n");
      const code = firstNewline >= 0 ? part.slice(firstNewline + 1) : part;
      html += `<pre><code>${escapeHtml(code)}</code></pre>`;
    }
  }
  return html;
}

function addPlotMessage() {
  const shell = addAssistantShell();
  shell.plotWrap.classList.remove("hidden");
  return shell.plot;
}

function appendDebug(text) {
  return;
}

async function loadInit() {
  const res = await fetch("/api/init");
  const data = await res.json();
  currentSessionId = data.current_session_id;
  allDatasets = data.datasets || [];
  datasetSourceMap = data.dataset_sources || {};
  renderSessions(data.sessions);
  renderDatasetsForSources();
  if (currentSessionId) {
    const turnsRes = await fetch("/api/sessions/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: currentSessionId })
    });
    const turnsData = await turnsRes.json();
    chatEl.innerHTML = "";
    (turnsData.turns || []).forEach((t) => {
      addMessage("user", t.user);
      const shell = addAssistantShell();
      shell.content.innerHTML = renderMarkdown(t.assistant || "");
      if (t.assistant_code) {
        shell.code.innerHTML = renderMarkdown("```" + t.assistant_code + "```");
        shell.code.classList.remove("hidden");
      }
      if (t.plotly_json_path) {
        fetch(`/api/plots/${encodeURIComponent(t.plotly_json_path)}`)
          .then((r) => r.json())
          .then((payload) => {
            shell.plotWrap.classList.remove("hidden");
            renderPlotlyWithRetry(shell.plot, payload, null);
          })
          .catch(() => {});
      } else if (t.image_path) {
        shell.plotWrap.classList.remove("hidden");
        const img = document.createElement("img");
        img.src = `/api/plots/${encodeURIComponent(t.image_path)}`;
        img.style.maxWidth = "100%";
        shell.plot.appendChild(img);
      }
    });
  }
}

function renderSessions(sessions) {
  sessionSelect.innerHTML = "";
  sessions.forEach((s) => {
    const opt = document.createElement("option");
    opt.value = s.id;
    opt.textContent = `${s.name} (…${s.id.slice(-6)})`;
    if (s.id === currentSessionId) opt.selected = true;
    sessionSelect.appendChild(opt);
  });
}

function renderDatasets(datasets) {
  const previous = datasetSelect.value;
  datasetSelect.innerHTML = "";
  datasets.forEach((d) => {
    const opt = document.createElement("option");
    opt.value = d;
    opt.textContent = d;
    datasetSelect.appendChild(opt);
  });
  if (datasets.length > 0) {
    const selected = datasets.includes(previous) ? previous : datasets[0];
    datasetSelect.value = selected;
    loadDatasetPreview(selected);
    return;
  }
  datasetPreview.textContent = "No datasets available for selected sources.";
}

function getDatasetsForSelectedSources() {
  const selectedSources = new Set(getSourceFilter());
  if (selectedSources.size === 0) return [];
  return allDatasets.filter((name) => selectedSources.has(datasetSourceMap[name] || "base"));
}

function renderDatasetsForSources() {
  renderDatasets(getDatasetsForSelectedSources());
}

async function loadDatasetPreview(name) {
  const res = await fetch(`/api/dataset/${encodeURIComponent(name)}`);
  if (!res.ok) {
    datasetPreview.textContent = "No preview available.";
    return;
  }
  const data = await res.json();
  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  ["Column", "Type", "Non-Null", "Unique"].forEach((h) => {
    const th = document.createElement("th");
    th.textContent = h;
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);
  table.appendChild(thead);
  const tbody = document.createElement("tbody");
  data.columns.forEach((col, i) => {
    const tr = document.createElement("tr");
    [col, data.dtypes[i], data.non_null[i], data.unique[i]].forEach((v) => {
      const td = document.createElement("td");
      td.textContent = v;
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  datasetPreview.innerHTML = "";
  datasetPreview.appendChild(table);
}

sessionSelect.addEventListener("change", async () => {
  const sid = sessionSelect.value;
  const res = await fetch("/api/sessions/select", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id: sid })
  });
  const data = await res.json();
  currentSessionId = sid;
  chatEl.innerHTML = "";
  (data.turns || []).forEach((t) => {
    addMessage("user", t.user);
    addAssistantMessageMarkdown(t.assistant);
  });
});

newSessionBtn.addEventListener("click", async () => {
  const res = await fetch("/api/sessions", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({}) });
  const data = await res.json();
  currentSessionId = data.id;
  const sessions = await (await fetch("/api/sessions")).json();
  renderSessions(sessions);
  chatEl.innerHTML = "";
});

renameBtn.addEventListener("click", async () => {
  const name = renameInput.value.trim();
  if (!name || !currentSessionId) return;
  await fetch(`/api/sessions/${currentSessionId}`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }) });
  const sessions = await (await fetch("/api/sessions")).json();
  renderSessions(sessions);
  renameInput.value = "";
});

deleteBtn.addEventListener("click", async () => {
  if (!currentSessionId) return;
  await fetch(`/api/sessions/${currentSessionId}`, { method: "DELETE" });
  const sessions = await (await fetch("/api/sessions")).json();
  currentSessionId = sessions[0]?.id || null;
  renderSessions(sessions);
  chatEl.innerHTML = "";
});

datasetSelect.addEventListener("change", () => loadDatasetPreview(datasetSelect.value));

sendBtn.addEventListener("click", async () => {
  const message = messageInput.value.trim();
  if (!message) return;
  if (currentController) {
    stopRequested = true;
    currentController.abort();
    currentController = null;
    sendBtn.textContent = "Send";
    return;
  }
  const selectedSources = getSourceFilter();
  if (selectedSources.length === 0) {
    addMessage("assistant", "Select at least one source to run analysis.");
    return;
  }
  addMessage("user", message);
  activeAssistant = addAssistantShell();
  activeAssistant.content.textContent = "Thinking...";
  messageInput.value = "";

  const form = new FormData();
  form.append("message", message);
  form.append("session_id", currentSessionId || "");
  form.append("source_filter", JSON.stringify(selectedSources));

  currentController = new AbortController();
  sendBtn.textContent = "Stop";

  let finished = false;
  try {
    const res = await fetch("/api/chat/stream", { method: "POST", body: form, signal: currentController.signal });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop();
      parts.forEach((p) => {
        if (!p.startsWith("data: ")) return;
        const payload = JSON.parse(p.replace("data: ", ""));
        if (payload.type === "token") {
          if (activeAssistant) activeAssistant.content.textContent = payload.content;
        } else if (payload.type === "final") {
          if (activeAssistant) activeAssistant.content.innerHTML = renderMarkdown(payload.content);
          finished = true;
          if (currentController) currentController.abort();
          currentController = null;
          sendBtn.textContent = "Send";
        } else if (payload.type === "code") {
          if (activeAssistant) {
            activeAssistant.code.innerHTML = renderMarkdown("```" + payload.content + "```");
            activeAssistant.code.classList.remove("hidden");
            maybeShowPlotPlaceholder(payload.content);
          }
        } else if (payload.type === "plotly") {
          const plotEl = activeAssistant ? activeAssistant.plot : addPlotMessage();
          if (activeAssistant) activeAssistant.plotWrap.classList.remove("hidden");
          if (activeAssistant) activeAssistant.plot.textContent = "";
          renderPlotlyWithRetry(plotEl, payload.content, activeAssistant);
          appendDebug("plotly event received");
        } else if (payload.type === "image") {
          const plotEl = activeAssistant ? activeAssistant.plot : addPlotMessage();
          if (activeAssistant) activeAssistant.plotWrap.classList.remove("hidden");
          if (activeAssistant) activeAssistant.plot.textContent = "";
          const img = document.createElement("img");
          img.src = `data:image/png;base64,${payload.content}`;
          img.style.maxWidth = "100%";
          img.onload = () => {
            if (activeAssistant) finishPlotProgress(activeAssistant, "Rendering chart 100%");
          };
          img.onerror = () => {
            plotEl.textContent = "Plot render failed. Please refresh.";
            if (activeAssistant) finishPlotProgress(activeAssistant, "Rendering chart 100%");
          };
          plotEl.appendChild(img);
          appendDebug("image event received");
        } else if (payload.type === "status") {
          appendDebug(payload.content);
        }
      });
    }
  } catch (err) {
    if (activeAssistant && stopRequested && !finished) {
      activeAssistant.content.textContent = "Stopped.";
    }
  }
  stopRequested = false;
  if (!finished) {
    currentController = null;
    sendBtn.textContent = "Send";
  }
});

messageInput.addEventListener("input", async () => {
  const text = messageInput.value.trim();
  if (text.length < 5) return;
  const selectedSources = getSourceFilter();
  await fetch("/api/prefetch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, session_id: currentSessionId, source_filter: selectedSources })
  });
});

messageInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendBtn.click();
  }
});

refreshBtn.addEventListener("click", async () => {
  await loadInit();
});

sidebarFileInput.addEventListener("change", () => {
  const files = Array.from(sidebarFileInput.files);
  if (files.length === 0) {
    sidebarUploadActions.classList.add("hidden");
    return;
  }
  sidebarUploadFileName.textContent = files.map((f) => f.name).join(", ");
  sidebarUploadActions.classList.remove("hidden");
});

sidebarUploadBtn.addEventListener("click", () => {
  sidebarFileInput.click();
});

sidebarUploadTickBtn.addEventListener("click", async () => {
  const files = Array.from(sidebarFileInput.files);
  if (files.length === 0) return;
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  await fetch("/api/upload", { method: "POST", body: form });
  sidebarFileInput.value = "";
  sidebarUploadActions.classList.add("hidden");
  await loadInit();
});

sidebarUploadCrossBtn.addEventListener("click", () => {
  sidebarFileInput.value = "";
  sidebarUploadActions.classList.add("hidden");
});

sourceFilterBoxes.forEach((box) => {
  box.addEventListener("change", () => {
    renderDatasetsForSources();
  });
});

loadInit();
