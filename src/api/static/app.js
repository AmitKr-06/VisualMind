/* ============================================================
   VisualMind · Chat UI logic
   ============================================================ */
const $ = (sel) => document.querySelector(sel);

const messages = $("#messages");
const form = $("#chat-form");
const input = $("#q");
const fileInput = $("#file");
const uploadStatus = $("#upload-status");
const sidInput = $("#sid");

// ---------- Session ----------
function newSessionId() {
  const r = Math.random().toString(36).slice(2, 10);
  return `session-${r}`;
}
let sessionId = newSessionId();
sidInput.value = sessionId;

// ---------- Helpers ----------
function copyToClipboard(text) {
  if (navigator.clipboard && window.isSecureContext) {
    return navigator.clipboard.writeText(text);
  }
  // Fallback for http://localhost
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.style.position = "fixed";
  ta.style.opacity = "0";
  document.body.appendChild(ta);
  ta.select();
  try { document.execCommand("copy"); } catch (e) {}
  document.body.removeChild(ta);
  return Promise.resolve();
}

// ---------- Message rendering ----------
function addMessage(role, text, meta = null) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "U" : "AI";

  // Inner wrapper holds the bubble + actions
  const bubbleWrap = document.createElement("div");
  bubbleWrap.className = "bubble-wrap";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  if (meta) {
    const metaEl = document.createElement("div");
    metaEl.className = "meta";

    if (meta.status) {
      const cls = meta.status === "ok" ? "ok"
                : meta.status === "no_evidence" ? "warn"
                : "err";
      const tag = document.createElement("span");
      tag.className = `tag ${cls}`;
      tag.textContent = meta.status;
      metaEl.appendChild(tag);
    }
    if (meta.confidence !== undefined) {
      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = `conf ${Number(meta.confidence).toFixed(2)}`;
      metaEl.appendChild(tag);
    }
    if (meta.elapsed_s !== undefined) {
      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = `${meta.elapsed_s}s`;
      metaEl.appendChild(tag);
    }
    bubble.appendChild(metaEl);
  }

  if (meta && meta.citations && meta.citations.length) {
    const cite = document.createElement("div");
    cite.className = "citations";
    const unique = [...new Set(meta.citations)];
    cite.innerHTML = "📎 " + unique.map((c) => `<code>${c}</code>`).join("");
    bubble.appendChild(cite);
  }

  // ---------- Action buttons (Copy / Edit) ----------
  const actions = document.createElement("div");
  actions.className = "bubble-actions";

  const copyBtn = document.createElement("button");
  copyBtn.type = "button";
  copyBtn.textContent = "📋 Copy";
  copyBtn.addEventListener("click", async () => {
    await copyToClipboard(text);
    copyBtn.textContent = "✓ Copied";
    copyBtn.classList.add("copied");
    setTimeout(() => {
      copyBtn.textContent = "📋 Copy";
      copyBtn.classList.remove("copied");
    }, 1500);
  });
  actions.appendChild(copyBtn);

  if (role === "user") {
    const editBtn = document.createElement("button");
    editBtn.type = "button";
    editBtn.textContent = "✏️ Edit";
    editBtn.addEventListener("click", () => {
      input.value = text;
      input.focus();
      // Move cursor to end
      input.setSelectionRange(text.length, text.length);
    });
    actions.appendChild(editBtn);
  }

  bubbleWrap.appendChild(bubble);
  bubbleWrap.appendChild(actions);

  wrap.appendChild(avatar);
  wrap.appendChild(bubbleWrap);
  messages.appendChild(wrap);
  messages.scrollTop = messages.scrollHeight;
}

// ---------- Chat ----------
async function ask(question) {
  addMessage("user", question);

  const submitBtn = form.querySelector('button[type="submit"]');
  submitBtn.disabled = true;

  const thinking = document.createElement("div");
  thinking.className = "msg ai";
  thinking.innerHTML = `<div class="avatar">AI</div><div class="bubble">Thinking…</div>`;
  messages.appendChild(thinking);
  messages.scrollTop = messages.scrollHeight;

  try {
    const resp = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        session_id: sessionId,
      }),
    });
    const data = await resp.json();
    thinking.remove();
    addMessage("ai", data.answer || "(no answer)", {
      status: data.status,
      confidence: data.confidence,
      elapsed_s: data.elapsed_s,
      citations: data.citations,
    });
  } catch (e) {
    thinking.remove();
    addMessage("ai", `Error: ${e.message}`, { status: "error" });
  } finally {
    submitBtn.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = input.value.trim();
  if (!q) return;
  input.value = "";
  ask(q);
});

// ---------- Upload ----------
fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  if (!file) return;

  uploadStatus.className = "upload-status";
  uploadStatus.textContent = `Uploading ${file.name}…`;

  const formData = new FormData();
  formData.append("file", file);
  formData.append("session_id", sessionId);

  try {
    const resp = await fetch("/upload", {
      method: "POST",
      body: formData,
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || "Upload failed");
    }
    const data = await resp.json();
    uploadStatus.className = "upload-status success";
    uploadStatus.textContent =
      `✓ ${data.filename} · ${data.chunks_added} chunks added (total: ${data.total_chunks})`;

    addMessage(
      "ai",
      `Uploaded "${data.filename}" — ${data.chunks_added} chunks are now searchable in this session. Ask me anything about it.`
    );
  } catch (e) {
    uploadStatus.className = "upload-status error";
    uploadStatus.textContent = `✗ ${e.message}`;
  } finally {
    fileInput.value = "";
  }
});

// ---------- Session actions ----------
$("#clear-chat").addEventListener("click", () => {
  messages.innerHTML = "";
});

$("#clear-session").addEventListener("click", async () => {
  if (!confirm("Delete this session and its uploaded files?")) return;
  try {
    await fetch(`/sessions/${sessionId}`, { method: "DELETE" });
    messages.innerHTML = "";
    sessionId = newSessionId();
    sidInput.value = sessionId;
    uploadStatus.className = "upload-status";
    uploadStatus.textContent = "New session started";
  } catch (e) {
    alert(`Could not delete: ${e.message}`);
  }
});

// ---------- Welcome ----------
addMessage(
  "ai",
  "Hi! Click the 📎 to attach or upload a PDF or image, then ask me anything about it. Without an upload, I'll answer from the default NCERT corpus (Life Processes + Electricity)."
);