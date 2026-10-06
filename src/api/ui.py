"""
HTML/CSS chat UI — served at /.

Uses Jinja-less string templates for zero dependencies.
Assets (CSS, JS) are served from src/api/static/.
"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["ui"])


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> str:
    """Serve the chat UI at /."""
    html = (INDEX_HTML
            .replace("{{TITLE}}", "VisualMind")
            .replace("{{SUBTITLE}}", "Ask anything about your study material"))
    return html


# ============================================================
# HTML (inline template)
# ============================================================
INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{{TITLE}} · {{SUBTITLE}}</title>
  <link rel="stylesheet" href="/static/style.css" />
</head>
<body>
  <div class="app">
    <!-- Sidebar -->
    <aside class="sidebar">
      <div class="logo">
        <span class="dot"></span>
        <span>{{TITLE}}</span>
      </div>

      <div class="session-box">
        <label for="sid">Session</label>
        <input id="sid" type="text" placeholder="auto-generated" readonly />
      </div>

      <div class="session-actions">
        <button id="clear-session" class="ghost">Delete session</button>
        <button id="clear-chat" class="ghost">Clear chat</button>
      </div>

      <footer class="sidebar-foot">
        <a href="/docs" target="_blank">API docs</a>
        <a href="/health" target="_blank">Health</a>
        <a href="/config" target="_blank">Config</a>
      </footer>
    </aside>

    <!-- Chat -->
    <main class="chat">
      <header class="chat-header">
        <h1>Ask your notes</h1>
        <p>Answers with citations from your uploaded PDFs or the default NCERT corpus.</p>
      </header>

      <div id="messages" class="messages"></div>

      <form id="chat-form" class="composer">
        <label for="file" class="composer-attach" title="Upload PDF or image">
          <span>📎</span>
        </label>
        <input id="file" type="file" accept="application/pdf,image/*" hidden />
        <input id="q" type="text" placeholder="Ask something, or attach a PDF…" autocomplete="off" required />
        <button type="submit" aria-label="Send">➤</button>
      </form>

      <div id="upload-status" class="upload-status"></div>
    </main>
  </div>

  <script src="/static/app.js"></script>
</body>
</html>
"""