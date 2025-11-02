import html
from typing import List

from fetchcraft.agents.model import Citation

import html, uuid

uid = f"owui-citations-{uuid.uuid4().hex[:8]}"

def _fmt_score(score):
    try:
        return f"{float(score):.2f}"
    except Exception:
        return html.escape(str(score))

def render_citations(citations, comp_id: str):
    if not citations:
        return f"<p class='muted'><em>Citations (0)</em></p>"

    items = []
    for i, c in enumerate(citations, 1):
        score = _fmt_score(getattr(c, "score", ""))
        filename = html.escape(getattr(c, "filename", ""))
        source = html.escape(getattr(c, "source", ""))
        preview = html.escape(getattr(c, "text_preview", ""))

        items.append(f"""
        <details class="cite" data-index="{i}">
          <summary>
            <span class="arrow" aria-hidden="true"></span>
            <b>#{i}</b>&nbsp;Score: {score}&nbsp;–&nbsp;{filename}
          </summary>
          <div class="cite-body">
            <p class="preview">{preview}</p>
            <p class="link">
              <a href="http://wingman.akhbar.home/files/{source}" target="_blank" rel="noopener">
                Open document
              </a>
            </p>
          </div>
        </details>
        """)

    return f"""
    <div id="{comp_id}" class="citations">
      <div class="citations-head">
        <strong>Citations ({len(citations)}):</strong>
        <div class="spacer"></div>
        <button type="button" class="btn btn-sm" data-role="toggle">Expand all</button>
      </div>
      <div class="citations-list">
        {''.join(items)}
      </div>
    </div>
    """


def render_response(answer: str, processing_time: float, llm_model: str, citations: List[Citation]) -> str:
    citations_html = render_citations(citations, uid)

    html_output = f"""
    <div class="owui-card" style="font-family:'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                                  color:var(--color-gray-100,#ececec);
                                  background-color:var(--color-stone-800,#1f2937);
                                  border:1px solid rgba(255,255,255,0.08);
                                  border-radius:14px;padding:16px; max-height: 400px">
      <div dir="auto">
        <p>{html.escape(answer)}</p>

        {citations_html}

        <p class="meta">
          <span>Processing Time: {processing_time} ms</span>
          &nbsp;•&nbsp;
          <span>Model: {html.escape(llm_model)}</span>
        </p>

        <button onclick="window.location.href='/docs'" class="btn">Back to API Docs</button>
      </div>
    </div>

    <style>
      .owui-card .muted {{ opacity: 0.7; }}
      .owui-card .meta {{ opacity: 0.75; margin-top: 12px; font-size: 0.9rem; }}
      .owui-card .btn {{
        background: transparent; color: inherit; border: 1px solid rgba(255,255,255,0.25);
        border-radius: 10px; padding: 6px 10px; cursor: pointer;
      }}
      .owui-card .btn:hover {{ border-color: rgba(255,255,255,0.4); }}
      .owui-card .btn-sm {{ padding: 4px 8px; font-size: 0.85rem; }}

      .owui-card .citations {{ margin-top: 12px; }}
      .owui-card .citations-head {{
        display: flex; align-items: center; gap: 10px; margin-bottom: 8px;
      }}
      .owui-card .citations-head .spacer {{ flex: 1; }}

      .owui-card .citations-list {{ display: grid; gap: 6px; }}

      .owui-card details.cite {{
        border-left: 2px solid rgba(255,255,255,0.18);
        padding-left: 10px; border-radius: 2px;
      }}
      .owui-card details.cite > summary {{
        list-style: none; cursor: pointer; display: flex; align-items: center; gap: 8px;
        font-weight: 600;
      }}
      .owui-card details.cite > summary::-webkit-details-marker {{ display: none; }}
      .owui-card .cite-body {{ margin: 6px 0 4px 18px; }}
      .owui-card .preview {{ white-space: pre-wrap; margin: 6px 0; line-height: 1.4; }}
      .owui-card .link a {{ text-decoration: underline; }}

      .owui-card .arrow::before {{ content: "▸"; display: inline-block; transition: transform 0.15s ease; }}
      .owui-card details[open] > summary .arrow::before {{ content: "▾"; }}

      @media (prefers-color-scheme: light) {{
        .owui-card {{ color:#0f172a; background-color:#f8fafc; border-color:rgba(2,6,23,0.08); }}
        .owui-card .btn {{ border-color: rgba(2,6,23,0.2); }}
      }}
    </style>

    <script>
    (function(id){{
      function init(){{
        var root = document.getElementById(id);
        if(!root) return;

        var toggleBtn = root.querySelector('[data-role="toggle"]');
        if(!toggleBtn) return;

        function items(){{ return Array.prototype.slice.call(root.querySelectorAll('details.cite')); }}

        toggleBtn.addEventListener('click', function(){{
          var ds = items();
          if(!ds.length) return;
          // Open if any is closed; otherwise close all
          var shouldOpen = ds.some(function(d){{ return !d.open; }});
          ds.forEach(function(d){{ d.open = shouldOpen; }});
          toggleBtn.textContent = shouldOpen ? 'Collapse all' : 'Expand all';
        }});
      }}

      if(document.readyState === 'loading'){{
        document.addEventListener('DOMContentLoaded', init);
      }} else {{
        setTimeout(init, 0);
      }}
    }})('{uid}');
    </script>
    """

    return html_output