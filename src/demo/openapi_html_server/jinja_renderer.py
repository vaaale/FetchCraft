# renderer.py
from __future__ import annotations

import uuid
from pathlib import Path
from typing import List, Any, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

# ---- Config ----
TEMPLATES_DIR = Path(__file__).parent / "templates"
FILES_BASE_URL = "http://wingman.akhbar.home/files"   # adjust if needed

# ---- Jinja environment ----
env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(["html", "xml"]),
    trim_blocks=True,
    lstrip_blocks=True,
)

def fmt_score(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except Exception:
        return str(value)

env.filters["fmt_score"] = fmt_score


def render_response(
    answer: str,
    processing_time: float,
    llm_model: str,
    citations: List[Any],
    *,
    heading: Optional[str] = None,
    answer_is_html: bool = False,
    files_base_url: str = FILES_BASE_URL,
) -> str:
    """
    Renders the response HTML using Jinja templates.

    - heading: optional title shown above the answer
    - answer_is_html: if True, renders answer as safe HTML; otherwise escapes it.
    - files_base_url: base URL for citation links.
    """
    uid = f"owui-citations-{uuid.uuid4().hex[:8]}"
    template = env.get_template("response.html")

    return template.render(
        uid=uid,
        heading=heading,
        answer=answer,
        answer_is_html=answer_is_html,
        processing_time=processing_time,
        llm_model=llm_model,
        citations=citations,
        files_base_url=files_base_url,
    )
