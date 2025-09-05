# app.py
import os
import time
import json
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Union

import streamlit as st
from dotenv import load_dotenv

# External module from your project
from hybrid_rag import setup_hybrid_rag, query_hybrid_rag

# -----------------------------------------------------------------------------
# Page Config & Global Styles
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Hybrid RAG System",
    page_icon="🌎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Global CSS (clean, modern, responsive) ----
st.markdown(
    """
    <style>
    
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

      :root {
        --bg-grad-start: #0f172a;  /* slate-900 */
        --bg-grad-mid:   #111827;  /* gray-900 */
        --bg-grad-end:   #020617;  /* slate-950 */
        --card-bg: rgba(255,255,255,0.04);
        --card-border: rgba(255,255,255,0.08);
        --glass: rgba(255,255,255,0.06);
      }

      html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
      }

      /* Full app gradient background */
      [data-testid="stAppViewContainer"] {
        background: linear-gradient(to bottom, #FDEBF7, #FCE7F3);
      }

      /* Hide default Streamlit chrome */
      #MainMenu { visibility: hidden; }
      header { visibility: hidden; }
      footer { visibility: hidden; }

      /* Card (glassmorphism) */
      .glass {
        background: var(--glass);
        border: 1px solid var(--card-border);
        box-shadow: 0 10px 30px rgba(2,6,23,0.5);
        backdrop-filter: blur(12px);
        border-radius: 18px;
        padding: 1.25rem;
      }

      .hero-title {
        font-size: clamp(26px, 3.2vw, 44px);
        font-weight: 800;
        line-height: 1.1;
        letter-spacing: -0.02em;
        margin: 0;
        background: linear-gradient(90deg, #e5e7eb 0%, #93c5fd 40%, #a78bfa 70%, #f472b6 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
      }

      .hero-sub {
    color: #000000;  /* dark black */
    font-size: clamp(14px, 1.3vw, 18px);
    margin-top: 8px;
}


      .logo-band {
        display: flex;
        gap: 12px;
        align-items: center;
        justify-content: flex-end;
      }

      .pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 12px;
        border-radius: 9999px;
        border: 1px solid var(--card-border);
        background: var(--card-bg);
        color: #e5e7eb;
        font-size: 12px;
        font-weight: 600;
      }

      .prompt-chip {
        display: inline-block;
        padding: 8px 14px;
        margin: 6px 8px 0 0;
        border-radius: 9999px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        color: #e5e7eb;
        font-size: 13px;
        cursor: pointer;
        user-select: none;
        transition: all .15s ease;
      }
      .prompt-chip:hover {
        transform: translateY(-1px);
        background: rgba(255,255,255,0.10);
      }

      /* Chat bubbles */
      .bubble {
        padding: 14px 16px;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.10);
        margin: 8px 0;
        word-wrap: break-word;
        white-space: pre-wrap;
      }
      .bubble-user {
        background: rgba(59,130,246,0.12);
        border-color: rgba(59,130,246,0.35);
      }
      .bubble-assistant {
        background: rgba(34,197,94,0.10);
        border-color: rgba(34,197,94,0.30);
      }
      .mono-box {
        background: rgba(2, 6, 23, 0.65);
        border: 1px solid rgba(148,163,184,0.25);
        border-radius: 12px;
        padding: 10px 12px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size: 12px;
        color: #e5e7eb;
        overflow-x: auto;
      }

      /* Sidebar */
      section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, rgba(2,6,23,0.85), rgba(2,6,23,0.92));
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Environment & Setup
# -----------------------------------------------------------------------------
load_dotenv()

REQUIRED_ENV = [
    "GROQ_API_KEY",
    "LLAMA_CLOUD_API_KEY",
    "LLAMA_CLOUD_INDEX_NAME",
    "LLAMA_CLOUD_PROJECT_NAME",
    "LLAMA_CLOUD_ORG_ID",
]

missing_env = [k for k in REQUIRED_ENV if not os.getenv(k)]

assets_path = Path("assets")
groq_logo_path = assets_path / "groq_logo.png"
llamacloud_logo_path = assets_path / "llamacloud_logo.png"

# -----------------------------------------------------------------------------
# Header / Hero
# -----------------------------------------------------------------------------
left, right = st.columns([5, 2], vertical_alignment="center")
with left:
    st.markdown("<h1 class='hero-title'>Hybrid RAG System: SQL + Document Retrieval</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='hero-sub'>
          Ask questions about US cities using a hybrid retrieval pipeline that blends
          structured SQL over city data with unstructured document search from LlamaCloud.
          Supports cities like New York City, Los Angeles, Chicago, Houston, Miami, and Seattle.
        </div>
        """,
        unsafe_allow_html=True,
    )
with right:
    band = []
    if groq_logo_path.exists():
        band.append(st.image(str(groq_logo_path), width=120))
    if llamacloud_logo_path.exists():
        band.append(st.image(str(llamacloud_logo_path), width=120))
    if not band:
        st.markdown(
            """
            <div class='logo-band'>
              <span class='pill'>GROQ</span>
              <span class='pill'>LlamaCloud</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

# -----------------------------------------------------------------------------
# Guard: Show setup help if env is missing
# -----------------------------------------------------------------------------
if missing_env:
    st.error("Missing required environment variables.")
    with st.expander("How to fix (create a .env file)", expanded=True):
        st.code(
            "GROQ_API_KEY=your_groq_api_key\n"
            "LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key\n"
            "LLAMA_CLOUD_INDEX_NAME=your_index_name\n"
            "LLAMA_CLOUD_PROJECT_NAME=your_project_name\n"
            "LLAMA_CLOUD_ORG_ID=your_organization_id\n",
            language="bash",
        )
        st.markdown(
            "- Restart the app after adding the file.\n"
            "- Ensure your LlamaCloud index contains the city PDFs."
        )
    st.stop()

# -----------------------------------------------------------------------------
# Cached Setup & Session State
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _init_workflow_sync():
    # Wrap the async initializer in a sync function and cache it.
    return asyncio.run(setup_hybrid_rag())

if "workflow" not in st.session_state:
    with st.spinner("Booting up the hybrid RAG pipeline…"):
        st.session_state.workflow = _init_workflow_sync()

if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, str]] = []

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _render_message(role: str, content: str, avatar: str = ""):
    cls = "bubble-assistant" if role == "assistant" else "bubble-user"
    with st.chat_message(role, avatar=avatar):
        st.markdown(f"<div class='bubble {cls}'>{content}</div>", unsafe_allow_html=True)

def _normalize_result(result: Any) -> Dict[str, Any]:
    """
    Support both:
      - string answer
      - dict with 'answer', optional 'sources', optional 'sql', optional 'rows'
    """
    if isinstance(result, str):
        return {"answer": result}
    if isinstance(result, dict):
        # Keep known fields only; avoid leaking unexpected objects
        normalized = {"answer": result.get("answer") or result.get("result") or ""}
        if "sources" in result and isinstance(result["sources"], list):
            normalized["sources"] = result["sources"]
        if "sql" in result:
            normalized["sql"] = result["sql"]
        if "rows" in result:
            normalized["rows"] = result["rows"]
        return normalized
    # Fallback
    return {"answer": str(result)}

def _render_rich_answer(payload: Dict[str, Any]):
    # Main answer
    st.markdown(payload.get("answer", ""))

    # Optional SQL (debug/trace)
    if "sql" in payload and payload["sql"]:
        with st.expander("SQL used (debug)"):
            st.code(str(payload["sql"]), language="sql")

    # Optional Rows (structured results)
    if "rows" in payload and payload["rows"]:
        with st.expander("Structured results"):
            # Try a pretty table if rows look tabular
            try:
                import pandas as pd
                df = pd.DataFrame(payload["rows"])
                st.dataframe(df, use_container_width=True, hide_index=True)
            except Exception:
                st.write(payload["rows"])

    # Optional Sources (document retrieval)
    if "sources" in payload and payload["sources"]:
        with st.expander("Sources & citations"):
            for i, src in enumerate(payload["sources"], start=1):
                title = src.get("title") or f"Source {i}"
                url = src.get("url") or ""
                score = src.get("score")
                snippet = src.get("snippet") or src.get("text") or ""
                meta = f" **(score: {score:.3f})**" if isinstance(score, (int, float)) else ""
                st.markdown(f"**{i}. {title}**{meta}")
                if url:
                    st.markdown(f"- Link: {url}")
                if snippet:
                    st.markdown(f"> {snippet}")

def _process_query(query_text: str):
    # Append & display user
    st.session_state.messages.append({"role": "user", "content": query_text})
    _render_message("user", query_text, avatar="🧑‍💻")

    # Call pipeline
    start = time.time()
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking…"):
            try:
                raw = asyncio.run(query_hybrid_rag(st.session_state.workflow, query_text))
                payload = _normalize_result(raw)
                _render_rich_answer(payload)
                answer_text = payload.get("answer", "")
            except Exception as e:
                st.error(f"Error during query: {e}")
                answer_text = "Sorry, something went wrong while processing your query."

            elapsed = time.time() - start
            st.caption(f"Responded in {elapsed:.2f}s")

    # Store assistant message for future render
    st.session_state.messages.append({"role": "assistant", "content": answer_text})

# -----------------------------------------------------------------------------
# Prompt Chips
# -----------------------------------------------------------------------------
EXAMPLE_QUERIES = [
    "Which city has the highest population?",
    "What state is Houston located in?",
    "Where is the Space Needle located?",
    "List places to visit in Miami.",
    "How do people in Chicago get around?",
    "What is the historical name of Los Angeles?",
]


chips_cols = st.columns(3)
for idx, q in enumerate(EXAMPLE_QUERIES):
    with chips_cols[idx % 3]:
        if st.button(q, use_container_width=True):
            _process_query(q)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Conversation (History Render)
# -----------------------------------------------------------------------------
# Replay chat history (so redesigned bubbles persist across reruns)
for m in st.session_state.messages:
    _render_message(
        m["role"],
        m["content"],
        avatar=("🧑‍💻" if m["role"] == "user" else "🤖")
    )

# -----------------------------------------------------------------------------
# Chat Input
# -----------------------------------------------------------------------------
user_query = st.chat_input("Ask a question about the cities…")
if user_query:
    _process_query(user_query)

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    
    st.markdown("### Status")
    ok = "✅"
    bad = "❌"
    for k in REQUIRED_ENV:
        st.write(f"{ok if os.getenv(k) else bad} `{k}`")

    st.divider()
    st.markdown("### Quick Actions")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    with col_b:
        if st.button("Reinitialize"):
            # Clear cached setup and reinit
            _init_workflow_sync.clear()
            try:
                with st.spinner("Reinitializing pipeline…"):
                    st.session_state.workflow = _init_workflow_sync()
            except Exception as e:
                st.error(f"Failed to reinitialize: {e}")

    st.divider()
    st.markdown("### Download")
    # Export transcript (simple Markdown)
    if st.session_state.messages:
        transcript_md = []
        for m in st.session_state.messages:
            prefix = "**You:**" if m["role"] == "user" else "**Assistant:**"
            transcript_md.append(f"{prefix}\n\n{m['content']}\n")
        md_blob = "\n---\n".join(transcript_md)
        st.download_button(
            label="Download Transcript (.md)",
            data=md_blob.encode("utf-8"),
            file_name="hybrid_rag_transcript.md",
            mime="text/markdown",
            use_container_width=True
        )

    st.divider()
    st.markdown(
        """
        ### Notes
        - Hybrid pipeline: SQL over structured city data + LlamaCloud document retrieval.
        - Ensure your LlamaCloud index contains the Wikipedia PDFs for target cities.
        """
    )
