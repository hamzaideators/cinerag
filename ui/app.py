import os
import re
import json
import hashlib
import requests
import streamlit as st
from datetime import datetime

API_URL = os.getenv("API_URL", "http://localhost:8000")
DOCS_PATH = os.getenv("DOCS_PATH", "movies_docs.json")

st.set_page_config(page_title="CineRAG", page_icon="🎬", layout="wide")

# Initialize session state for feedback tracking
if "feedback_history" not in st.session_state:
    st.session_state.feedback_history = []
if "current_answer" not in st.session_state:
    st.session_state.current_answer = None
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = {}

# -------- Helpers
def parse_tmdb_id(tmdb_id: str):
    # "tmdb:movie:19995" -> ("movie", "19995")
    m = re.match(r"tmdb:(\w+):(\d+)", tmdb_id or "")
    if not m:
        return None, None
    return m.group(1), m.group(2)

def tmdb_url_from_id(tmdb_id: str):
    kind, num = parse_tmdb_id(tmdb_id)
    if not num:
        return None
    # kind is usually "movie" in this project
    return f"https://www.themoviedb.org/{kind}/{num}"

@st.cache_resource(show_spinner=False)
def load_movies_index(path: str):
    try:
        with open(path, "r") as f:
            docs = json.load(f)
    except Exception:
        return {}
    idx = {}
    for d in docs:
        tmdb_id = d.get("id") or d.get("tmdb_id")
        if not tmdb_id:
            continue
        idx[tmdb_id] = {
            "title": d.get("title"),
            "year": d.get("year"),
            "url": d.get("tmdb_url") or tmdb_url_from_id(tmdb_id),
            "poster_url": d.get("poster_url")
        }
    return idx

MOVIES_INDEX = load_movies_index(DOCS_PATH)

def human_label_from_id(tmdb_id: str):
    meta = MOVIES_INDEX.get(tmdb_id) or {}
    title = meta.get("title") or "Unknown title"
    year = meta.get("year")
    url = meta.get("url") or tmdb_url_from_id(tmdb_id)
    label = f"{title} ({year})" if year else title
    return label, url

def humanize_answer(answer_text: str):
    if not answer_text:
        return answer_text
    pattern = re.compile(r"tmdb:\\w+:\\d+")
    def repl(m):
        tmdb_id = m.group(0)
        label, url = human_label_from_id(tmdb_id)
        return f"**[{label}]({url})**" if url else f"**{label}**"
    return pattern.sub(repl, answer_text)

def render_citation_card(c):
    tmdb_id = c.get("tmdb_id")
    # Prefer API-provided values, fall back to our local index
    idx_meta = MOVIES_INDEX.get(tmdb_id) or {}
    title = c.get("title") or idx_meta.get("title") or "Unknown title"
    year = c.get("year") or idx_meta.get("year")
    url = c.get("url") or idx_meta.get("url") or tmdb_url_from_id(tmdb_id)
    left, right = st.columns([0.85, 0.15])
    with left:
        line = f"**{title}**"
        if year:
            line += f" ({year})"
        st.markdown(line)
        if url:
            st.markdown(f"[TMDB link]({url})")
    with right:
        st.caption(tmdb_id or "")

def call_api(payload):
    r = requests.post(f"{API_URL}/ask", json=payload, timeout=90)
    r.raise_for_status()
    return r.json()

def get_answer_key(query: str, answer: str) -> str:
    """Generate unique key for query+answer pair to track feedback"""
    content = f"{query}::{answer}"
    return hashlib.md5(content.encode()).hexdigest()[:12]

def send_feedback(query: str, answer: str, citations: list, thumb: str, comment: str = "") -> tuple:
    """
    Send feedback to API and return (success: bool, message: str)
    """
    try:
        response = requests.post(
            f"{API_URL}/feedback",
            json={
                "query": query,
                "answer": answer,
                "citations": citations,
                "thumb": thumb,
                "comment": comment
            },
            timeout=15
        )
        response.raise_for_status()
        return True, "Feedback sent successfully!"
    except Exception as e:
        return False, f"Could not send feedback: {e}"

def render_feedback_section(query: str, answer: str, citations: list):
    """
    Render feedback UI with state management, comments, and history
    """
    answer_key = get_answer_key(query, answer)
    feedback_status = st.session_state.feedback_given.get(answer_key)

    st.divider()
    st.subheader("Was this helpful?")

    # If feedback already given, show status
    if feedback_status:
        thumb = feedback_status["thumb"]
        comment = feedback_status.get("comment", "")
        timestamp = feedback_status.get("timestamp", "")

        if thumb == "up":
            st.success(f"✓ You gave positive feedback {timestamp}")
        else:
            st.info(f"✓ You gave negative feedback {timestamp}")
            if comment:
                st.caption(f"Your comment: *\"{comment}\"*")

        if st.button("↻ Change feedback", key=f"change_{answer_key}"):
            # Clear feedback status to allow re-submission
            del st.session_state.feedback_given[answer_key]
            st.rerun()
        return

    # Show feedback buttons
    citation_ids = [c.get("tmdb_id") for c in citations]

    col1, col2 = st.columns(2)

    with col1:
        if st.button("👍 Yes, helpful", key=f"up_{answer_key}", type="primary", use_container_width=True):
            success, message = send_feedback(query, answer, citation_ids, "up", "")

            if success:
                # Store feedback in session state
                st.session_state.feedback_given[answer_key] = {
                    "thumb": "up",
                    "comment": "",
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                }
                st.session_state.feedback_history.append({
                    "query": query,
                    "thumb": "up",
                    "comment": "",
                    "timestamp": datetime.now().isoformat()
                })
                st.rerun()
            else:
                st.error(message)

    with col2:
        if st.button("👎 Not quite", key=f"down_{answer_key}", use_container_width=True):
            # Show comment field for negative feedback
            st.session_state[f"show_comment_{answer_key}"] = True
            st.rerun()

    # Show comment field if thumbs down was clicked
    if st.session_state.get(f"show_comment_{answer_key}", False):
        st.markdown("**Help us improve:** What was wrong with this answer?")
        comment = st.text_area(
            "Optional comment",
            placeholder="e.g., Wrong movie, missing information, inaccurate details...",
            key=f"comment_{answer_key}",
            label_visibility="collapsed"
        )

        col_submit, col_cancel = st.columns([1, 1])
        with col_submit:
            if st.button("Submit negative feedback", key=f"submit_down_{answer_key}", type="primary"):
                success, message = send_feedback(query, answer, citation_ids, "down", comment)

                if success:
                    st.session_state.feedback_given[answer_key] = {
                        "thumb": "down",
                        "comment": comment,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    }
                    st.session_state.feedback_history.append({
                        "query": query,
                        "thumb": "down",
                        "comment": comment,
                        "timestamp": datetime.now().isoformat()
                    })
                    # Clear the comment flag
                    del st.session_state[f"show_comment_{answer_key}"]
                    st.rerun()
                else:
                    st.error(message)

        with col_cancel:
            if st.button("Cancel", key=f"cancel_{answer_key}"):
                del st.session_state[f"show_comment_{answer_key}"]
                st.rerun()

# -------- Sidebar
with st.sidebar:
    st.header("Settings")
    backend = st.selectbox("Retrieval backend",
                           ["auto", "hybrid_rerank", "hybrid", "qdrant", "es"],
                           index=0)
    provider_display = st.selectbox("LLM Provider",
                                    ["auto (from .env)", "openai", "anthropic", "vllm"],
                                    index=0)
    # Map display value to API value
    provider = None if provider_display.startswith("auto") else provider_display

    top_k = st.slider("Top K", 3, 20, 7)
    year_from, year_to = st.slider("Year range", 1950, 2025, (1990, 2020))
    genres_str = st.text_input("Genres (comma-separated)", value="")
    st.caption('Tip: try queries like *"blue aliens on a moon called Pandora"*')

# -------- Main
st.title("🎬 CineRAG — Movie Finder")
query = st.text_input("Describe the movie you’re thinking of",
                      value="blue aliens on Pandora with human avatars")

if st.button("Search", type="primary"):
    with st.spinner("Thinking…"):
        payload = {
            "query": query,
            "top_k": top_k,
            "backend": backend,
            "provider": provider,
            "year": [year_from, year_to],
            "genres": [g.strip() for g in genres_str.split(",") if g.strip()]
        }
        try:
            data = call_api(payload)
            raw_answer = data.get("answer") or "I'm not sure based on the context I have."
            answer = humanize_answer(raw_answer)
            citations = data.get("citations") or []
            retrieved = data.get("retrieved") or []

            # ---- Store results in session state
            st.session_state.current_answer = {
                "query": query,
                "answer": answer,
                "citations": citations,
                "retrieved": retrieved,
                "backend": data.get("backend"),
                "raw_data": data
            }

        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.session_state.current_answer = None

# ---- Display results from session state
if st.session_state.current_answer:
    result = st.session_state.current_answer
    answer = result["answer"]
    citations = result["citations"]

    # ---- Answer Section
    st.subheader("🧠 Our Recommendation")
    st.write(answer)

    # ---- Recommended Movies Grid
    if citations:
        st.divider()
        st.subheader("🎬 Recommended Movies")

        # Add CSS for equal height movie cards
        st.markdown("""
            <style>
            .movie-poster {
                width: 100%;
                aspect-ratio: 2/3;
                object-fit: cover;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            }
            .movie-title {
                margin-top: 8px;
                text-align: center;
                min-height: 50px;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            </style>
        """, unsafe_allow_html=True)

        # Create grid with 5 columns per row
        cols_per_row = 5

        # Display posters in rows
        for row_start in range(0, len(citations), cols_per_row):
            cols = st.columns(cols_per_row)

            for idx, c in enumerate(citations[row_start:row_start + cols_per_row]):
                with cols[idx]:
                    tmdb_id = c.get("tmdb_id")
                    idx_meta = MOVIES_INDEX.get(tmdb_id) or {}
                    title = c.get("title") or idx_meta.get("title") or "Unknown"
                    year = c.get("year") or idx_meta.get("year")
                    url = c.get("url") or idx_meta.get("url") or tmdb_url_from_id(tmdb_id)
                    poster_url = idx_meta.get("poster_url")

                    # Display poster or placeholder
                    if poster_url:
                        st.markdown(f'<img src="{poster_url}" class="movie-poster" alt="{title}">', unsafe_allow_html=True)
                    else:
                        # Fallback: show title card
                        st.info(f"**{title}**")

                    # Title and year below poster
                    year_str = f" ({year})" if year else ""
                    title_display = f"**{title}**{year_str}"

                    if url:
                        st.markdown(f'<div class="movie-title"><a href="{url}" target="_blank">{title}{year_str}</a></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="movie-title">{title}{year_str}</div>', unsafe_allow_html=True)

    # ---- Debug: Raw JSON
    with st.expander("🔍 See raw API response (debug)"):
        st.json(result.get("raw_data", {}))

    # ---- Feedback section
    render_feedback_section(result["query"], answer, citations)

# ---- Feedback History (optional expander at bottom)
if st.session_state.feedback_history:
    st.divider()
    with st.expander(f"📋 Feedback History ({len(st.session_state.feedback_history)} submitted)"):
        for i, fb in enumerate(reversed(st.session_state.feedback_history), 1):
            thumb_icon = "👍" if fb["thumb"] == "up" else "👎"
            timestamp = fb.get("timestamp", "")
            comment = fb.get("comment", "")

            st.markdown(f"**{thumb_icon} {timestamp}**")
            st.caption(f"Query: *{fb['query']}*")
            if comment:
                st.caption(f"Comment: *{comment}*")
            if i < len(st.session_state.feedback_history):
                st.markdown("---")
