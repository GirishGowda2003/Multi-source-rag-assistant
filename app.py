"""Streamlit chatbot: upload files / URLs, filter by source, persistent chat history."""
import os
import json
import asyncio
import tempfile
from pathlib import Path

import streamlit as st
from rag_project import Config, KnowledgeBase, Retriever, RAGChatbot, aload_sources, chunk_documents

st.set_page_config(page_title="RAG Chatbot", page_icon="📚", layout="wide")
cfg = Config()
HISTORY_FILE = Path("chat_history.json")      # persistent memory across restarts


@st.cache_resource(show_spinner="Loading embedding model and vector store...")
def get_kb():
    return KnowledgeBase(cfg)


@st.cache_resource(show_spinner=False)
def get_retriever():
    return Retriever(get_kb(), cfg)


def load_history():
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_history(messages):
    HISTORY_FILE.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")


def make_bot(api_key, messages):
    bot = RAGChatbot(get_retriever(), cfg, api_key=api_key)
    users = [m["content"] for m in messages if m["role"] == "user"]
    bots = [m["content"] for m in messages if m["role"] == "assistant"]
    bot.history = list(zip(users, bots))          # restore conversational memory
    return bot


# ----------------------------- sidebar ------------------------------------ #
kb = get_kb()
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Groq API key", type="password",
                            value=os.environ.get("GROQ_API_KEY", ""))
    st.divider()
    st.subheader("Add knowledge")
    files = st.file_uploader("PDF / CSV / TXT", type=["pdf", "csv", "txt"], accept_multiple_files=True)
    url = st.text_input("Web page URL")
    if st.button("Index sources", type="primary"):
        specs = []
        tmp = Path(tempfile.mkdtemp())
        for f in files or []:
            p = tmp / f.name
            p.write_bytes(f.getbuffer())
            kind = {".pdf": "pdf", ".csv": "csv"}.get(p.suffix.lower(), "text")
            specs.append({"type": kind, "path": str(p), "name": f.name})
        if url.strip():
            specs.append({"type": "web", "path": url.strip()})
        if not specs:
            st.warning("Add a file or a URL first.")
        else:
            with st.spinner("Loading sources in parallel..."):
                docs, report = asyncio.run(aload_sources(specs))      # non-blocking, parallel
                n = kb.add(chunk_documents(docs, cfg))
            st.success(f"Indexed {n} chunks")
            st.json(report)

    st.divider()
    st.metric("Chunks in database", kb.count())
    inventory = kb.sources()
    names = sorted({name for (_, name) in inventory})
    chosen = st.multiselect("Search only in these sources (empty = all)", names)
    if st.button("Clear chat history"):
        HISTORY_FILE.unlink(missing_ok=True)
        st.session_state.pop("messages", None)
        st.session_state.pop("bot", None)
        st.rerun()

# ----------------------------- chat --------------------------------------- #
st.title("📚 RAG Chatbot")
st.caption("Answers come only from your indexed documents. Sources are shown under each answer.")

if "messages" not in st.session_state:
    st.session_state.messages = load_history()
if not api_key:
    st.info("Enter your Groq API key in the sidebar to start chatting.")
    st.stop()
if kb.count() == 0:
    st.info("The knowledge base is empty. Index a file or URL from the sidebar.")
    st.stop()
if "bot" not in st.session_state:
    st.session_state.bot = make_bot(api_key, st.session_state.messages)
bot = st.session_state.bot

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for s in m["sources"]:
                    st.markdown(f"**[{s['n']}] {s['source']}**  \n{s['snippet']}...")

if question := st.chat_input("Ask something about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and re-ranking..."):
            prep = bot.prepare(question, source_names=chosen or None)
        if prep["short_circuit"]:
            answer = cfg.idk_message
            st.markdown(answer)
        else:
            answer = st.write_stream(c.content for c in bot.llm.stream(prep["messages"]))
        bot.commit(question, answer)
        sources = bot.format_sources(prep["hits"])
        with st.expander("Sources"):
            for s in sources:
                st.markdown(f"**[{s['n']}] {s['source']}** (rerank {s['rerank_score']:.2f})  \n{s['snippet']}...")
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
    save_history(st.session_state.messages)
