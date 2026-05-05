import streamlit as st
from rag import create_rag, ask_question
import os

st.set_page_config(page_title="StudyBot", page_icon="🎓", layout="wide")

with st.sidebar:
    st.title("🎓 StudyBot")

    # New Chat
    if st.button("🆕 New Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    # Chat History (ONLY QUESTIONS)
    st.subheader("💬 Chat History")

    if "messages" in st.session_state and st.session_state.messages:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"• {msg['content']}")
    else:
        st.write("No questions yet")

    st.markdown("---")

    # Optional: Clear Chat
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ─────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────
st.title("💬 StudyBot")

# Initialize session
if "messages" not in st.session_state:
    st.session_state.messages = []

if "ready" not in st.session_state:
    with st.spinner("Loading PDFs..."):
        retriever, llm, prompt = create_rag()
        st.session_state.retriever = retriever
        st.session_state.llm = llm
        st.session_state.prompt = prompt
        st.session_state.ready = True


# ─────────────────────────────────────
# CHAT HISTORY
# ─────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# ─────────────────────────────────────
# INPUT
# ─────────────────────────────────────
query = st.chat_input("Ask something from your syllabus...")

if query:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.write(query)

    # Generate response
    with st.spinner("Thinking..."):
        answer = ask_question(
            query,
            st.session_state.retriever,
            st.session_state.llm,
            st.session_state.prompt
        )

    # Save response
    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.write(answer)