import sys
from pathlib import Path

# ============================================================
# FIX PROJECT PATH
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ============================================================
# IMPORTS
# ============================================================

import streamlit as st

from src.rag_chain import get_rag_response


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background-color: #0e1117;
    }

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        color: #ffffff;
        margin-top: 10px;
        margin-bottom: 5px;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #9ca3af;
        font-size: 16px;
        margin-bottom: 30px;
    }

    /* Source box */
    .source-box {
        background-color: #161b22;
        border-radius: 10px;
        padding: 12px;
        margin-top: 10px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🤖 RAG Chatbot")

    st.markdown("---")

    st.markdown(
        """
        ### About

        This chatbot uses:

        - 📄 PDF documents
        - ✂️ Text chunking
        - 🧠 HuggingFace embeddings
        - 🔍 FAISS vector search
        - 🦙 Ollama
        - 💬 Streamlit
        """
    )

    st.markdown("---")

    st.markdown("### Current Model")

    st.info("llama3.2:3b")

    st.markdown("---")

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🤖 RAG Chatbot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions about your document'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# INITIALIZE CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# WELCOME MESSAGE
# ============================================================

if len(st.session_state.messages) == 0:

    with st.chat_message("assistant"):

        st.markdown(
            """
            👋 **Hello!**

            I'm your RAG chatbot.

            Ask me something about the PDF in your knowledge base.

            **Try asking:**

            - What is RAG?
            - What is artificial intelligence?
            - What is a primary key?
            - What is FAISS?
            - What are the stages of SDLC?
            """
        )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # Show sources for assistant messages
        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander("📚 View Sources"):

                for source in message["sources"]:

                    st.write(
                        f"📄 **{source['source']}** "
                        f"| Page **{source['page']}**"
                    )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask something about your document..."
)


# ============================================================
# PROCESS USER QUESTION
# ============================================================

if question:

    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(question)

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    # --------------------------------------------------------
    # Generate assistant response
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🔍 Searching your documents..."
        ):

            try:

                answer, documents = get_rag_response(
                    question
                )


                # ------------------------------------------------
                # Display answer
                # ------------------------------------------------

                st.markdown(answer)


                # ------------------------------------------------
                # Extract sources
                # ------------------------------------------------

                sources = []

                for document in documents:

                    source_path = document.metadata.get(
                        "source",
                        "Unknown"
                    )

                    page_number = document.metadata.get(
                        "page",
                        None
                    )

                    if page_number is not None:

                        page_number = page_number + 1

                    else:

                        page_number = "Unknown"


                    source = {
                        "source": source_path,
                        "page": page_number
                    }


                    # Avoid duplicate sources
                    if source not in sources:

                        sources.append(source)


                # ------------------------------------------------
                # Display sources
                # ------------------------------------------------

                if sources:

                    with st.expander(
                        "📚 View Sources"
                    ):

                        for source in sources:

                            st.write(
                                f"📄 **{source['source']}** "
                                f"| Page **{source['page']}**"
                            )


                # ------------------------------------------------
                # Save assistant message
                # ------------------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    }
                )


            except Exception as e:

                error_message = str(e)

                st.error(
                    "❌ Something went wrong."
                )

                with st.expander(
                    "🔧 Error Details"
                ):

                    st.code(
                        error_message
                    )