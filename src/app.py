import sys
from pathlib import Path
import tempfile
import shutil

import streamlit as st


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.rag_chain import get_rag_response
from src.vector_store import create_vector_store


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="StudyMate RAG",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main application */

    .stApp {
        background-color: #0e1117;
        color: white;
    }


    /* Main title */

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }


    /* Subtitle */

    .subtitle {
        color: #a0a0a0;
        font-size: 17px;
        margin-bottom: 25px;
    }


    /* Source card */

    .source-box {
        background-color: #161b22;
        padding: 12px;
        border-radius: 10px;
        margin-top: 8px;
        border: 1px solid #30363d;
    }


    /* Document card */

    .document-card {
        background-color: #161b22;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 8px;
        border: 1px solid #30363d;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "documents_processed" not in st.session_state:

    st.session_state.documents_processed = False


if "uploaded_files" not in st.session_state:

    st.session_state.uploaded_files = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📚 StudyMate")

    st.markdown(
        """
        **AI-powered document assistant**

        Upload one or multiple PDF documents
        and ask questions about their contents.
        """
    )

    st.divider()


    # ========================================================
    # PDF UPLOAD
    # ========================================================

    st.subheader("📄 Upload Documents")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can select multiple PDF files."
    )


    # ========================================================
    # SHOW SELECTED FILES
    # ========================================================

    if uploaded_files:

        st.write(
            f"**{len(uploaded_files)} PDF(s) selected**"
        )

        for uploaded_file in uploaded_files:

            st.markdown(
                f"""
                <div class="document-card">
                    📄 {uploaded_file.name}
                </div>
                """,
                unsafe_allow_html=True
            )


        # ====================================================
        # PROCESS BUTTON
        # ====================================================

        process_button = st.button(
            "🚀 Process Documents",
            use_container_width=True,
            type="primary"
        )


        if process_button:

            temp_dir = None

            try:

                with st.spinner(
                    "Processing documents..."
                ):

                    # ----------------------------------------
                    # Create temporary directory
                    # ----------------------------------------

                    temp_dir = tempfile.mkdtemp(
                        prefix="studymate_"
                    )


                    # ----------------------------------------
                    # Save uploaded PDFs
                    # ----------------------------------------

                    pdf_paths = []

                    progress = st.progress(
                        0
                    )

                    total_files = len(
                        uploaded_files
                    )


                    for index, uploaded_file in enumerate(
                        uploaded_files
                    ):

                        pdf_path = (
                            Path(temp_dir)
                            / uploaded_file.name
                        )


                        with open(
                            pdf_path,
                            "wb"
                        ) as file:

                            file.write(
                                uploaded_file.getbuffer()
                            )


                        pdf_paths.append(
                            str(pdf_path)
                        )


                        progress.progress(
                            int(
                                ((index + 1)
                                 / total_files)
                                * 50
                            )
                        )


                    # ----------------------------------------
                    # Create FAISS vector store
                    # ----------------------------------------

                    create_vector_store(
                        pdf_paths
                    )


                    progress.progress(
                        100
                    )


                    # ----------------------------------------
                    # Save document information
                    # ----------------------------------------

                    st.session_state.documents_processed = True

                    st.session_state.uploaded_files = [
                        uploaded_file.name
                        for uploaded_file
                        in uploaded_files
                    ]


                    # ----------------------------------------
                    # Clear old chat
                    # ----------------------------------------

                    st.session_state.messages = []


                st.success(
                    f"Successfully processed "
                    f"{len(uploaded_files)} PDF(s)! 🎉"
                )


            except Exception as e:

                st.error(
                    "❌ Error processing documents"
                )

                st.exception(
                    e
                )


            finally:

                # --------------------------------------------
                # Remove temporary files
                # --------------------------------------------

                if temp_dir:

                    shutil.rmtree(
                        temp_dir,
                        ignore_errors=True
                    )


    st.divider()


    # ========================================================
    # CURRENT DOCUMENTS
    # ========================================================

    st.subheader(
        "📚 Current Knowledge Base"
    )


    if st.session_state.documents_processed:

        st.success(
            "🟢 Ready for questions"
        )

        st.write(
            "**Documents:**"
        )

        for filename in st.session_state.uploaded_files:

            st.write(
                f"📄 {filename}"
            )

    else:

        st.info(
            "No documents processed yet."
        )


    st.divider()


    # ========================================================
    # MODEL
    # ========================================================

    st.subheader(
        "🤖 AI Model"
    )

    st.write(
        "Ollama"
    )

    st.code(
        "llama3.2:3b"
    )


    st.divider()


    # ========================================================
    # CLEAR CHAT
    # ========================================================

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🤖 StudyMate RAG</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Upload your PDFs and chat with your documents.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# WELCOME MESSAGE
# ============================================================

if not st.session_state.messages:

    with st.chat_message(
        "assistant"
    ):

        st.markdown(
            """
            👋 **Hello! I'm StudyMate.**

            I can answer questions using your uploaded
            PDF documents.

            ### 🚀 Getting Started

            1. Upload one or more PDFs from the sidebar.
            2. Click **Process Documents**.
            3. Ask me questions about your documents.

            ### 💡 Example Questions

            - What is RAG?
            - Explain this concept in simple words.
            - Give me the important points.
            - Summarize this topic.
            - Explain it with an example.
            """
        )


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


        # ====================================================
        # SOURCES
        # ====================================================

        if (
            message["role"] == "assistant"
            and "sources" in message
        ):

            sources = message["sources"]


            if sources:

                st.markdown(
                    "### 📚 Sources"
                )


                displayed_sources = set()


                for document in sources:

                    source = document.metadata.get(
                        "source",
                        "Unknown document"
                    )

                    page = document.metadata.get(
                        "page",
                        None
                    )


                    if page is not None:

                        source_text = (
                            f"📄 **{source}** "
                            f"— Page {page + 1}"
                        )

                    else:

                        source_text = (
                            f"📄 **{source}**"
                        )


                    if source_text not in displayed_sources:

                        st.markdown(
                            f"""
                            <div class="source-box">
                                {source_text}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        displayed_sources.add(
                            source_text
                        )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask StudyMate about your documents..."
)


if question:

    # ========================================================
    # CHECK DOCUMENTS
    # ========================================================

    if not st.session_state.documents_processed:

        st.warning(
            "📄 Please upload and process at least "
            "one PDF before asking questions."
        )

        st.stop()


    # ========================================================
    # USER MESSAGE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )


    # ========================================================
    # ASSISTANT RESPONSE
    # ========================================================

    with st.chat_message(
        "assistant"
    ):

        try:

            with st.spinner(
                "StudyMate is thinking..."
            ):

                answer, documents = get_rag_response(
                    question
                )


            # ------------------------------------------------
            # Answer
            # ------------------------------------------------

            st.markdown(
                answer
            )


            # ------------------------------------------------
            # Sources
            # ------------------------------------------------

            if documents:

                st.markdown(
                    "### 📚 Sources"
                )


                displayed_sources = set()


                for document in documents:

                    source = document.metadata.get(
                        "source",
                        "Unknown document"
                    )

                    page = document.metadata.get(
                        "page",
                        None
                    )


                    if page is not None:

                        source_text = (
                            f"📄 **{source}** "
                            f"— Page {page + 1}"
                        )

                    else:

                        source_text = (
                            f"📄 **{source}**"
                        )


                    if source_text not in displayed_sources:

                        st.markdown(
                            f"""
                            <div class="source-box">
                                {source_text}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        displayed_sources.add(
                            source_text
                        )


            # ------------------------------------------------
            # Save assistant response
            # ------------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": documents
                }
            )


        except Exception as e:

            error_message = (
                f"❌ Error generating response: {e}"
            )


            st.error(
                error_message
            )


            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message
                }
            )