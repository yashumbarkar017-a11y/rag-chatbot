from langchain_ollama import ChatOllama

from src.vector_store import load_vector_store


# ============================================================
# OLLAMA MODEL
# ============================================================

def get_llm():
    """
    Create the local Ollama language model.
    """

    return ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )


# ============================================================
# PERSONA
# ============================================================

STUDYMATE_PERSONA = """
You are StudyMate, a friendly and knowledgeable academic assistant.

Your purpose is to help students understand information from
their uploaded documents.

PERSONALITY:
- Friendly
- Patient
- Clear
- Encouraging
- Student-focused
- Professional

COMMUNICATION STYLE:
- Explain difficult concepts in simple language.
- Avoid unnecessary technical jargon.
- Give examples when they help understanding.
- Use bullet points for lists and multiple concepts.
- Use short paragraphs.
- Highlight important terms when useful.
- For simple questions, give concise answers.
- For complex questions, provide a structured explanation.

KNOWLEDGE RULES:
- Use the provided document context as the primary source.
- Do not invent facts.
- Do not make up information that is not present in the
  retrieved document context.
- If the answer is present in the context, answer confidently.
- If the answer cannot be found in the context, say:
  "I couldn't find the answer in the provided documents."
- Do not pretend that information exists in the documents
  when it does not.

STUDENT SUPPORT:
- When explaining a concept, explain it step-by-step when useful.
- When appropriate, provide a simple example.
- Help the student understand rather than simply giving
  complicated terminology.
"""


# ============================================================
# RAG RESPONSE
# ============================================================

def get_rag_response(question: str):
    """
    Retrieve relevant document chunks from FAISS and
    generate a StudyMate response using Ollama.
    """

    # --------------------------------------------------------
    # 1. Load FAISS vector database
    # --------------------------------------------------------

    vector_store = load_vector_store()


    # --------------------------------------------------------
    # 2. Retrieve relevant document chunks
    # --------------------------------------------------------

    documents = vector_store.similarity_search(
        question,
        k=3
    )


    # --------------------------------------------------------
    # 3. Check whether documents were retrieved
    # --------------------------------------------------------

    if not documents:

        return (
            "I couldn't find the answer in the provided documents.",
            []
        )


    # --------------------------------------------------------
    # 4. Build document context
    # --------------------------------------------------------

    context_parts = []

    for index, document in enumerate(
        documents,
        start=1
    ):

        context_parts.append(
            f"""
DOCUMENT CHUNK {index}
-------------------------
{document.page_content}
-------------------------
"""
        )


    context = "\n".join(context_parts)


    # --------------------------------------------------------
    # 5. Create StudyMate prompt
    # --------------------------------------------------------

    prompt = f"""
{STUDYMATE_PERSONA}

============================================================
DOCUMENT CONTEXT
============================================================

{context}

============================================================
USER QUESTION
============================================================

{question}

============================================================
INSTRUCTIONS FOR THIS RESPONSE
============================================================

Answer the user's question using the document context.

Remember:

1. Stay in the StudyMate persona.
2. Use the retrieved documents as your source.
3. Explain the answer clearly.
4. Keep the explanation student-friendly.
5. Give an example if it improves understanding.
6. Do not invent information.
7. If the answer is not available in the document context,
   say:

   "I couldn't find the answer in the provided documents."

Now provide the answer.
"""


    # --------------------------------------------------------
    # 6. Get Ollama model
    # --------------------------------------------------------

    llm = get_llm()


    # --------------------------------------------------------
    # 7. Generate response
    # --------------------------------------------------------

    response = llm.invoke(prompt)


    # --------------------------------------------------------
    # 8. Extract response text
    # --------------------------------------------------------

    answer = response.content


    # --------------------------------------------------------
    # 9. Return answer and source documents
    # --------------------------------------------------------

    return answer, documents