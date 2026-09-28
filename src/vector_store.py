from pathlib import Path

from langchain_community.vectorstores import FAISS

from src.document_loader import load_and_split_pdf
from src.embeddings import get_embeddings


VECTORSTORE_PATH = "vectorstore"


def create_vector_store(pdf_path: str):
    """
    Load a PDF, split it into chunks, create embeddings,
    and save the FAISS vector store.
    """

    print("Loading PDF...")

    chunks = load_and_split_pdf(pdf_path)

    print(f"Loaded {len(chunks)} chunks.")

    print("Creating embeddings...")

    embeddings = get_embeddings()

    print("Creating FAISS vector store...")

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    Path(VECTORSTORE_PATH).mkdir(
        parents=True,
        exist_ok=True
    )

    vector_store.save_local(VECTORSTORE_PATH)

    print("Vector store created successfully!")

    return vector_store


def load_vector_store():
    """
    Load an existing FAISS vector store.
    """

    embeddings = get_embeddings()

    vector_store = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store