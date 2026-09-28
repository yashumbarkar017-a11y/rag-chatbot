from pathlib import Path

from langchain_community.vectorstores import FAISS

from src.document_loader import load_and_split_pdf
from src.embeddings import get_embeddings


# ============================================================
# VECTOR STORE LOCATION
# ============================================================

VECTORSTORE_PATH = "vectorstore"


# ============================================================
# CREATE VECTOR STORE
# ============================================================

def create_vector_store(pdf_paths):
    """
    Create a FAISS vector store from one or multiple PDFs.

    Parameters
    ----------
    pdf_paths : list[str] or str
        Paths of the PDF files.

    Returns
    -------
    FAISS
        Created FAISS vector store.
    """

    # --------------------------------------------------------
    # Convert single path to list
    # --------------------------------------------------------

    if isinstance(pdf_paths, str):
        pdf_paths = [pdf_paths]

    # --------------------------------------------------------
    # Store chunks from all PDFs
    # --------------------------------------------------------

    all_chunks = []

    # --------------------------------------------------------
    # Process every PDF
    # --------------------------------------------------------

    for pdf_path in pdf_paths:

        print(
            f"\nLoading PDF: {pdf_path}"
        )

        chunks = load_and_split_pdf(
            pdf_path
        )

        print(
            f"Chunks created: {len(chunks)}"
        )

        all_chunks.extend(
            chunks
        )

    # --------------------------------------------------------
    # Make sure chunks exist
    # --------------------------------------------------------

    if not all_chunks:

        raise ValueError(
            "No text could be extracted from the uploaded PDFs."
        )

    print(
        f"\nTotal chunks: {len(all_chunks)}"
    )

    # --------------------------------------------------------
    # Create embeddings
    # --------------------------------------------------------

    print(
        "Creating embeddings..."
    )

    embeddings = get_embeddings()

    # --------------------------------------------------------
    # Create FAISS
    # --------------------------------------------------------

    print(
        "Creating FAISS vector store..."
    )

    vector_store = FAISS.from_documents(
        all_chunks,
        embeddings
    )

    # --------------------------------------------------------
    # Create directory
    # --------------------------------------------------------

    Path(
        VECTORSTORE_PATH
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save vector store
    # --------------------------------------------------------

    vector_store.save_local(
        VECTORSTORE_PATH
    )

    print(
        "\nVector store created successfully!"
    )

    return vector_store


# ============================================================
# LOAD VECTOR STORE
# ============================================================

def load_vector_store():
    """
    Load the existing FAISS vector store.
    """

    embeddings = get_embeddings()

    vector_store = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store