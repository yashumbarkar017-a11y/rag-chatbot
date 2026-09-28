from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(pdf_path: str):
    """
    Load a PDF and return its pages as LangChain documents.
    """

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    # Store only the PDF filename in metadata.
    # This allows the Streamlit UI to show the correct
    # source document for every retrieved chunk.
    filename = Path(pdf_path).name

    for document in documents:
        document.metadata["source"] = filename

    return documents


def split_documents(documents):
    """
    Split PDF documents into smaller chunks
    for embedding and retrieval.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            "\n\n",
            "\n",
            " ",
            ""
        ]
    )

    chunks = text_splitter.split_documents(
        documents
    )

    return chunks


def load_and_split_pdf(pdf_path: str):
    """
    Load a PDF and split it into chunks.
    """

    documents = load_pdf(pdf_path)

    chunks = split_documents(
        documents
    )

    return chunks