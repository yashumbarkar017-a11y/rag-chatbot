from src.vector_store import create_vector_store


pdf_path = "data/rag_sample_document.pdf"

vector_store = create_vector_store(pdf_path)

print("\nFAISS vector store is ready!")