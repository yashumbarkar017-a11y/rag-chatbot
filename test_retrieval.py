from src.vector_store import load_vector_store


# Load the existing FAISS database
vector_store = load_vector_store()

# Question from the user
question = "What is RAG?"

# Search for the most relevant chunks
results = vector_store.similarity_search(
    question,
    k=3
)

print("\nQuestion:")
print(question)

print("\nRetrieved Documents:")
print("=" * 60)

for i, doc in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print(doc.page_content)
    print("\nMetadata:")
    print(doc.metadata)