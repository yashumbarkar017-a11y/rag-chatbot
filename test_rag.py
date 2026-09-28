from src.rag_chain import get_rag_response


question = "What is RAG?"

answer, documents = get_rag_response(question)

print("\nQUESTION:")
print(question)

print("\nANSWER:")
print(answer)

print("\nSOURCES:")

for document in documents:
    print(document.metadata)