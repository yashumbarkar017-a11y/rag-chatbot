from src.document_loader import load_and_split_pdf

pdf_path = "data/rag_sample_document.pdf"

chunks = load_and_split_pdf(pdf_path)

print("PDF loaded successfully!")
print("Number of chunks:", len(chunks))

if chunks:
    print("\nFirst chunk:")
    print(chunks[0].page_content[:1000])

    print("\nMetadata:")
    print(chunks[0].metadata)