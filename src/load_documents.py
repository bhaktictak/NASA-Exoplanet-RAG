from langchain_community.document_loaders import PyPDFDirectoryLoader

# Folder containing all PDFs
DATA_PATH = "docs"

# Load all PDFs recursively
loader = PyPDFDirectoryLoader(DATA_PATH)

documents = loader.load()

print(f"\nTotal Documents Loaded: {len(documents)}\n")

# Display first few pages
for i, doc in enumerate(documents[:5]):
    print("=" * 80)
    print(f"Document {i+1}")
    print(f"Source: {doc.metadata['source']}")
    print(f"Page: {doc.metadata['page']}")
    print(doc.page_content[:500])  # First 500 characters
    print()