import re
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 1: Load PDFs
loader = PyPDFDirectoryLoader("docs")
documents = loader.load()

# Step 2: Chunk documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print(f"Chunks before cleaning: {len(chunks)}")


# Step 3: Cleaning function
def clean_text(text):
    # Remove browser-generated read:// links
    text = re.sub(r"read://\S+", "", text)

    # Remove dates like 6/29/26
    text = re.sub(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", "", text)

    # Remove times like 11:16 AM
    text = re.sub(r"\b\d{1,2}:\d{2}\s?(AM|PM)\b", "", text)

    # Remove page numbers like 1/53 or 12/120
    text = re.sub(r"\b\d+/\d+\b", "", text)
    
    # Replace multiple spaces/newlines with a single space
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# Step 4: Apply cleaning
for chunk in chunks:
    chunk.page_content = clean_text(chunk.page_content)

print("\nFIRST CLEANED CHUNK\n")
print("=" * 80)
print(chunks[0].page_content)

print("\nMetadata:")
print(chunks[0].metadata)