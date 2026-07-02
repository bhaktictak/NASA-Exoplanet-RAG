from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 1: Load all PDF pages
loader = PyPDFDirectoryLoader("docs")
documents = loader.load()

print(f"Loaded {len(documents)} pages.\n")

# Step 2: Create a text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Step 3: Split pages into chunks
chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.\n")

# Step 4: Show the first chunk
print("=" * 80)
print("FIRST CHUNK")
print("=" * 80)
print(chunks[0].page_content)

print("\nMetadata:")
print(chunks[0].metadata)