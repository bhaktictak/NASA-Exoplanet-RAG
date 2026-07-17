import re
import torch
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import embedding_model, CHROMA_DB_PATH
from langchain_chroma import Chroma


# ----------------------------
# Step 1 : Load Documents
# ----------------------------

loader = PyPDFDirectoryLoader("docs")
documents = loader.load()


# ----------------------------
# Step 2 : Chunk Documents
# ----------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)


# ----------------------------
# Step 3 : Clean Text
# ----------------------------

def clean_text(text):

    text = re.sub(r"read://\S+", "", text)

    text = re.sub(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", "", text)

    text = re.sub(r"\b\d{1,2}:\d{2}\s?(AM|PM)\b", "", text)

    text = re.sub(r"\b\d+/\d+\b", "", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


for chunk in chunks:
    chunk.page_content = clean_text(chunk.page_content)
# ----------------------------
# Step 3.1 : Remove Low-Information Chunks
# ----------------------------

chunks = [
    chunk
    for chunk in chunks
    if len(chunk.page_content.split()) >= 20
]

print(f"\nChunks after filtering : {len(chunks)}")


print(f"\nTotal Clean Chunks : {len(chunks)}")

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=CHROMA_DB_PATH
)


print("\nVector Database Created Successfully!")

print(f"\nTotal Embedded Chunks : {vector_db._collection.count()}")