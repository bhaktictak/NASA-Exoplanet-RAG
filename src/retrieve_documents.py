from langchain_chroma import Chroma
from config import embedding_model, CHROMA_DB_PATH, TOP_K
# ----------------------------
# Step 1 : Load Embedding Model
# ----------------------------

vector_db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

print("\nVector Database Loaded Successfully!")

# ----------------------------
# Step 3 : User Query
# ----------------------------


query = input("\nEnter your question: ")

# ----------------------------
# Step 4 : Similarity Search
# ----------------------------

results = vector_db.similarity_search(
    query=query,
    k=TOP_K
)
# ----------------------------
# Step 5 : Display Results
# ----------------------------

print("\nTop 3 Retrieved Chunks\n")

for i, doc in enumerate(results, start=1):

    print("=" * 100)

    print(f"Result {i}")

    print(f"\nSource : {doc.metadata['source']}")

    print(f"Page : {doc.metadata['page'] + 1}")

    print("\nRetrieved Text:\n")

    print(doc.metadata)
    print(doc.page_content)

    print("\n")