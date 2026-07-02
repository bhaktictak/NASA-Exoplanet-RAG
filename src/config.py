from langchain_huggingface import HuggingFaceEmbeddings

# ----------------------------
# Embedding Model
# ----------------------------

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME
)

# ----------------------------
# Chroma Database
# ----------------------------

CHROMA_DB_PATH = "chroma_db"

# ----------------------------
# Retrieval Settings
# ----------------------------

TOP_K = 3

# ----------------------------
# Gemini Settings
# ----------------------------

GEMINI_MODEL = "gemini-2.5-flash"