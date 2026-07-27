from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings

# ----------------------------
# Project Paths
# ----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

CHROMA_DB_PATH = BASE_DIR / "chroma_db"
DOCS_PATH = BASE_DIR / "docs"

# ----------------------------
# Embedding Model
# ----------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda"},
)

# ----------------------------
# RAG Settings
# ----------------------------

TOP_K = 5

GEMINI_MODEL = "gemini-2.5-flash"