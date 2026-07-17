from langchain_huggingface import HuggingFaceEmbeddings


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda"},
)


CHROMA_DB_PATH = "chroma_db"

TOP_K = 5

GEMINI_MODEL = "gemini-2.5-flash"