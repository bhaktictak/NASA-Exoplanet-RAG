import os

from dotenv import load_dotenv

from google import genai
from google.genai import types

from langchain_chroma import Chroma

from config import (
    embedding_model,
    CHROMA_DB_PATH,
    TOP_K,
    GEMINI_MODEL,
)

# ----------------------------
# Load Environment Variables
# ----------------------------

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# ----------------------------
# Load Vector Database
# ----------------------------

vector_db = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embedding_model
)

print("\n NASA Space RAG Chatbot")
print("Type 'exit' to quit.\n")

while True:

    question = input("You: ")

    if question.lower() == "exit":
        break

    # ----------------------------
    # Retrieve Documents
    # ----------------------------

    docs = vector_db.similarity_search(
        question,
        k=TOP_K
    )

    context = "\n\n".join(
        doc.page_content for doc in docs
    )
    

    prompt = f"""
You are a NASA Space Science Assistant.

Use ONLY the provided context.

If the context contains information that partially answers the question, answer using that information.

Only say "I couldn't find that information in the provided NASA documents." if the retrieved context is completely unrelated to the user's question.

Be concise and factual.

Context:
{context}

Question:
{question}

Answer:
"""

    # ----------------------------
    # Gemini Response
    # ----------------------------

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3
        ),
    )

    print("\n", response.text)
    print("\n" + "=" * 100 + "\n")