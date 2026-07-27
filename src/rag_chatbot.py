import os
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from models import RAGResponse
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


def answer_question(question: str, chat_history=None):

    docs = vector_db.max_marginal_relevance_search(
        question,
        k=TOP_K,
        fetch_k=20
    )
    

    sources = {}

    for doc in docs:
        source = Path(doc.metadata["source"]).stem
        page = doc.metadata["page"] + 1

        if source not in sources:
            sources[source] = set()

        sources[source].add(page)
    context = "\n\n".join(
        doc.page_content for doc in docs
    )
    
    conversation = ""

    if chat_history:
        for msg in chat_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation += f"{role}: {msg['content']}\n"


    prompt = f"""
You are a NASA Space Science Assistant.

Use ONLY the provided context.

The conversation history is provided to help understand follow-up questions.

If the current question refers to something mentioned earlier (like "it", "they", "that telescope"), use the conversation history to understand what the user means.

Do NOT invent facts.

Only answer using the retrieved context.

Conversation History:

{conversation}

Retrieved Context:

{context}

Current Question:

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

    rag_response = RAGResponse(
        answer=response.text,
        sources=list(sources.keys())
    )
    return rag_response, sources
    
if __name__ == "__main__":
    print("\n NASA Space RAG Chatbot")
    print("Type 'exit' to quit.\n")
    while True:

        question = input("You: ")
    
        if question.lower() == "exit":
            break
        
        rag_response, sources = answer_question(question)
    
        print("\nAnswer:")
        print(rag_response.answer)
    
        print("\nSources:")
    
        for source, pages in sources.items():
            pages = sorted(pages)
    
            if len(pages) == 1:
                print(f"• {source} (Page {pages[0]})")
            else:
                print(f"• {source} (Pages {', '.join(map(str, pages))})")
    
        print("\n" + "=" * 100 + "\n")