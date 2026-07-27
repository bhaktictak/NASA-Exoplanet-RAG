import streamlit as st
import re
import time
from rag_chatbot import answer_question

# ====================================================
# PAGE CONFIGURATION
# ====================================================
st.set_page_config(
    page_title="NASA Space Science Assistant",
    page_icon="🚀",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional typography and spacing
st.markdown("""
<style>
    .reportview-container {
        font-family: 'Inter', sans-serif;
    }
    .stChatMessage {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .source-box {
        font-size: 0.9em;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)

# ====================================================
# HELPER FUNCTIONS
# ====================================================
def format_source_name(source_name):
    """Removes numeric prefixes and replaces underscores with spaces."""
    cleaned = re.sub(r'^\d+_', '', source_name)
    return cleaned.replace('_', ' ')

def clear_chat():
    """Clears the chat history from session state."""
    st.session_state.messages = []
    # Re-initialize the welcome message
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! I'm your NASA Space Science Assistant.\nAsk me anything about exoplanets, JWST, habitability or NASA documents.",
        "sources": None,
        "time": None
    })

# ====================================================
# SIDEBAR
# ====================================================
with st.sidebar:
    st.title("🚀 NASA RAG")
    st.markdown("---")
    st.markdown("""
    **Embedding**  
    MiniLM-L6-v2  

    **Vector Database**  
    ChromaDB  

    **Retriever**  
    MMR  

    **LLM**  
    Gemini 2.5 Flash  
    """)
    st.markdown("---")
    st.markdown("""
    **Knowledge Base**  
    NASA Documents  
    """)
    st.markdown("---")
    st.button("Clear Chat", on_click=clear_chat, use_container_width=True, type="primary")

# ====================================================
# MAIN PAGE HEADER
# ====================================================
st.title("🚀 NASA Space Science Assistant")
st.markdown("Ask questions about NASA documents.")
st.markdown("<br>", unsafe_allow_html=True)

# ====================================================
# SESSION STATE INITIALIZATION
# ====================================================
if "messages" not in st.session_state:
    clear_chat()

if "selected_prompt" not in st.session_state:
    st.session_state.selected_prompt = None

# ====================================================
# RENDER CHAT HISTORY
# ====================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display expandable sources if they exist
        if msg.get("sources"):
            with st.expander("📄 Sources"):
                for src, pages in msg["sources"].items():
                    formatted_name = format_source_name(src)
                    pages_list = ", ".join(map(str, sorted(list(pages))))
                    st.markdown(f"**{formatted_name}**  \n*(Page{'s' if len(pages) > 1 else ''} {pages_list})*")
        
        # Display generation time
        if msg.get("time"):
            st.caption(f"⏱️ Generated in {msg['time']:.2f}s")

# ====================================================
# EXAMPLE QUESTIONS (BONUS)
# ====================================================
# Only show examples if the chat history contains only the initial welcome message
if len(st.session_state.messages) == 1:
    st.markdown("### 💡 Try asking:")
    example_prompts = [
        "What are exoplanets?",
        "What is the Transit Method?",
        "What is JWST?",
        "What is the Habitable Zone?"
    ]
    cols = st.columns(2)
    for i, ex_prompt in enumerate(example_prompts):
        if cols[i % 2].button(ex_prompt, use_container_width=True):
            st.session_state.selected_prompt = ex_prompt
            st.rerun()

# ====================================================
# CHAT INPUT & PROCESSING
# ====================================================
# Capture user input either from chat_input or an example button click
user_query = st.chat_input("Message NASA Assistant...")

if st.session_state.selected_prompt:
    user_query = st.session_state.selected_prompt
    st.session_state.selected_prompt = None

if user_query:
    # 1. Display and store user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching NASA documents..."):
            start_time = time.time()
            try:
                # Call the backend function
                history = st.session_state.messages[-6:]

                rag_response, sources = answer_question(
                    user_query,
                    history
                )
                answer_text = rag_response.answer
                error_occurred = False
            except Exception as e:
                answer_text = "Something went wrong while generating the response."
                sources = {}
                error_occurred = True
            
            end_time = time.time()
            generation_time = end_time - start_time

        # Render response
        st.markdown(answer_text)
        
        # Render sources in an expander
        if sources and not error_occurred:
            with st.expander("📄 Sources"):
                for src, pages in sources.items():
                    formatted_name = format_source_name(src)
                    pages_list = ", ".join(map(str, sorted(list(pages))))
                    st.markdown(f"**{formatted_name}**  \n*(Page{'s' if len(pages) > 1 else ''} {pages_list})*")
        
        # Render generation time
        if not error_occurred:
            st.caption(f"⏱️ Generated in {generation_time:.2f}s")

    # 3. Store assistant message in state
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "sources": sources if not error_occurred else None,
        "time": generation_time if not error_occurred else None
    })