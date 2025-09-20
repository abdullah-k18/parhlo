import os
import streamlit as st
from groq import Groq
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Setup Pinecone client
pc = Pinecone(api_key=pinecone_api_key)

# Index + namespace (already created & loaded before)
index_name = "exam-prep"   # 🔹 change if your index name is different
namespace = "notes"

pinecone_index = pc.Index(index_name)

# Embedding model for queries
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# System prompt for the Groq model
system_prompt = """
You are a kind, friendly, and knowledgeable physics teacher. 
When students ask questions, carefully read the context provided. 
Use the context to understand the idea, but do not copy-paste the text directly. 
Instead, explain the answer in your own simple, clear words, 
as if teaching a high school student. 

- Always simplify complex terms.  
- Explain in an interactive and engaging way, so the student feels interested 
  and motivated to learn.  
- Use simple, everyday examples (not overly complex ones) to make the ideas clear.  
- Feel free to ask rhetorical questions or give fun analogies to keep the explanation lively.  
- If the context does not contain the answer, politely say: 
  'I could not find the answer in the provided notes.'  
"""



# Initialize Groq
groq = Groq(api_key=groq_api_key)

# Function to query Pinecone + Groq
def perform_rag(query: str) -> str:
    # Embed the user query
    query_embedding = hf_embeddings.embed_query(query)

    # Search Pinecone
    top_matches = pinecone_index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True,
        namespace=namespace
    )

    # Collect contexts
    contexts = [match["metadata"].get("text", match.get("id", "")) for match in top_matches["matches"]]
    context_text = "\n\n-------\n\n".join(contexts[:5])

    # Construct augmented query
    augmented_query = f"Context:\n{context_text}\n\nQuestion: {query}"

    # Ask Groq LLM
    res = groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query}
        ]
    )

    return res.choices[0].message.content

# ---- Streamlit UI ----
st.set_page_config(page_title="PARHLO", page_icon="📘")

st.title("📘 Physics")

query = st.text_input("Ask a question")

if st.button("Get Answer"):
    if query.strip():
        with st.spinner("🔎 Searching..."):
            try:
                answer = perform_rag(query)
                st.success("✅ Explanation:")
                st.write(answer)
            except Exception as e:
                st.error(f"❌ Error: {e}")
    else:
        st.warning("Please enter a question first.")
