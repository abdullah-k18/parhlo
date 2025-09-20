import os
from pdf2image import convert_from_path
import pytesseract
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

# Load environment variables
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Setup Pinecone
pc = Pinecone(api_key=pinecone_api_key)

index_name = "exam-prep"
namespace = "notes"

# Create index if not exists
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,  # all-MiniLM-L6-v2 embedding size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# Load embeddings model
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 1: Extract text from PDF using OCR
pdf_file = "Physics 12 watermark Chapter 1 - 4.pdf"
pages = convert_from_path(pdf_file)

vectors = []

for i, page in enumerate(pages):
    text = pytesseract.image_to_string(page, lang="eng")
    if text.strip():  # avoid empty pages
        chunks = [text[j:j+800] for j in range(0, len(text), 800)]
        for idx, chunk in enumerate(chunks):
            embedding = hf_embeddings.embed_query(chunk)
            vectors.append({
                "id": f"{pdf_file}-page-{i+1}-chunk-{idx}",  # readable ID, no uuid
                "values": embedding,
                "metadata": {
                    "source": pdf_file,
                    "page": i+1,
                    "chunk": idx,
                    "text": chunk
                }
            })

# Step 2: Upsert into Pinecone
index.upsert(vectors=vectors, namespace=namespace)

print(f"✅ Uploaded {len(vectors)} chunks from {pdf_file} into Pinecone (namespace: {namespace})!")
