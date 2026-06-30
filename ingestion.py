import os
import fitz
import chromadb
from dotenv import load_dotenv
from google import genai
import time

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"), http_options={'api_version': 'v1'})

def extract(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    chunks = []
    for i in range (0, len(text), 450):
        chunk = text[i:i + 500]
        chunks.append(chunk)
    return chunks


def embed(chunks):
    embeddings = []
    for chunk in chunks:
        response = client.models.embed_content(
            model="models/gemini-embedding-2",
            contents=chunk
        )
        embeddings.append(response.embeddings[0].values)
        time.sleep(0.7)
    return embeddings

def store(chunks, embeddings):
    chroma_client = chromadb.PersistentClient(path="vector_store") # save to disk at the vector_store/ folder
    collection = chroma_client.get_or_create_collection(name="documents") # creates a new collection called "documents" or gets it if it already exists
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            embeddings=[embeddings[i]],
            ids=[f"doc_{i}"]
        )

def ingest(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            print(f"Ingesting {filename}...")
            chunks = extract(pdf_path)
            embeddings = embed(chunks)
            store(chunks, embeddings)