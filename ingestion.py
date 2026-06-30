import os
import fitz
import chromadb
import google.genai
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = google.genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

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
            model="text-embedding-004",
            contents=chunk
        )
        embeddings.append(response.embeddings[0].values)
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

def ingest(pdf_path):
    chunks = extract(pdf_path)
    embeddings = embed(chunks)
    store(chunks, embeddings)