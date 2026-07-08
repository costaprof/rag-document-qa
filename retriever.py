import os
import chromadb
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"), http_options={'api_version': 'v1'})

def retrieve(query_embedding, n_results=3):
    chroma_client = chromadb.PersistentClient(path="vector_store")
    collection = chroma_client.get_or_create_collection(name="documents")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results["documents"][0]

def search(user_question):
    query_embedding = embed_query(user_question)
    return retrieve(query_embedding)

def embed_query(user_question):
    response = client.models.embed_content(
        model="models/gemini-embedding-2",
        contents=user_question
    )
    return response.embeddings[0].values
