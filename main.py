from ingestion import ingest
from retriever import search
from generator import generate
import os


if not os.path.exists("vector_store") or not os.listdir("vector_store"):
    print("Ingesting documents...")
    ingest("docs/")
    print("Ingestion complete.")
else:
    print("Vector store already exists, skipping ingestion.")

print("Ready! Type 'quit' to exit.\n")

while True:
    question = input("Ask a question: ")
    if question.lower() == "quit":
        break
    chunks = search(question)
    answer = generate(question, chunks)
    print(f"\nAnswer: {answer}\n")