import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def generate(user_question, chunks):
    context = "\n\n".join(chunks)
    prompt = f"""You are a helpful assistant. Answer the question based only on the provided context. 
If the answer is not in the context, say "I could not find this information in the provided documents."

Context:
{context}

Question: {user_question}
"""
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=prompt
    )
    return response.text