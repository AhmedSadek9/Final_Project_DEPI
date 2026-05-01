import os
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import requests
import json

# 1. Setting up the relative path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(BASE_DIR, "data", "lecture 3.pdf")

# Define the function at the beginning
def ask_ai(question, index, embed_model, chunks):
    q_embed = embed_model.encode([question])
    D, I = index.search(np.array(q_embed).astype('float32'), k=3)
    context = "\n".join([chunks[idx] for idx in I[0]])
    
    url = "http://localhost:11434/api/generate"
    prompt = f"""
    Answer the question accurately based on the provided context from the Simplex Method lecture. 
    Pay close attention to mathematical constraints and tableau values.
    
    Context:
    {context}
    
    Question: {question}
    Answer:"""
    
    data = {
        "model": "gemma2:2b",
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=data)
        return response.json().get('response', 'Error in response')
    except Exception as e:
        return f"Make sure Ollama is running! Error: {e}"

# 2. Run the program and process the file
if not os.path.exists(pdf_path):
    print(f"[ERROR] File not found at relative path: {pdf_path}")
    print("Please ensure the PDF file is inside a folder named 'data' next to your script.")
else:
    print("[WAIT] Reading Simplex lecture file with high precision...")
    all_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n"
    
    chunks = [all_text[i:i+800] for i in range(0, len(all_text), 700)]

    print("[WAIT] Initializing search engine (FAISS)...")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embed_model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))

    print("\n[SUCCESS] Chatbot is ready to discuss the Simplex Method!")

    # Question Testing
    query_1 = "What are the requirements for a linear program to be in standard form?"
    print(f"\n[QUERY]: {query_1}")
    print(f"[BOT RESPONSE]: {ask_ai(query_1, index, embed_model, chunks)}")

    query_2 = "Explain the difference between slack and surplus variables."
    print(f"\n[QUERY]: {query_2}")
    print(f"[BOT RESPONSE]: {ask_ai(query_2, index, embed_model, chunks)}")