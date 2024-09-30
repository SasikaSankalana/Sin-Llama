import PyPDF2
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from supabase import create_client, Client
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Supabase setup
url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def chunk_text(text, chunk_size=1000, overlap=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def create_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    return embeddings[0].tolist()

def store_in_supabase(chunk, embedding, page_number):
    try:
        # Insert the chunk and embedding into the Supabase table
        response = supabase.table("pdf_embeddings").insert({
            "content": chunk,
            "embedding": embedding,  # Assuming embedding is a list (vector format)
            "page_number": page_number
        }).execute()
        
        # Print success message based on the response
        if response.data:
            print(f"Successfully stored chunk for page {page_number}")
        else:
            print(f"Failed to store chunk for page {page_number}: {response}")
        return response.data
    except Exception as e:
        print(f"Error storing data in Supabase: {str(e)}")
        if hasattr(e, 'response'):
            print(f"Response status code: {e.response.status_code}")
            print(f"Response content: {e.response.content}")
        return None

def process_pdf(file_path):
    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        embedding = create_embedding(chunk)
        page_number = i + 1  # This is an approximation, you might want to adjust this
        result = store_in_supabase(chunk, embedding, page_number)
        if result is None:
            print(f"Failed to store chunk {i+1}/{len(chunks)}")
        else:
            print(f"Processed and stored chunk {i+1}/{len(chunks)}")

if __name__ == "__main__":
    pdf_path = "./content/General_Terms.pdf"
    process_pdf(pdf_path)
    print("PDF processing complete!")