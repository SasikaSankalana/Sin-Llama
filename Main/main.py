from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
import os

app = FastAPI()

# ====================
# Load Models and Tokenizers
# ====================

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the Translate Model (fine-tuned on Singlish to English)
translate_model_path = "path_to_translate_model"  # Replace with your model path
translate_tokenizer = AutoTokenizer.from_pretrained(translate_model_path)
translate_model = AutoModelForCausalLM.from_pretrained(translate_model_path).to(device)

# Load the base LLaMA 3 model (for NL to SQL)
llama_model_path = "path_to_llama3_base_model"  # Replace with your model path
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_path)
llama_model = AutoModelForCausalLM.from_pretrained(llama_model_path).to(device)

# Load the Output Model (fine-tuned model for final output)
output_model_path = "path_to_finetuned_output_model"  # Replace with your model path
output_tokenizer = AutoTokenizer.from_pretrained(output_model_path)
output_model = AutoModelForCausalLM.from_pretrained(output_model_path).to(device)

# ====================
# Supabase Connection
# ====================

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# ====================
# Load FAISS Index
# ====================

def load_embeddings_from_database():
    """
    Load embeddings from Supabase and create a FAISS index.
    Assumes embeddings are stored in a table with columns 'id' and 'embedding' (stored as array).
    """
    response = supabase.table('embeddings_table').select('id, embedding').execute()
    rows = response.data
    
    if not rows:
        raise ValueError("No embeddings found in the database.")
    
    ids = []
    embeddings = []
    for row in rows:
        ids.append(row['id'])
        embedding = np.array(row['embedding'], dtype='float32')  # Assuming 'embedding' is stored as a list in Supabase
        embeddings.append(embedding)
    
    embeddings = np.vstack(embeddings).astype('float32')
    embedding_dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(embedding_dim)
    faiss_index.add(embeddings)
    return faiss_index, ids

faiss_index, embedding_ids = load_embeddings_from_database()

# Load the embedding model for query embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# ====================
# Helper Functions
# ====================

class Query(BaseModel):
    query: str

def generate_response(model, tokenizer, input_text):
    """
    Generate a response from a given model and tokenizer.
    """
    inputs = tokenizer.encode(input_text, return_tensors='pt').to(device)
    outputs = model.generate(inputs, max_new_tokens=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def execute_sql(sql_query):
    """
    Execute the given SQL query using Supabase and return the results.
    """
    response = supabase.rpc('run_sql', {"sql_query": sql_query}).execute()  # Use Supabase's RPC (or stored procedure) to run SQL
    if response.error:
        raise HTTPException(status_code=500, detail=response.error.message)
    return response.data

def perform_vector_search(query_text, top_k=5):
    """
    Perform a vector search using FAISS and return the associated data.
    """
    query_embedding = embedding_model.encode([query_text]).astype('float32')
    distances, indices = faiss_index.search(query_embedding, top_k)
    
    results = []
    for idx in indices[0]:
        embedding_id = embedding_ids[idx]
        response = supabase.table('embeddings_table').select('*').eq('id', embedding_id).execute()
        data = response.data
        if data:
            results.append(data[0])  # Assuming data contains the row dictionary
    return results

def generate_sql_query(natural_language_query):
    """
    Generate a PostgreSQL query for the banking database using the user's prompt.
    Load the SQL prompt from a separate file.
    """
    # Load the SQL prompt template from a file
    with open('./prompts/sql_prompt.txt', 'r') as file:
        prompt_template = file.read()

    # Insert the natural language query into the prompt
    prompt = prompt_template.format(natural_language_query=natural_language_query)

    # Generate SQL in JSON format using the model
    sql_json = generate_response(llama_model, llama_tokenizer, prompt)
    return sql_json

# ====================
# API Endpoint
# ====================

@app.post('/process_query')
def process_query(query: Query):
    try:
        singlish_query = query.query

        # Step 1: Translate Singlish to English
        translated_query = generate_response(translate_model, translate_tokenizer, singlish_query)

        # Step 2: Generate SQL query from English query using the defined prompt
        sql_json = generate_sql_query(translated_query)

        sql_result = ""

        if sql_json.hasSQL:
            sql_query = sql_json.query

            if sql_query.lower().startswith("select"):

                # Step 3: Execute SQL query
                sql_result = execute_sql(sql_query)
                

        # Step 4: Perform vector search
        vector_results = perform_vector_search(translated_query)

        # Step 5: Combine SQL and vector results
        combined_data = {
            'sql_result': sql_result,
            'vector_result': vector_results
        }

        # Step 6: Generate final output using the Output model
        combined_input = f"Original Query: {singlish_query}\nData: {combined_data}"
        final_output = generate_response(output_model, output_tokenizer, combined_input)

        return {
            'translated_query': translated_query,
            'sql_query': sql_query,
            'sql_result': sql_result,
            'vector_results': vector_results,
            'final_output': final_output
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ====================
# Run the App
# ====================

if __name__ == '__main__':
    uvicorn.run("main:app", host='0.0.0.0', port=8000)
