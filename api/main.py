# api/main.py

import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel # For defining request body structure
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
import time
import uvicorn # Import uvicorn to run the app

# --- Configuration ---
# Path relative to the project root (where 'api' folder is)
# Adjust if your 'chroma_db' is not directly in the parent directory
CHROMA_DB_DIR = "../chroma_db" 
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_LLM_MODEL = "llama3:latest" # Or "mistral:7b-instruct-v0.2-q4_K_M"

app = FastAPI()

# --- CORS Configuration ---
# Enable CORS for all origins, methods, and headers, allowing frontend to access it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Global variables to store the RAG chain and vector store
qa_chain = None
vector_store = None

# --- Request Body Model ---
# Define the expected structure for the POST request body
class QueryRequest(BaseModel):
    query: str

# --- Startup Event: Initialize RAG Components ---
@app.on_event("startup")
async def startup_event():
    """
    Initializes RAG components when the FastAPI application starts up.
    This ensures models and vector store are loaded only once.
    """
    print("Starting FastAPI API. Initializing RAG components...")
    global vector_store, qa_chain
    
    # Load Vector Store
    vector_store = load_vector_store(CHROMA_DB_DIR, OLLAMA_EMBEDDING_MODEL)
    if vector_store:
        # Setup RAG Chain if vector store loaded successfully
        qa_chain = setup_rag_chain(vector_store, OLLAMA_LLM_MODEL)
    else:
        print("Failed to load vector store. RAG system will not be functional.")
        # Optionally, raise an exception here to prevent the app from starting if critical
        # raise HTTPException(status_code=500, detail="Failed to load RAG components")

def load_vector_store(db_dir: str, embedding_model_name: str):
    """
    Loads the ChromaDB vector store from the specified directory.
    """
    print(f"Loading vector store from {db_dir} with embedding model {embedding_model_name}...")
    try:
        embeddings = OllamaEmbeddings(model=embedding_model_name)
        vector_store = Chroma(persist_directory=db_dir, embedding_function=embeddings)
        print("Vector store loaded successfully.")
        return vector_store
    except Exception as e:
        print(f"Error loading vector store. Make sure '{db_dir}' exists and contains a valid ChromaDB. Error: {e}")
        return None

def setup_rag_chain(vector_store: Chroma, llm_model_name: str):
    """
    Sets up the RAG (Retrieval Augmented Generation) chain.
    """
    print(f"Initializing Ollama LLM with model: {llm_model_name}...")
    try:
        llm = ChatOllama(model=llm_model_name)
        print("LLM initialized.")
    except Exception as e:
        print(f"Error initializing Ollama LLM. Make sure Ollama server is running and model '{llm_model_name}' is pulled. Error: {e}")
        return None

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    template = """
    You are an AI assistant providing information about Ubestream.com.
    Answer the question truthfully and concisely using ONLY the following context.
    Provide the answer directly, without any introductory phrases like "According to the context" or "Based on the information".
    If the answer is not found in the context, state that you cannot answer based on the provided information.
    Do not make up any information.
    If the question is a general greeting or unrelated to the context, respond appropriately but do not invent facts about Ubestream.com.

    Context:
    {context}

    Question: {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    print("RAG chain setup complete.")
    return qa_chain

# --- API Endpoints ---

@app.post("/ask")
async def ask_question_endpoint(request_body: QueryRequest):
    """
    API endpoint to receive a question and return an answer from the RAG system.
    """
    query = request_body.query

    if not query:
        raise HTTPException(status_code=400, detail="No query provided")

    if qa_chain is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized. Please check server logs.")

    print(f"Received question: '{query}'")
    try:
        start_time = time.time()
        result = qa_chain.invoke({"query": query})
        end_time = time.time()
        inference_time = end_time - start_time
        print(f"Inference time for '{query}': {inference_time:.2f} seconds")

        answer = result["result"]
        return {"answer": answer, "inference_time": f"{inference_time:.2f} seconds"}
    except Exception as e:
        print(f"Error during RAG inference: {e}")
        raise HTTPException(status_code=500, detail="Failed to get answer from RAG system. Check Ollama server status.")

@app.get("/status")
async def status_check_endpoint():
    """
    API endpoint to check if the RAG system is ready.
    """
    if qa_chain is not None:
        return {"status": "ready", "message": "RAG system initialized and ready."}
    else:
        return {"status": "initializing", "message": "RAG system still initializing or failed to load."}

# --- Run the FastAPI application ---
# This block is for direct execution of the script.
# In a production environment, you would typically run Uvicorn via the command line:
# uvicorn api.main:app --host 0.0.0.0 --port 8006 --reload
if __name__ == '__main__':
    uvicorn.run("main:app", host='0.0.0.0', port=8006, reload=True)
