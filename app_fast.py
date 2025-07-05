# app.py

import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
import time # Import the time module

# --- Configuration ---
# Directory where the Chroma vector store is saved
CHROMA_DB_DIR = "./chroma_db"
# Ollama embedding model used during data preparation (must match)
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
# Ollama LLM for generating answers
# Recommended: "llama3:latest" or try "mistral:7b-instruct-v0.2-q4_K_M" for potentially faster inference
OLLAMA_LLM_MODEL = "mistral:7b-instruct-v0.2-q4_K_M" 

def load_vector_store(db_dir, embedding_model_name):
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

def setup_rag_chain(vector_store, llm_model_name):
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

    # Define the retriever to fetch relevant documents from the vector store
    # k=3 means it will retrieve the top 3 most relevant chunks for better accuracy.
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Define the prompt template for the LLM
    # This prompt is designed to make the LLM answer directly and concisely,
    # without explicitly stating "According to the provided context".
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
    
    # Create a ChatPromptTemplate
    prompt = ChatPromptTemplate.from_template(template)

    # Create the RAG chain
    # chain_type="stuff" means it will stuff all retrieved documents into the prompt.
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        # return_source_documents=True, # Removed to make output more concise, as requested
        chain_type_kwargs={"prompt": prompt} # Pass the custom prompt
    )
    print("RAG chain setup complete.")
    return qa_chain

def main():
    """
    Main function to run the Q&A application.
    """
    # 1. Load Vector Store
    vector_store = load_vector_store(CHROMA_DB_DIR, OLLAMA_EMBEDDING_MODEL)
    if vector_store is None:
        return

    # 2. Setup RAG Chain
    qa_chain = setup_rag_chain(vector_store, OLLAMA_LLM_MODEL)
    if qa_chain is None:
        return

    print("\n--- Ubestream.com Q&A Assistant ---")
    print("Type your questions about Ubestream.com. Type 'exit' to quit.")

    while True:
        query = input("\nYour question: ")
        if query.lower() == 'exit':
            print("Exiting Q&A assistant. Goodbye!")
            break

        if not query.strip():
            print("Please enter a question.")
            continue

        try:
            start_time = time.time() # Record start time
            # Invoke the RAG chain with the user's query
            result = qa_chain.invoke({"query": query})
            end_time = time.time() # Record end time
            
            print("\n--- Answer ---")
            print(result["result"])
            print(f"\nInference time: {end_time - start_time:.2f} seconds") # Print inference time

            # Optional: If you ever need to see source documents again, uncomment these lines:
            # print("\n--- Source Documents ---")
            # for i, doc in enumerate(result["source_documents"]):
            #     print(f"Document {i+1} (Source: {doc.metadata.get('source', 'N/A')}):")
            #     print(doc.page_content[:200] + "...") # Print first 200 chars
            #     print("-" * 20)

        except Exception as e:
            print(f"An error occurred during Q&A: {e}")
            print("Please ensure your Ollama server is running and the specified models are pulled.")

if __name__ == "__main__":
    main()
