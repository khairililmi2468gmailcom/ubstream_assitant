# prepare_rag_data.py

import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

# --- Configuration ---
# Input file containing the crawled text
INPUT_FILE = "ubestream_all_deep_content.txt"
# Directory to save the Chroma vector store
CHROMA_DB_DIR = "./chroma_db"
# Ollama embedding model to use
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"

def load_and_clean_data(file_path):
    """
    Loads text data from a file and performs initial cleaning.
    """
    print(f"Loading data from {file_path}...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        print("Data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Input file '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

    # Basic cleaning: remove multiple newlines, extra spaces, and specific crawler tags
    cleaned_text = re.sub(r'\n\s*\n', '\n', raw_text) # Replace multiple newlines with single
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip() # Replace multiple spaces with single
    
    # Remove specific crawler tags like [ALT: ...], [TITLE: ...], [INPUT_VALUE: ...]
    # These tags are useful for crawling but might add noise for RAG if not desired.
    # Adjust this regex if you want to keep some of these tags.
    cleaned_text = re.sub(r'\[ALT:.*?\]', '', cleaned_text)
    cleaned_text = re.sub(r'\[TITLE:.*?\]', '', cleaned_text)
    cleaned_text = re.sub(r'\[INPUT_VALUE:.*?\]', '', cleaned_text)
    cleaned_text = re.sub(r'\[TEXTAREA:.*?\]', '', cleaned_text)
    cleaned_text = re.sub(r'\[OPTION:.*?\]', '', cleaned_text)

    # Remove the "--- Content from: URL ---" headers as they are metadata, not content
    cleaned_text = re.sub(r'--- Content from:.*? ---\n', '', cleaned_text)

    print("Data cleaned.")
    return cleaned_text

def split_text_into_chunks(text):
    """
    Splits the cleaned text into smaller, manageable chunks for embedding.
    """
    print("Splitting text into chunks...")
    # Using RecursiveCharacterTextSplitter for robust splitting
    # It tries to split by different characters in order, preserving semantic units.
    # chunk_size: maximum size of each chunk (in characters)
    # chunk_overlap: overlap between consecutive chunks to maintain context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Adjust based on your data and model's context window
        chunk_overlap=200, # Common overlap value
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.create_documents([text])
    print(f"Text split into {len(chunks)} chunks.")
    return chunks

def create_and_save_vector_store(chunks, embedding_model_name, db_dir):
    """
    Creates embeddings for text chunks and saves them to a ChromaDB vector store.
    """
    print(f"Initializing Ollama embeddings with model: {embedding_model_name}...")
    try:
        embeddings = OllamaEmbeddings(model=embedding_model_name)
        print("Embedding model initialized.")
    except Exception as e:
        print(f"Error initializing Ollama embeddings. Make sure Ollama server is running and model '{embedding_model_name}' is pulled. Error: {e}")
        return None

    print(f"Creating and saving ChromaDB vector store to {db_dir}...")
    try:
        # Create the vector store from the documents and embeddings
        # This will automatically compute embeddings for each chunk
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=db_dir
        )
        vector_store.persist() # Ensure the database is written to disk
        print("Vector store created and saved successfully.")
        return vector_store
    except Exception as e:
        print(f"Error creating or saving vector store: {e}")
        return None

def main():
    """
    Main function to orchestrate data loading, cleaning, splitting, and vector store creation.
    """
    # 1. Load and Clean Data
    cleaned_text = load_and_clean_data(INPUT_FILE)
    if cleaned_text is None:
        return

    # 2. Split Text into Chunks
    text_chunks = split_text_into_chunks(cleaned_text)
    if not text_chunks:
        print("No text chunks created. Exiting.")
        return

    # 3. Create and Save Vector Store
    vector_store = create_and_save_vector_store(text_chunks, OLLAMA_EMBEDDING_MODEL, CHROMA_DB_DIR)
    if vector_store:
        print("\nData preparation complete. You can now run app.py for Q&A.")

if __name__ == "__main__":
    main()
