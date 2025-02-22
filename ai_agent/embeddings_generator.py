import os
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

def load_menu_to_chroma(menu_file_path="data/menu.json", persist_directory="embeddings/menu_db"):
    """
    Loads menu data from a JSON file and stores it in ChromaDB.
    
    Args:
        menu_file_path (str): Path to the menu JSON file.
        persist_directory (str): Path to store the ChromaDB vector index.

    Returns:
        Chroma: The initialized Chroma vector store.
    """
    # Load API key
    load_dotenv()

    # Initialize OpenAI Embeddings
    embeddings = OpenAIEmbeddings()

    # Load menu data from JSON file
    with open(menu_file_path, "r") as file:
        menu_items = json.load(file)

    # Convert menu items to text format
    texts = [f"{item['name']} - ₹{item['price']}" for item in menu_items]

    # Store in ChromaDB
    vector_store = Chroma.from_texts(texts, embeddings, persist_directory=persist_directory)

    print("Menu data loaded from JSON and stored in ChromaDB!")
    
    return vector_store  # Returning the vector store

