import os
import json
import random
import gradio as gr
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.tools import Tool
from langchain.agents import initialize_agent
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage
from ai_agent.embeddings_generator import load_menu_to_chroma

# Load Environment Variables
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# Load Menu into ChromaDB
vector_store = load_menu_to_chroma()

# File path for storing orders
ORDER_DB = "data/orders.json"

# Order Storage
orders = []

def greet_user():
    """Generates a greeting message from the AI."""
    greeting = AIMessage(content="Hello! Welcome to our Zukachin restaurant. 🍽️ What would you like to order?")
    return greeting.content  

def get_menu_details(query: str):
    """Uses LLM to understand the query and search ChromaDB accordingly."""
    
    # Let the LLM decide if the user is asking about the menu
    llm_prompt = f"Does the user want to see the menu? Query: {query}"
    llm_response = llm.predict_messages([HumanMessage(content=llm_prompt)]).content.lower()
    
    if "yes" in llm_response:
        docs = vector_store.similarity_search("", k=10)  # Fetch all menu items
        menu_items = "\n".join([doc.page_content for doc in docs])
        return f"📜 Here's our menu:\n\n{menu_items}\n\nLet me know what you'd like to order! 🍽️"

    # Let the LLM rephrase the query to extract relevant details
    reformulated_query = llm.predict_messages([HumanMessage(content=f"Rephrase this query to match menu items: {query}")]).content
    
    # Search ChromaDB using the LLM-generated query
    docs = vector_store.similarity_search(reformulated_query, k=3)

    if not docs:
        return "❌ Sorry, I couldn't find that item on our menu. Try asking for something else!"

    found_items = [doc.page_content for doc in docs]

    return f"✅ Here's what I found:\n\n" + "\n".join(found_items) + "\n\nWould you like to place an order? 😊"

def place_order(query: str):
    """Uses LLM to extract dish names, validates them, and confirms the order."""

    # Step 1: Use LLM to extract dish names from the user's order request
    llm_prompt = f"Extract only the food item names from this order request: '{query}'. Return them as a comma-separated list."
    extracted_items = llm.predict_messages([HumanMessage(content=llm_prompt)]).content
    
    # Convert extracted dish names into a list
    order_items = [item.strip() for item in extracted_items.split(",") if item.strip()]

    if not order_items:
        return "❌ Sorry, I couldn't identify any valid dishes from your request. Please order again!"

    # Step 2: Validate extracted dishes against ChromaDB
    total_price = 0
    ordered_items = []

    for item in order_items:
        docs = vector_store.similarity_search(item, k=1)  # Search for the dish in ChromaDB
        if docs:
            item_details = docs[0].page_content  # Format: "Dish Name - ₹Price"
            name, price = item_details.split(" - ₹")
            total_price += int(price)
            ordered_items.append(name)

    # Step 3: Ensure only valid dishes are confirmed
    if not ordered_items:
        return "❌ None of the items you requested are available. Please check the menu and try again!"

    # Step 4: Store the order
    order_data = {
        "items": ordered_items,
        "total_price": total_price,
        "order_id": random.randint(1000, 9999)  # Generate a random order ID
    }
    orders.append(order_data)
    save_order(order_data)

    return f"✅ Order Confirmed!\nItems: {', '.join(ordered_items)}\nTotal Bill: ₹{total_price}\nOrder ID: {order_data['order_id']} 🎉"

def get_delivery_time(query: str):
    """Asks the user for Dine-In or Home Delivery before giving an estimated delivery time."""
    
    # Use LLM to determine if the user mentioned delivery
    llm_prompt = f"Does the user want Dine-In or Home Delivery? Query: '{query}'. Answer with 'dine-in', 'delivery'."
    llm_response = llm.predict_messages([HumanMessage(content=llm_prompt)]).content.lower()

    if "dine-in" in llm_response:
        return "🍽️ Great choice! Your table is reserved. Enjoy your meal!"
    
    elif "delivery" in llm_response:
        delivery_time = random.randint(20, 45)  # Generate a random time between 20-45 min
        return f"🚴 Your order will be delivered in approximately **{delivery_time} minutes**."
    
    else:
        return "Would you like **Dine-In** or **Home Delivery**? 😊"

def save_order(order_data):
    """Save the order to a JSON file for tracking."""
    try:
        # Load existing orders
        with open(ORDER_DB, "r") as file:
            existing_orders = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_orders = []

    # Append the new order
    existing_orders.append(order_data)

    # Save back to file
    with open(ORDER_DB, "w") as file:
        json.dump(existing_orders, file, indent=4)

def bid_farewell(query: str):
    """Uses LLM to determine if the user wants to end the conversation and responds accordingly."""
    
    llm_prompt = f"Does this message indicate that the user is leaving or ending the conversation? '{query}' Answer with 'yes' or 'no'."
    llm_response = llm.predict_messages([HumanMessage(content=llm_prompt)]).content.lower()

    if "yes" in llm_response:
        return "🙏 Thank you for visiting Zukachin Restaurant! 🍽️ We hope to see you again soon. Have a great day! 😊"
    
    return None  # If not a farewell, do nothing


# Define LangChain Tools
menu_tool = Tool(name="Menu Lookup", func=get_menu_details, description="Fetch menu details.")
order_tool = Tool(name="Order Placement", func=place_order, description="Place an order for food.")
delivery_tool = Tool(name="Delivery Time", func=get_delivery_time, description="Get estimated delivery time.")
farewell_tool  = Tool(name="Farewell Handler", func=bid_farewell, description="Detects when the user wants to leave and bids farewell.")

# Initialize Memory and Agent
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools=[menu_tool, order_tool, delivery_tool,farewell_tool],
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,
    memory=memory
)

# Expose chatbot function for UI
def chatbot_interface(user_input, chat_history=None):
    """Handles chatbot interaction with Gradio"""

    # First, check if the LLM decides to say goodbye
    farewell_message = bid_farewell(user_input)
    if farewell_message:
        return farewell_message  # Immediately return farewell message
    
    try:
        response = agent.run(user_input)
        return response if response else "🤖 Sorry, I didn't understand that."
    except Exception as e:
        return f"⚠️ Error: {str(e)}"


with gr.Blocks() as demo:
    gr.Markdown("# 🍽️ Zukachin Restaurant Chatbot")
    gr.Markdown(greet_user())  # Display greeting message
    chatbot = gr.ChatInterface(fn=chatbot_interface, title="Restaurant Chatbot")

# Launch Gradio App
if __name__ == "__main__":
    demo.launch()

