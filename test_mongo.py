from pymongo import MongoClient
from dotenv import load_dotenv
import os
from datetime import datetime

# Load environment variables from .env file
load_dotenv()
MONGO_URL = os.getenv('MONGO_URL')

client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=3000)
db = client["Steam-Karnival"]
chats_col = db["whatsapp_chats_test"]  # Use a test collection

# Example user chat data
user_chat = {
    "user_mobile_number": "+911234567890",
    "user_timestamp": datetime.utcnow(),
    "user_question": "Hello! How are you?",
    "response": "I'm fine, how can I help you today?"
}

try:
    result = chats_col.insert_one(user_chat)
    print(f"Chat saved with id: {result.inserted_id}")
except Exception as e:
    print(f"MongoDB error: {e}")