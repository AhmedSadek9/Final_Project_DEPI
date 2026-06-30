from fastapi import APIRouter
from datetime import datetime
from bson import ObjectId
from pydantic import BaseModel

from backend.data.mongodb import (
    chat_sessions_collection,
    chat_messages_collection
)

router = APIRouter()


# ==========================
# Create New Chat
# ==========================

@router.post("/new-chat")
def new_chat():

    session = {

        "title": "New Chat",

        "created_at": datetime.utcnow()

    }

    result = chat_sessions_collection.insert_one(session)

    return {

        "session_id": str(result.inserted_id),

        "title": "New Chat"

    }
    
    
    # ==========================
# Get All Chats
# ==========================

@router.get("/chat-sessions")
def get_chat_sessions():

    sessions = []

    cursor = chat_sessions_collection.find().sort(
        "created_at",
        -1
    )

    for session in cursor:

        sessions.append({

            "id": str(session["_id"]),

            "title": session["title"]

        })

    return {

        "sessions": sessions

    }
    
    # ==========================
# Save Message
# ==========================

class MessageRequest(BaseModel):

    session_id: str

    sender: str

    message: str

# ==========================
# Rename Chat
# ==========================

class RenameChatRequest(BaseModel):

    session_id: str

    title: str


@router.post("/save-message")
def save_message(req: MessageRequest):

    chat_messages_collection.insert_one(

        {

            "session_id": req.session_id,

            "sender": req.sender,

            "message": req.message,

            "created_at": datetime.utcnow()

        }

    )

    return {

        "success": True

    }

# ==========================
# Rename Chat
# ==========================

@router.put("/rename-chat")
def rename_chat(req: RenameChatRequest):

    chat_sessions_collection.update_one(

        {

            "_id": ObjectId(req.session_id)

        },

        {

            "$set": {

                "title": req.title

            }

        }

    )

    return {

        "success": True

    }
    
    # ==========================
# Get Messages
# ==========================

@router.get("/chat-messages/{session_id}")
def get_messages(session_id: str):

    messages = []

    cursor = chat_messages_collection.find(

        {

            "session_id": session_id

        }

    ).sort(

        "created_at",

        1

    )

    for msg in cursor:

        messages.append(

            {

                "sender": msg["sender"],

                "text": msg["message"]

            }

        )

    return {

        "messages": messages

    }