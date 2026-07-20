
"""
Query Router.

Decides how a query should be processed.
"""

from enum import Enum
from memory.memory import ConversationMemory

class QueryType(Enum):
    DOCUMENT = "document"
    MEMORY = "memory"
    CHAT = "chat"

def route_query(query: str)-> QueryType:
    query = query.lower()
    memory_keywords=[
        "previous",
        "earlier",
        "before",
        "last question",
        "history",
        "conversation",
        "remember"
    ]
    
    chat_keywords = [
        "hello",
        "hi",
        "hey",
        "good morning",
        "good evening"
    ]

    for keyword in memory_keywords:
        if keyword in query:
            return QueryType.MEMORY

    for keyword in chat_keywords:
        if keyword in query:
            return QueryType.CHAT

    return QueryType.DOCUMENT