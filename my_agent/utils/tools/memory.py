# tools/memory.py
import os
from typing import List, Dict, Optional
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool
from datetime import datetime
import json
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Initialization
# -----------------------------
_embeddings = OpenAIEmbeddings(
    openai_api_key=os.environ.get("API_KEY"),
    openai_api_base=os.environ.get("API_BASE_URL")
)

_vectorstore = Chroma(
    collection_name="agent_memory",
    embedding_function=_embeddings,
    persist_directory="./chroma_db"
)

# -----------------------------
# Document wrapper
# -----------------------------
class Document:
    def __init__(self, page_content: str, metadata: Optional[Dict] = None):
        self.page_content = page_content
        self.metadata = metadata or {}

# -----------------------------
# Memory Functions
# -----------------------------

def retrieve_memories(query: str, k: int = 5) -> List[Dict]:
    """Retrieve k most relevant memories based on query."""
    try:
        results = _vectorstore.similarity_search(query, k=k)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "timestamp": doc.metadata.get("timestamp", "unknown")
            }
            for doc in results
        ]
    except Exception as e:
        print(f"Memory retrieval error: {e}")
        return []

def add_memory(content: str, metadata: Optional[Dict] = None, memory_type: str = "interaction"):
    """Store a new memory with timestamp."""
    meta = metadata or {}
    meta.update({
        "timestamp": datetime.now().isoformat(),
        "type": memory_type
    })
    doc = Document(page_content=content, metadata=meta)
    _vectorstore.add_documents([doc])

def add_model_training_memory(model_type: str, purpose: str, parameters: Dict, result_summary: str):
    """Store model training memory."""
    content = f"""Model: {model_type}
Purpose: {purpose}
Parameters: {json.dumps(parameters, indent=2)}
Result: {result_summary}"""
    add_memory(
        content=content,
        metadata={"model_type": model_type, "purpose": purpose, "parameters": json.dumps(parameters)},
        memory_type="model_training"
    )

def clear_all_memories() -> bool:
    """Clear all memories."""
    global _vectorstore  # move this to the top
    try:
        _vectorstore.delete_collection()
        _vectorstore = Chroma(
            collection_name="agent_memory",
            embedding_function=_embeddings,
            persist_directory="./chroma_db"
        )
        return True
    except Exception as e:
        print(f"Error clearing memories: {e}")
        return False

def format_memories_for_prompt(memories: List[Dict]) -> str:
    """Format memories for injection into system prompt."""
    if not memories:
        return ""
    
    formatted = "\n=== RELEVANT CONTEXT FROM PAST INTERACTIONS ===\n"
    for i, mem in enumerate(memories, 1):
        timestamp = mem.get("metadata", {}).get("timestamp", "unknown")
        mem_type = mem.get("metadata", {}).get("type", "unknown")
        formatted += f"\n[{i}] ({mem_type} - {timestamp}):\n{mem['content']}\n"
    formatted += "=== END CONTEXT ===\n"
    return formatted

# -----------------------------
# LangChain Tool Wrappers
# -----------------------------

# -----------------------------
# LangChain Tool Wrappers
# -----------------------------

@tool
def search_memory_tool(query: str, max_results: int = 5) -> str:
    """Search past memories for relevant context based on a query.
    
    Args:
        query: Natural language query to search memories.
        max_results: Maximum number of relevant memories to return.
        
    Returns:
        Formatted string of found memories, or a message if none found.
    """
    memories = retrieve_memories(query, k=max_results)
    if not memories:
        return "No relevant memories found."
    
    result = f"Found {len(memories)} relevant memories:\n\n"
    for i, mem in enumerate(memories, 1):
        timestamp = mem.get("timestamp", "unknown")
        mem_type = mem.get("metadata", {}).get("type", "unknown")
        result += f"{i}. [{mem_type}] ({timestamp})\n{mem['content']}\n\n"
    return result


@tool
def add_memory_tool(content: str, memory_type: str = "note", model_type: str = "", purpose: str = "") -> str:
    """Store a new memory for future reference.
    
    Args:
        content: Information to store.
        memory_type: Type of memory (note, model_training, insight, preference).
        model_type: Optional; type of model if relevant.
        purpose: Optional; purpose or use case of the memory.
        
    Returns:
        Confirmation message of stored memory.
    """
    metadata = {}
    if model_type:
        metadata["model_type"] = model_type
    if purpose:
        metadata["purpose"] = purpose
    add_memory(content=content, metadata=metadata, memory_type=memory_type)
    return f"Memory stored successfully (type: {memory_type})"


@tool
def clear_memory_tool(confirm: str = "no") -> str:
    """Clear all stored memories. This action cannot be undone.
    
    Args:
        confirm: Must be "yes" to actually clear memories.
        
    Returns:
        Confirmation message of memory clearance status.
    """
    if confirm.lower() != "yes":
        return "Memory clear cancelled. Set confirm='yes' to actually clear all memories."
    success = clear_all_memories()
    return "All memories have been cleared." if success else "Failed to clear memories. Check logs for details."


@tool
def get_model_purpose_tool(model_type: str) -> str:
    """Retrieve past purposes and use cases for a specific model type from memory.
    
    Args:
        model_type: Type of model (e.g., perceptron, mlp, som, fuzzy, gp, ga, aco, pso).
        
    Returns:
        Formatted string of past usage or a message if none found.
    """
    query = f"{model_type} model purpose use case training"
    memories = retrieve_memories(query, k=3)
    if not memories:
        return f"No past usage found for {model_type}. This might be the first time using this model."
    
    result = f"Past usage of {model_type}:\n\n"
    for i, mem in enumerate(memories, 1):
        result += f"{i}. {mem['content']}\n\n"
    return result
