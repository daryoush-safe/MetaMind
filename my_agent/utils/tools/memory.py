# tools/memory.py
import os
from typing import List, Dict, Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from datetime import datetime
import json

# -----------------------------
# Embeddings setup
# -----------------------------
def get_embeddings():
    """Get embeddings function - uses local model to avoid API issues."""
    try:
        # Try to use sentence-transformers (local, no API needed)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        return embeddings
    except ImportError:
        print("WARNING: sentence-transformers not installed. Install with: pip install sentence-transformers langchain-huggingface")
        # Fallback: try OpenAI only if using real OpenAI API
        api_base = os.environ.get("API_BASE_URL", "")
        if "api.openai.com" in api_base:
            return OpenAIEmbeddings(
                openai_api_key=os.environ.get("API_KEY"),
                openai_api_base=api_base
            )
        else:
            raise Exception("Cannot initialize embeddings. Install sentence-transformers: pip install sentence-transformers langchain-huggingface")

def get_vectorstore():
    embeddings = get_embeddings()
    vectorstore = Chroma(
        collection_name="agent_memory",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    return vectorstore

# -----------------------------
# Memory Functions
# -----------------------------

def retrieve_memories(query: str, k: int = 5) -> List[Dict]:
    """Retrieve k most relevant memories based on query."""
    if not query:
        query = "general query"
    if not isinstance(query, str):
        query = str(query)
    
    try:
        vectorstore = get_vectorstore()
        collection = vectorstore._collection
        if collection.count() == 0:
            print("Memory database is empty - no memories to retrieve")
            return []
        
        results = vectorstore.similarity_search(query, k=k)
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
    """Store a new memory safely with timestamp."""
    try:
        # Validate and sanitize content
        if content is None:
            content = "Empty memory"
        elif not isinstance(content, str):
            content = str(content)
        
        content = content.strip()
        if len(content) == 0:
            content = "Empty memory placeholder"
        
        if len(content) < 3:
            content = f"Note: {content}"
        
        # Sanitize metadata
        meta = metadata or {}
        safe_meta = {}
        for k, v in meta.items():
            if v is None:
                safe_meta[k] = "none"
            else:
                safe_meta[k] = str(v)
        
        safe_meta.update({
            "timestamp": datetime.now().isoformat(),
            "type": memory_type
        })
        
        doc = Document(page_content=content, metadata=safe_meta)
        vectorstore = get_vectorstore()
        vectorstore.add_documents([doc])
        print(f"✓ Memory stored: {memory_type}")
        
    except Exception as e:
        print(f"Memory storage error: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")

def add_model_training_memory(model_type: str, purpose: str, parameters: Dict, result_summary: str):
    """Store model training memory."""
    try:
        model_type = str(model_type) if model_type else "unknown"
        purpose = str(purpose) if purpose else "not specified"
        result_summary = str(result_summary) if result_summary else "no results"
        
        params_str = json.dumps(parameters, indent=2, default=str)
        if len(params_str) > 500:
            params_str = params_str[:500] + "..."
        
        content = f"""Model: {model_type}
Purpose: {purpose}
Parameters: {params_str}
Result: {result_summary}"""
        
        add_memory(
            content=content,
            metadata={
                "model_type": model_type,
                "purpose": purpose,
                "parameters": params_str
            },
            memory_type="model_training"
        )
    except Exception as e:
        print(f"Error storing model training memory: {e}")

def clear_all_memories() -> bool:
    """Clear all memories."""
    try:
        vectorstore = get_vectorstore()
        embeddings = get_embeddings()
        vectorstore.delete_collection()
        vectorstore = Chroma(
            collection_name="agent_memory",
            embedding_function=embeddings,
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
        content = mem.get("content", "")
        if len(content) > 500:
            content = content[:500] + "..."
        formatted += f"\n[{i}] ({mem_type} - {timestamp}):\n{content}\n"
    formatted += "=== END CONTEXT ===\n"
    return formatted

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
    if not query or len(query.strip()) == 0:
        return "Error: Query cannot be empty"
    
    memories = retrieve_memories(query, k=max_results)
    if not memories:
        return "No relevant memories found."
    
    result = f"Found {len(memories)} relevant memories:\n\n"
    for i, mem in enumerate(memories, 1):
        timestamp = mem.get("timestamp", "unknown")
        mem_type = mem.get("metadata", {}).get("type", "unknown")
        content = mem.get("content", "")
        if len(content) > 300:
            content = content[:300] + "..."
        result += f"{i}. [{mem_type}] ({timestamp})\n{content}\n\n"
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
    if not content or len(content.strip()) == 0:
        return "Error: Cannot store empty memory"
    
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
    """Retrieve the purpose and use cases for a specific model type from memory.
    
    Args:
        model_type: Type of model (perceptron, mlp, som, hopfield, fuzzy, gp, ga, aco, pso)
    
    Returns:
        Past purposes and use cases for this model type
    """
    if not model_type or len(model_type.strip()) == 0:
        return "Error: Model type cannot be empty"
    
    query = f"{model_type} model purpose use case training"
    memories = retrieve_memories(query, k=3)
    
    if not memories:
        return f"No past usage found for {model_type}. This might be the first time using this model."
    
    result = f"Past usage of {model_type}:\n\n"
    for i, mem in enumerate(memories, 1):
        content = mem.get("content", "")
        if len(content) > 300:
            content = content[:300] + "..."
        result += f"{i}. {content}\n\n"
    
    return result