import os
import re
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

# --- CONFIGURATION ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(
    index_name=os.getenv("INDEX_NAME"),
    embedding=embeddings
)

# Use a high-quality model for the final answer
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.3)

def extract_page_numbers(query):
    """
    Detects multiple page numbers.
    Matches: "page 10", "slide 5 and 6", "p. 12"
    Returns a list of integers, e.g., [5, 6]
    """
    # Find all digits preceded by page indicators
    matches = re.findall(r"(?:page|slide|p\.)\s*(\d+)", query.lower())
    if matches:
        # Convert to distinct integers
        return list(set(int(m) for m in matches))
    return []

def chat_with_data(query: str):
    print(f"\n[Query] {query}")
    
    # 1. Check for Targeted Page Search
    target_pages = extract_page_numbers(query)
    search_kwargs = {"k": 5} # Fetch top 5 chunks
    
    if target_pages:
        print(f"      🎯 Targeted Search: Filtering for Slides {target_pages}")
        # [FIX 5] Use '$in' operator for multi-page filtering
        search_kwargs["filter"] = {"page_number": {"$in": target_pages}}

    # 2. Search Pinecone
    results = vectorstore.similarity_search(query, **search_kwargs)
    
    if not results:
        if target_pages:
            return f"I looked specifically for Slides {target_pages}, but I couldn't find them in the database."
        return "I could not find any relevant information in your notes."

    # 3. Resolve Context (Page-Centric Deduplication)
    unique_content = set()
    final_context_text = ""
    global_context = ""

    for doc in results:
        # Extract Global Context (just once, favoring non-empty values)
        if not global_context and doc.metadata.get("global_context"):
            global_context = doc.metadata.get("global_context")

        # LOGIC:
        # We prioritize 'parent_context' (Full Page Text) if available.
        # Note: Since we optimized storage, 'parent_context' might be empty if 
        # this chunk is not the first one. In that case, we fall back to 'page_content'.
        if "parent_context" in doc.metadata and doc.metadata["parent_context"]:
            content = doc.metadata["parent_context"]
        else:
            content = doc.page_content

        # Get Page Number
        page_num = doc.metadata.get("page_number", "Unknown")

        # Deduplicate
        if content not in unique_content:
            unique_content.add(content)
            final_context_text += f"\n-- Slide {page_num} --\n{content}\n"

    # 4. Construct System Prompt
    system_prompt = f"""
    You are an expert Exam Tutor. 
    
    GLOBAL SUBJECT CONTEXT:
    {global_context}
    
    SPECIFIC NOTES CONTENT:
    {final_context_text}
    
    INSTRUCTIONS:
    - Answer the user's question primarily using the SPECIFIC NOTES CONTENT.
    - If you encounter text like "[IMAGE ANALYSIS]", treat it as a description of a diagram/image on that slide.
    - If the user asks about a specific slide (e.g., "Explain Slide 47"), focus ONLY on the content labeled "-- Slide 47 --".
    - Always cite the slide number (e.g., [Slide 5]) at the end of the response.
    - If the information is not in the context, state "I cannot find that information in the provided notes."
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]
    
    # 5. Generate Response
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"[Error] Failed to generate response: {e}"

# --- TEST BLOCK ---
if __name__ == "__main__":
    print(chat_with_data("Explain DDOS"))
    print("--------------------------------------------------")
    print(chat_with_data("Compare the diagram on page 47 and page 48"))