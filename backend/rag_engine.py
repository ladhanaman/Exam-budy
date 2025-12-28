import os
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

def chat_with_data(query: str):
    print(f"\n[Query] {query}")
    
    # 1. Search Pinecone
    # We search for the best *child* chunks (high precision)
    results = vectorstore.similarity_search(query, k=5)
    
    if not results:
        return "I could not find any relevant information in your notes."

    # 2. Resolve Context (Small-to-Big)
    # We grab the 'parent_context' from metadata if it exists.
    # We also deduplicate parents (if multiple children from same parent are found).
    unique_contexts = set()
    final_context_text = ""
    global_context = ""

    for doc in results:
        # Extract Global Context (just once)
        if not global_context:
            global_context = doc.metadata.get("global_context", "")

        # Decide whether to use Parent or Child text
        if "parent_context" in doc.metadata and doc.metadata["parent_context"]:
            content = doc.metadata["parent_context"]
        else:
            content = doc.page_content

        # Deduplicate
        if content not in unique_contexts:
            unique_contexts.add(content)
            # Add Source/Page info for the LLM
            page_num = doc.metadata.get("page_number", "Unknown")
            # If page number is missing (common in Markdown export), source is helpful
            source_lbl = doc.metadata.get("source", "Notes")
            final_context_text += f"\n-- Source: {source_lbl} --\n{content}\n"

    # 3. Construct System Prompt
    system_prompt = f"""
    You are an expert Exam Tutor. 
    
    GLOBAL SUBJECT CONTEXT:
    {global_context}
    
    SPECIFIC NOTES CONTENT:
    {final_context_text}
    
    INSTRUCTIONS:
    - Answer the user's question primarily using the SPECIFIC NOTES CONTENT.
    - If you encounter text like "[IMAGE ANALYSIS]", treat it as a description of a diagram/image in the notes.
    - If the answer involves a list or steps, format it clearly.
    - Always cite the source (e.g.,) at the end of the sentence.
    - If the information is not in the context, state "I cannot find that information in the provided notes."
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]
    
    # 4. Generate Response
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"[Error] Failed to generate response: {e}"

# --- TEST BLOCK ---
if __name__ == "__main__":
    # Test Question
    test_q = "Explain the DDOS diagram"
    print(chat_with_data(test_q))