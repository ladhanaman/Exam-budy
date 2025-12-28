import os
import base64
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

# Optimization: Check current path to handle imports correctly
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

# 1. Setup Resources
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(
    index_name=os.getenv("INDEX_NAME"),
    embedding=embeddings
)

# Main Tutor LLM
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.3)

# Vision LLM
vision_model = ChatGroq(model_name="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)

def encode_image(image_path):
    """Helper to convert local image to base64 for the API"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_image_on_demand(image_path):
    """
    This runs ONLY if a retrieved chunk has an image.
    It asks the Vision model to describe it specifically for the user's question.
    """
    try:
        print(f"Analyzing Image: {image_path}")
        b64_image = encode_image(image_path)
        
        msg = HumanMessage(content=[
            {"type": "text", "text": "Describe this diagram/image in detail. Focus on text, labels, and relationships shown."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
        ])
        response = vision_model.invoke([msg])
        return f"\n[VISUAL CONTEXT from {os.path.basename(image_path)}]: {response.content}\n"
    except Exception as e:
        print(f"Vision Failed: {e}")
        return "[Error analyzing image]"

def chat_with_data(query: str):
    """
    Main RAG function for Exam Prep.
    Searches the entire knowledge base (Slides + Images).
    """
    print(f"\n🤔 User Question: {query}")
    
    # 1. Retrieve Top 3 Chunks
    # We fetch 3 chunks to get enough context
    results = vectorstore.similarity_search(query, k=3)
    
    context_text = ""
    visual_context = ""
    
    # 2. Process Results
    for doc in results:
        # Add the text content
        context_text += f"\n-- Slide {doc.metadata.get('page_number')} --\n{doc.page_content}\n"
        
        # 3. Check for Images (The "Lazy Vision" Step)
        if "image_paths" in doc.metadata and doc.metadata["image_paths"]:
            # We only look at the first image of the top result to save time/latency
            img_path = doc.metadata["image_paths"][0]
            
            # Verify file exists before trying to read
            if os.path.exists(img_path):
                visual_context += analyze_image_on_demand(img_path)

    # 4. Construct Final Prompt
    final_system_prompt = f"""
    You are an expert Exam Tutor. Answer the user's question using the provided context.
    
    CONTEXT FROM SLIDES:
    {context_text}
    
    VISUAL CONTEXT (From Diagrams):
    {visual_context}
    
    INSTRUCTIONS:
    - If the answer is in the text, use it.
    - If the answer is in the Visual Context, explicitly mention "According to the diagram..."
    - If you don't know, say "I can't find that in your notes."
    """
    
    messages = [
        SystemMessage(content=final_system_prompt),
        HumanMessage(content=query)
    ]
    
    # 5. Generate Answer
    response = llm.invoke(messages)
    return response.content

# --- TEST AREA ---
if __name__ == "__main__":
    # Test: Standard RAG
    print(chat_with_data("Explain Steganography"))