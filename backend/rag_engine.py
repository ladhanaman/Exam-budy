import os
import re  # <--- NEW IMPORT
import base64
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from pinecone import Pinecone
from dotenv import load_dotenv

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

# --- CONFIGURATION ---
INDEX_NAME = os.getenv("INDEX_NAME")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# 1. Setup Resources
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# We use the raw Pinecone client for precise ID-based updates
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# LangChain Store for Retrieval
vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings
)

# LLMs
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.4)
vision_model = ChatGroq(model_name="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.3)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def permanent_vision_analysis(doc):
    """
    Analyzes the image, updates Pinecone with the description, 
    and deletes the local image file.
    """
    if "image_paths" not in doc.metadata or not doc.metadata["image_paths"]:
        return ""

    img_path = doc.metadata["image_paths"][0]
    
    if not os.path.exists(img_path):
        return ""

    print(f"Analyzing & Integrating Image: {os.path.basename(img_path)}")
    
    try:
        b64_image = encode_image(img_path)
        msg = HumanMessage(content=[
            {"type": "text", "text": "Describe this diagram/image in detail. Focus on text, labels, and relationships shown. Be concise."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
        ])
        response = vision_model.invoke([msg])
        description = f"\n\n[IMAGE DESCRIPTION for {os.path.basename(img_path)}]: {response.content}"
    except Exception as e:
        print(f"Vision Failed: {e}")
        return ""
    
    new_text_content = doc.page_content + description
    
    try:
        current_vec = embeddings.embed_query(doc.page_content)
        search_filter = {
            "source": doc.metadata["source"],
            "page_number": doc.metadata["page_number"]
        }
        matches = index.query(vector=current_vec, top_k=1, filter=search_filter, include_metadata=True)
        
        if matches['matches']:
            target_id = matches['matches'][0]['id']
            print(f"Updating Vector ID: {target_id}")
            
            new_vec = embeddings.embed_query(new_text_content)
            new_metadata = doc.metadata.copy()
            new_metadata["image_paths"] = [] 
            new_metadata["text"] = new_text_content
            
            index.upsert(vectors=[{
                "id": target_id,
                "values": new_vec,
                "metadata": new_metadata
            }])
            os.remove(img_path)
            print("Local image deleted.")
            return description
            
    except Exception as e:
        print(f"Update failed: {e}")
        return description 

    return description

# --- NEW HELPER FUNCTION ---
def extract_page_number(query):
    """
    Looks for 'page X' or 'slide X' in the user query.
    Returns the integer if found, else None.
    """
    match = re.search(r"(?:page|slide)\s+(\d+)", query.lower())
    if match:
        return int(match.group(1))
    return None

def chat_with_data(query: str):
    print(f"\nUser Question: {query}")
    
    # 1. Check for Page Number in Query
    target_page = extract_page_number(query)
    search_kwargs = {"k": 5}
    
    if target_page:
        print(f"Targeted Search: Looking for Page {target_page}...")
        search_kwargs["filter"] = {"page_number": {"$eq": target_page}}
    
    # 2. Retrieve with Optional Filter
    results = vectorstore.similarity_search(query, **search_kwargs)
    
    if not results and target_page:
        return f"I looked specifically for page {target_page}, but I couldn't find a record for it in the database."

    context_text = ""
    
    # 3. Process Results
    for doc in results:
        # Check if this chunk has a pending image to analyze
        if "image_paths" in doc.metadata and doc.metadata["image_paths"]:
            vision_desc = permanent_vision_analysis(doc)
            doc.page_content += vision_desc
        
        context_text += f"\n-- Slide {doc.metadata.get('page_number')} --\n{doc.page_content}\n"

    # 4. Construct Final Prompt
    final_system_prompt = f"""
    You are an expert Exam Tutor. Answer the user's question using the provided context.
    
    CONTEXT:
    {context_text}
    
    INSTRUCTIONS:
    - If the answer is in the text, use it.
    - If you see [IMAGE DESCRIPTION], use that detail to explain diagrams.
    - Always cite the slide number (e.g., [Slide 5]) at the end of the response.
    - If you don't know, say "I can't find that in your notes.
    - And keep the response concise but make sure it's correct and contain all the information."
    """
    
    messages = [
        SystemMessage(content=final_system_prompt),
        HumanMessage(content=query)
    ]
    
    response = llm.invoke(messages)
    return response.content

if __name__ == "__main__":
    print(chat_with_data("Tell me everything in slide 57"))