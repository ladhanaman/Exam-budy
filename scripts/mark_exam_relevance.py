import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()

# --- CONFIG ---
INDEX_NAME = os.getenv("INDEX_NAME")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Clients
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

def extract_questions_from_pyq(pyq_text):
    """
    Uses LLM to clean raw exam paper text into a clean Python list of questions.
    """
    prompt = f"""
    You are an exam parser. Extract the technical questions from this exam paper text. 
    Return ONLY a list of strings separated by newlines. Do not include question numbers (1., 2., etc).
    
    RAW EXAM TEXT:
    {pyq_text}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    # Split by newline and filter empty strings
    questions = [q.strip() for q in response.content.split("\n") if q.strip()]
    return questions

def mark_slides_as_important(questions):
    """
    Iterates through questions, finds matching slides, and updates metadata.
    """
    print(f"🔥 processing {len(questions)} exam questions...")
    
    count_updates = 0
    
    for q in questions:
        # Basic cleanup
        q = q.replace("- ", "").strip()
        if len(q) < 5: continue
            
        print(f"   🔍 Searching matches for: '{q[:40]}...'")
        
        # 1. Embed the question
        q_vector = embeddings.embed_query(q)
        
        # 2. Semantic Search in Pinecone
        # We find top 3 matches for this specific question
        results = index.query(
            vector=q_vector,
            top_k=3,
            include_metadata=True
        )
        
        # 3. Update Metadata if High Confidence
        for match in results['matches']:
            # Threshold: 0.60 (adjust based on your embedding model performance)
            if match['score'] > 0.60: 
                print(f"       ✅ Linked to Slide {match['metadata'].get('page_number')} (Score: {match['score']:.2f})")
                
                # UPDATE: Set exam_relevance = True
                # This is a partial update, it won't delete other metadata
                index.update(
                    id=match['id'],
                    set_metadata={"exam_relevance": True}
                )
                count_updates += 1

    print(f"\n🎉 Success! Marked {count_updates} chunks as High Relevance.")

if __name__ == "__main__":
    # EXAMPLE USAGE
    # In a real app, you would read this from a user-uploaded PDF using Docling
    dummy_pyq_text = """
    1. Explain the vessel image example in Steganography with a diagram.
    2. Define active attacks vs passive attacks.
    3. What is the difference between Block Cipher and Stream Cipher?
    4. Explain the RSA algorithm.
    """
    
    print("Parsing PYQ...")
    qs = extract_questions_from_pyq(dummy_pyq_text)
    
    print(f"Extracted {len(qs)} questions.")
    mark_slides_as_important(qs)