import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

def reset_index():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("INDEX_NAME")

    # 1. Delete the wrong index if it exists
    if index_name in pc.list_indexes().names():
        print(f"🗑️ Deleting mismatched index '{index_name}'...")
        pc.delete_index(index_name)
        # Wait a moment for deletion to propagate
        time.sleep(5) 

    # 2. Create the correct index (Dimension 384)
    print(f"🛠️ Creating new index '{index_name}' with dimension=384...")
    pc.create_index(
        name=index_name,
        dimension=384,  # <--- CHANGED FROM 1024 TO 384
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
    
    # Wait for readiness
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)
        
    print("✅ Index reset complete! You can now run the ingestion script.")

if __name__ == "__main__":
    reset_index()