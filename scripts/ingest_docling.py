import os
import glob
import re
import time
import base64
import io
from pathlib import Path
from dotenv import load_dotenv

# Docling Imports
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat, DocItemLabel

# LangChain Imports
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from PIL import Image

load_dotenv()

# --- CONFIGURATION ---
# Use a smaller model for embedding to save costs/latency, matching your index dimension (384)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
INDEX_NAME = os.getenv("INDEX_NAME")

# Models for Analysis
# Summarizer for Global Context
llm_summarizer = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.4)
# Vision Model for Image Description
vision_model = ChatGroq(model_name="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.2)


def encode_image_base64(pil_image):
    """
    Optimizes and converts a PIL image to a base64 string.
    Optimization: Resize to max 768px, Convert to JPEG, Quality 85.
    """
    # 1. Resize if too large (Max dimension 768px is the cost/quality sweet spot)
    max_dim = 768
    if max(pil_image.size) > max_dim:
        pil_image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    
    # 2. Convert to RGB (in case of RGBA/PNG) to allow JPEG saving
    if pil_image.mode in ('RGBA', 'P'):
        pil_image = pil_image.convert('RGB')

    # 3. Save as JPEG with reduced quality
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG", quality=85, optimize=True)
    
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def describe_image(pil_image, page_no):
    """
    Generates a description for an image using the Vision model.
    """
    print(f"      [Vision] Scanning image on page {page_no}...")
    try:
        b64_img = encode_image_base64(pil_image)
        msg = HumanMessage(content=[
            {"type": "text", "text": "Describe this technical diagram, chart, or table row in detail. Focus on text labels, data values, and relationships. Do not just say 'image'."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
        ])
        # Low temperature for factual descriptions
        res = vision_model.invoke([msg])
        # Format explicitly so it can be searched
        return f"\n\n[IMAGE ANALYSIS (Page {page_no})]: {res.content}\n\n"
    except Exception as e:
        print(f"[Error] Vision analysis failed: {e}")
        return "\n[Image Content Not Analyzed]\n"

def get_global_context(full_markdown):
    """Summarizes the entire document to provide global context."""
    print("[Context] Generating Global Summary...")
    try:
        # Limit input to first 8000 chars to save tokens
        msg = HumanMessage(content=f"Summarize this academic document in exactly 2 sentences. Focus on the main subject matter:\n\n{full_markdown[:8000]}")
        res = llm_summarizer.invoke([msg])
        return res.content
    except Exception as e:
        print(f"      [Error] Summary failed: {e}")
        return "Academic Course Material."

def process_file_smartly(file_path):
    """
    Main processing pipeline:
    1. OCR & Layout Analysis (Docling)
    2. Image Description (Vision Model)
    3. Markdown Export (Table Structure)
    4. Parent-Child Chunking (Precision Retrieval)
    """
    # 1. Setup Docling Options
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True 
    pipeline_options.do_table_structure = True
    pipeline_options.generate_picture_images = True
    
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    
    print(f"[Process] Analyzing layout for: {file_path}")
    doc_result = converter.convert(file_path)
    
    # 2. Run Eager Vision Pass
    # We collect descriptions first, keyed by page number
    image_descriptions = {}
    
    print("   ... Running Vision Model on detected images")
    for item, level in doc_result.document.iterate_items():
        if item.label == DocItemLabel.PICTURE and hasattr(item, "image") and item.image:
            page_no = item.prov[0].page_no if item.prov else 1
            desc = describe_image(item.image.pil_image, page_no)
            
            if page_no not in image_descriptions:
                image_descriptions[page_no] = ""
            image_descriptions[page_no] += desc

    # 3. Export to Markdown
    # This solves the "Table Soup" problem by keeping structure
    full_markdown = doc_result.document.export_to_markdown()
    
    # 4. Inject Image Descriptions
    # We append the descriptions to the text so they are searchable.
    # While exact placement is hard, appending them ensures they exist in the index.
    injected_markdown = full_markdown
    
    if image_descriptions:
        print("   ... Injecting visual descriptions into text")
        injected_markdown += "\n\n--- DETAILED IMAGE DESCRIPTIONS ---\n"
        for p_no, desc in image_descriptions.items():
            injected_markdown += f"Page {p_no}:{desc}\n"

    # 5. Generate Global Context
    global_context = get_global_context(injected_markdown)
    print(f"   [Context] {global_context}")

    # 6. Parent-Child Chunking
    print("   ... Creating Parent-Child Chunks")
    
    # Parent Splitter: Large chunks (1200 chars) for the LLM to read
    parent_splitter = MarkdownTextSplitter(chunk_size=1200, chunk_overlap=100)
    # Child Splitter: Small chunks (400 chars) for the Index to find
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    
    parent_chunks = parent_splitter.split_text(injected_markdown)
    final_docs = []
    
    for i, parent_text in enumerate(parent_chunks):
        # Create Children from this Parent
        child_chunks = child_splitter.split_text(parent_text)
        
        for child_text in child_chunks:
            # Create the Document object
            # - page_content: The CHILD text (used for vector search)
            # - metadata: Contains the PARENT text (used for answer generation)
            doc = Document(
                page_content=child_text, 
                metadata={
                    "source": os.path.basename(file_path),
                    "chunk_id": f"P{i}_C{hash(child_text)}",
                    "parent_context": parent_text,
                    "global_context": global_context,
                    "is_parent_child": True
                }
            )
            final_docs.append(doc)

    print(f"[Done] Created {len(final_docs)} Parent-Child chunks.")
    return final_docs

def ingest(docs):
    if not docs: return
    print(f"[Upload] Ingesting {len(docs)} chunks to Pinecone...")
    PineconeVectorStore.from_documents(
        documents=docs, 
        embedding=embeddings, 
        index_name=INDEX_NAME
    )
    print("[Success] Ingestion complete.")

if __name__ == "__main__":
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
        
    files = glob.glob("uploads/*.pdf")
    if not files:
        print("[Warning] No PDFs found in uploads folder.")
    
    for f in files:
        docs = process_file_smartly(f)
        ingest(docs)