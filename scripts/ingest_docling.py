import os
import glob
import re
import time
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

load_dotenv()

# --- CONFIG ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
INDEX_NAME = os.getenv("INDEX_NAME")

# Models for Eager Analysis
llm_summarizer = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.3)
vision_model = ChatGroq(model_name="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)

def encode_image_base64(pil_image):
    """Converts PIL image to base64 for Vision Model"""
    import io
    import base64
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def describe_image(pil_image, page_no):
    """
    EAGER VISION: Generates a description immediately during ingestion.
    """
    print(f"      👁️ Scanning image on page {page_no}...")
    try:
        b64_img = encode_image_base64(pil_image)
        msg = HumanMessage(content=[
            {"type": "text", "text": "Describe this technical diagram or table row concisely. Focus on text labels, relationships, and data values. Do not just say 'image'."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
        ])
        # We use a lower temperature for factual descriptions
        res = vision_model.invoke([msg])
        return f"\n\n> **[IMAGE ANALYSIS (Page {page_no})]:** {res.content}\n\n"
    except Exception as e:
        print(f"      ❌ Vision Error: {e}")
        return "\n> [Image Content Not Analyzed]\n"

def get_global_context(full_markdown):
    """Summarizes the entire document for context grounding."""
    print("🌍 Generating Global Context...")
    try:
        msg = HumanMessage(content=f"Summarize this document in 2 sentences. Focus on the main subject matter:\n\n{full_markdown[:8000]}")
        res = llm_summarizer.invoke([msg])
        return res.content
    except Exception as e:
        return "Academic Course Material."

def process_file_smartly(file_path):
    # 1. Setup Docling with Table Structure & OCR
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True 
    pipeline_options.do_table_structure = True
    pipeline_options.generate_picture_images = True
    
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    
    print(f"🧐 Parsing & Structuring: {file_path}")
    doc_result = converter.convert(file_path)
    
    # 2. Extract "Eager Vision" Descriptions
    # We iterate through items to find images, generate descriptions, 
    # and we will inject them into the Markdown flow using a replacement map.
    image_descriptions = {}
    
    print("   ... Running Eager Vision Pass (This fixes Blind Search)")
    for item, level in doc_result.document.iterate_items():
        if item.label == DocItemLabel.PICTURE and hasattr(item, "image") and item.image:
            # Generate ID to link text to image
            # Docling items usually don't have stable IDs in the export, 
            # so we will append descriptions to the page text buffer.
            page_no = item.prov[0].page_no if item.prov else 1
            desc = describe_image(item.image.pil_image, page_no)
            
            if page_no not in image_descriptions:
                image_descriptions[page_no] = ""
            image_descriptions[page_no] += desc

    # 3. Export to Markdown (Fixes Table Soup)
    # Docling's markdown export handles tables beautifully automatically.
    full_markdown = doc_result.document.export_to_markdown()
    
    # 4. Inject Vision Descriptions into Markdown
    # Since exact injection is hard, we append the image descriptions 
    # to the end of their respective pages (conceptually) or just chunks.
    # A simpler way: We will rely on text splitting to carry them.
    # Let's simple append all descriptions for a page "near" that page's text markers if possible.
    # For now, we will append them to the global text to ensure they are searchable.
    # Better strategy: We construct the text page-by-page manually to insert images correctly.
    
    # RE-STRATEGY: Build Markdown Page-by-Page
    final_text_buffer = ""
    global_context = get_global_context(full_markdown)
    print(f"📝 Context: {global_context}")

    # We manually split the full markdown by page markers if Docling adds them, 
    # but Docling export is continuous. 
    # Let's just append the image descriptions at the start of the document or relevant sections? 
    # No, that loses context.
    
    # OPTIMAL PATH: We just append the image analysis text to the full markdown 
    # effectively treating it as "Appendix" content that is searchable.
    # *However*, for RAG, it's better if it's close. 
    # Let's inject all image descriptions at the very top of the text so they are prioritized?
    # No, let's inject them at the end.
    
    injected_markdown = f"Global Context: {global_context}\n\n" + full_markdown
    
    for p_no, desc in image_descriptions.items():
        # Try to insert near "Page X" marker if it exists, otherwise append
        marker = f"## Page {p_no}"
        if marker in injected_markdown:
            injected_markdown = injected_markdown.replace(marker, f"{marker}\n{desc}")
        else:
            injected_markdown += f"\n\n--- Visuals from Page {p_no} ---\n{desc}"

    # 5. Parent-Child Chunking Strategy
    print("   ... Applying Parent-Child Chunking")
    
    # PARENT SPLITTER: Large chunks (Context)
    parent_splitter = MarkdownTextSplitter(chunk_size=1200, chunk_overlap=100)
    # CHILD SPLITTER: Small chunks (Precision Hooks)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    
    parent_chunks = parent_splitter.split_text(injected_markdown)
    final_docs = []
    
    for i, parent_text in enumerate(parent_chunks):
        # Create Children from this Parent
        child_chunks = child_splitter.split_text(parent_text)
        
        for child_text in child_chunks:
            # MAGIC HAPPENS HERE:
            # We embed the CHILD (specific)
            # But we store the PARENT (context) in metadata
            doc = Document(
                page_content=child_text, # <--- Search searches this
                metadata={
                    "source": os.path.basename(file_path),
                    "chunk_id": f"P{i}_C{hash(child_text)}",
                    "parent_context": parent_text, # <--- LLM reads this
                    "is_parent_child": True
                }
            )
            final_docs.append(doc)

    print(f"🧩 Created {len(final_docs)} Parent-Child chunks.")
    return final_docs

def ingest(docs):
    if not docs: return
    print(f"📤 Ingesting {len(docs)} chunks...")
    PineconeVectorStore.from_documents(
        documents=docs, 
        embedding=embeddings, 
        index_name=INDEX_NAME
    )
    print("✅ Upload Success!")

if __name__ == "__main__":
    files = glob.glob("uploads/*.pdf")
    if not files:
        print("❌ No PDFs found in uploads/")
    
    for f in files:
        docs = process_file_smartly(f)
        ingest(docs)