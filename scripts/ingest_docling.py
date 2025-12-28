import os
import glob
import re  # <--- Added Regex module
from pathlib import Path
from dotenv import load_dotenv

# Docling Imports
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat, DocItemLabel

# LangChain Imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

load_dotenv()

# --- CONFIG ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
INDEX_NAME = os.getenv("INDEX_NAME")

def is_noise(text):
    """
    Detects if a line of text is likely 'OCR Garbage' or raw data tables.
    Returns True if it should be deleted.
    """
    # 1. Detect RGB Color Codes (e.g., "R202 G212 B75", "R198 G99 859")
    # Pattern: Look for "R" followed by numbers, then "G", etc.
    if re.search(r'R\d+\s+G\d+', text): 
        return True
        
    # 2. Detect Lines that are just random numbers/symbols (OCR noise)
    # If a line is short (< 50 chars) but has > 50% numbers, it's usually garbage
    num_count = sum(c.isdigit() for c in text)
    if len(text) > 0 and (num_count / len(text)) > 0.5:
        return True

    return False

def process_grouped_by_page(file_path):
    # 1. Setup Docling
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True 
    pipeline_options.do_table_structure = True
    pipeline_options.generate_picture_images = True
    
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    
    print(f"🧐 Analyzing layout for: {file_path}")
    result = converter.convert(file_path)
    
    # 2. Setup Image Storage
    unit_name = os.path.splitext(os.path.basename(file_path))[0]
    image_output_dir = Path(f"database/{unit_name}")
    image_output_dir.mkdir(parents=True, exist_ok=True)

    page_text_buffer = {}
    page_image_buffer = {}
    
    print("   ... Extracting Text & Images (with Noise Filtering)")
    
    # Iterate through every element
    for item, level in result.document.iterate_items():
        page_no = item.prov[0].page_no if item.prov else 1
        
        if page_no not in page_text_buffer:
            page_text_buffer[page_no] = ""
            page_image_buffer[page_no] = []

        # --- A. HANDLE IMAGES ---
        if item.label == DocItemLabel.PICTURE:
            if hasattr(item, "image") and item.image:
                img_count = len(page_image_buffer[page_no]) + 1
                img_filename = f"page_{page_no}_img_{img_count}.png"
                img_path = image_output_dir / img_filename
                item.image.pil_image.save(img_path, format="PNG")
                page_image_buffer[page_no].append(str(img_path))

        # --- B. HANDLE TEXT (With Cleaning) ---
        if hasattr(item, 'text') and item.text.strip():
            raw_text = item.text.strip()
            
            # CHECK FOR NOISE BEFORE ADDING
            if not is_noise(raw_text):
                page_text_buffer[page_no] += f"{raw_text}\n"
            else:
                # Optional: Print what we deleted to verify
                # print(f"      🗑️ Removed Noise: {raw_text[:30]}...") 
                pass

    # 4. Create Final Documents
    final_docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    
    sorted_pages = sorted(page_text_buffer.keys())
    
    for p_num in sorted_pages:
        full_page_text = page_text_buffer[p_num]
        images_on_page = page_image_buffer.get(p_num, [])
        
        base_metadata = {
            "source": os.path.basename(file_path),
            "page_number": p_num,
            "content_type": "course_material",
            "image_paths": images_on_page
        }

        if len(full_page_text) > 1000:
            page_chunks = text_splitter.split_text(full_page_text)
            for chunk in page_chunks:
                final_docs.append(Document(page_content=chunk, metadata=base_metadata))
        else:
            final_docs.append(Document(page_content=full_page_text, metadata=base_metadata))

    print(f"🧩 Processed {len(sorted_pages)} slides into {len(final_docs)} chunks.")
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
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
        
    files = glob.glob("uploads/*.pdf")
    if not files:
        print("❌ No PDFs found in uploads/")
    
    for f in files:
        docs = process_grouped_by_page(f)
        ingest(docs)