import os
import glob
import re
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
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()

# --- CONFIG ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
INDEX_NAME = os.getenv("INDEX_NAME")

# Initialize Summarizer LLM
llm_summarizer = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.3)

def is_noise(text):
    """
    Detects if a line of text is likely 'OCR Garbage' or raw data tables.
    Returns True if it should be deleted.
    """
    if re.search(r'R\d+\s+G\d+', text): 
        return True
    
    num_count = sum(c.isdigit() for c in text)
    if len(text) > 0 and (num_count / len(text)) > 0.5:
        return True

    return False

def get_global_context(full_text_sample):
    """Generates a 2-sentence summary of the unit to ground the chunks."""
    print("🌍 Generating Global Context...")
    try:
        # We limit input to 10k chars to save tokens/time
        msg = HumanMessage(content=f"Summarize the following academic text in exactly 2 sentences. Focus on the main topic and key concepts:\n\n{full_text_sample[:10000]}") 
        res = llm_summarizer.invoke([msg])
        return res.content
    except Exception as e:
        print(f"⚠️ Could not generate summary: {e}")
        return "Academic course material."

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
            if not is_noise(raw_text):
                page_text_buffer[page_no] += f"{raw_text}\n"

    # 3. Generate Global Context (New Step)
    sorted_pages = sorted(page_text_buffer.keys())
    # Grab text from the first 5 pages to guess the context
    all_text_preview = " ".join([page_text_buffer[p] for p in sorted_pages[:5]])
    global_context = get_global_context(all_text_preview)
    print(f"📝 Context: {global_context}")

    # 4. Create Final Documents
    final_docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    
    for p_num in sorted_pages:
        full_page_text = page_text_buffer[p_num]
        images_on_page = page_image_buffer.get(p_num, [])
        
        base_metadata = {
            "source": os.path.basename(file_path),
            "page_number": p_num,
            "content_type": "course_material",
            "image_paths": images_on_page,
            "global_context": global_context # useful to see in debug
        }

        # Logic: We split the text first, then PREPEND context to every chunk
        # This ensures the context isn't lost if the text is long
        raw_chunks = []
        if len(full_page_text) > 1000:
            raw_chunks = text_splitter.split_text(full_page_text)
        else:
            raw_chunks = [full_page_text]
            
        for chunk in raw_chunks:
            # ENRICHMENT: Context + Content
            enriched_content = f"Global Context: {global_context}\n\nSlide Content: {chunk}"
            final_docs.append(Document(page_content=enriched_content, metadata=base_metadata))

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