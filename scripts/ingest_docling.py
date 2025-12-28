import os
import glob
import time
import base64
import io
import logging
from PIL import Image
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

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# --- CONFIGURATION ---
INDEX_NAME = os.getenv("INDEX_NAME")
SUMMARIZER_MODEL = "llama-3.1-8b-instant" 
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Embedding Model: sentence-transformers/all-MiniLM-L6-v2
# Max Sequence Length: 256 tokens (~1000 chars)
# Vector Dimension: 384
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm_summarizer = ChatGroq(model_name=SUMMARIZER_MODEL, temperature=0.3)
vision_model = ChatGroq(model_name=VISION_MODEL, temperature=0.2)

def optimize_image(pil_image, max_dim=768, quality=85):
    """
    Optimizes image size and format (JPEG) to save tokens/cost.
    """
    if max(pil_image.size) > max_dim:
        pil_image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    
    if pil_image.mode in ('RGBA', 'P'):
        pil_image = pil_image.convert('RGB')
        
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def should_process_image(pil_image):
    """
    The 'Bouncer': Rejects small icons, lines, or footer logos.
    """
    w, h = pil_image.size
    if w < 100 or h < 100: return False 
    ratio = w / h
    if ratio > 5 or ratio < 0.2: return False 
    return True

def describe_image(pil_image, page_no):
    """
    Sequential Vision Processing (Simple & Reliable).
    """
    if not should_process_image(pil_image):
        return ""
        
    logger.info(f"[Vision] Analyzing image on Page {page_no}...")
    try:
        b64_img = optimize_image(pil_image)
        msg = HumanMessage(content=[
            {"type": "text", "text": "Describe this technical diagram/chart concisely. Focus on data labels and relationships."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
        ])
        res = vision_model.invoke([msg])
        return f"\n\n> **[IMAGE ANALYSIS]:** {res.content}\n"
    except Exception as e:
        logger.error(f"Vision failed on Page {page_no}: {e}")
        return ""

def get_global_context(text_sample):
    logger.info("Generating Global Summary...")
    try:
        msg = HumanMessage(content=f"Summarize main topic in 2 sentences:\n\n{text_sample[:4000]}")
        res = llm_summarizer.invoke([msg])
        return res.content
    except Exception as e:
        logger.error(f"Summary failed: {e}")
        return "Academic Course Material."

def process_file_smartly(file_path):
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True 
    pipeline_options.do_table_structure = True
    pipeline_options.generate_picture_images = True
    
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    
    logger.info(f"Processing: {file_path}")
    doc_result = converter.convert(file_path)
    
    # --- STEP 1: GATHER CONTENT PAGE-BY-PAGE ---
    page_content_map = {}
    full_text_preview = "" 

    for item, level in doc_result.document.iterate_items():
        page = item.prov[0].page_no if item.prov else 1
        
        if page not in page_content_map:
            page_content_map[page] = ""

        # Handle Content Types
        text_to_add = ""
        
        if item.label == DocItemLabel.PICTURE and hasattr(item, "image"):
            text_to_add = describe_image(item.image.pil_image, page)
            
        elif item.label == DocItemLabel.TABLE:
            try:
                df = item.export_to_dataframe()
                if not df.empty:
                    text_to_add = "\n" + df.to_markdown() + "\n"
            except:
                text_to_add = "\n" + item.text + "\n"
                
        elif item.label in [DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE]:
             text_to_add = f"\n## {item.text}\n"
             
        elif hasattr(item, "text") and item.text.strip():
             text_to_add = f"{item.text}\n"

        page_content_map[page] += text_to_add
        
        if len(full_text_preview) < 5000:
            full_text_preview += text_to_add

    # --- STEP 2: GLOBAL CONTEXT ---
    global_context = get_global_context(full_text_preview)
    
    # --- STEP 3: CONDITIONAL CHUNKING (OPTIMIZED) ---
    final_docs = []
    
    # OPTIMIZATION: Reduced from 1000 to 800 to fit all-MiniLM-L6-v2 context window
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    
    sorted_pages = sorted(page_content_map.keys())
    
    for p_num in sorted_pages:
        page_text = page_content_map[p_num].strip()
        if not page_text: continue
        
        # CONDITIONAL SPLIT LOGIC:
        # Check if the page fits in one chunk (with a small buffer)
        # If yes, 1 chunk. If no, split.
        if len(page_text) <= CHUNK_SIZE:
            chunks = [page_text]
        else:
            chunks = text_splitter.split_text(page_text)
            
        for i, chunk_text in enumerate(chunks):
            doc = Document(
                page_content=chunk_text,
                metadata={
                    "source": os.path.basename(file_path),
                    "page_number": p_num,
                    "chunk_index": i,
                    "global_context": global_context,
                    # Optimization: Store full text ONLY if split, saving DB space.
                    # RAG Engine will check this field first, then fallback to page_content.
                    "parent_context": page_text if len(chunks) > 1 else "" 
                }
            )
            final_docs.append(doc)

    return final_docs

def ingest(docs):
    if not docs: return
    logger.info(f"Ingesting {len(docs)} chunks...")
    PineconeVectorStore.from_documents(docs, embeddings, index_name=INDEX_NAME)
    logger.info("Ingestion Complete.")

if __name__ == "__main__":
    files = glob.glob("uploads/*.pdf")
    if not files:
        logger.warning("No PDFs found in uploads/")
    
    for f in files:
        docs = process_file_smartly(f)
        ingest(docs)