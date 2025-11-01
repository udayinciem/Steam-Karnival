# -*- coding: utf-8 -*-
"""
PDF OCR Extraction Module with Tesseract
Extracts text from PDFs using OCR for scanned documents and hybrid approach
"""

import os
import io
import fitz  # PyMuPDF
from PIL import Image
import numpy as np

# OCR libraries - optional imports
try:
    import pytesseract
    _TESSERACT_AVAILABLE = True
except ImportError:
    _TESSERACT_AVAILABLE = False

try:
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

from dotenv import load_dotenv

load_dotenv()


def check_tesseract_available():
    """Check if Tesseract OCR is available."""
    if not _TESSERACT_AVAILABLE:
        return False, "pytesseract not installed. Install with: pip install pytesseract pillow"
    
    try:
        # Try to get Tesseract version
        pytesseract.get_tesseract_version()
        return True, "Tesseract is available"
    except Exception as e:
        return False, f"Tesseract not found in PATH. Error: {e}"


def extract_text_from_pdf_simple(pdf_buffer):
    """
    Extract text from PDF using PyMuPDF (fallback for text-based PDFs).
    
    Args:
        pdf_buffer: BytesIO buffer containing PDF data
        
    Returns:
        str: Extracted text or None if error
    """
    text = []
    try:
        pdf_document = fitz.open(stream=pdf_buffer.getvalue(), filetype="pdf")
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            page_text = page.get_text()
            if page_text and page_text.strip():
                text.append(page_text)
        
        pdf_document.close()
        return "\n".join(text) if text else None
    except Exception as e:
        print(f"Error extracting text from PDF (simple method): {e}")
        return None


def pdf_to_images(pdf_buffer):
    """
    Convert PDF pages to PIL Images for OCR processing.
    
    Args:
        pdf_buffer: BytesIO buffer containing PDF data
        
    Returns:
        list: List of PIL Image objects
    """
    images = []
    try:
        pdf_document = fitz.open(stream=pdf_buffer.getvalue(), filetype="pdf")
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            
            # Convert page to image (pixmap)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
            
            # Convert to PIL Image
            img_data = pix.tobytes("ppm")
            img = Image.open(io.BytesIO(img_data))
            images.append(img)
        
        pdf_document.close()
        return images
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return []


def ocr_page(image, tesseract_config='--psm 6'):
    """
    Perform OCR on a single image using Tesseract.
    
    Args:
        image: PIL Image object
        tesseract_config: Tesseract configuration options
        
    Returns:
        str: OCR text or None if error
    """
    if not _TESSERACT_AVAILABLE:
        print("Tesseract not available")
        return None
    
    try:
        # Perform OCR with specified config
        text = pytesseract.image_to_string(image, config=tesseract_config)
        return text.strip() if text else None
    except Exception as e:
        print(f"Error during OCR: {e}")
        return None


def extract_text_from_pdf_ocr(pdf_buffer, use_ocr=True, hybrid_mode=True):
    """
    Extract text from PDF using OCR or hybrid approach.
    
    Args:
        pdf_buffer: BytesIO buffer containing PDF data
        use_ocr: If True, use OCR; if False, use simple extraction
        hybrid_mode: If True, try simple extraction first, use OCR for empty pages
        
    Returns:
        str: Extracted text
    """
    all_text = []
    
    try:
        pdf_document = fitz.open(stream=pdf_buffer.getvalue(), filetype="pdf")
        total_pages = pdf_document.page_count
        
        print(f"Processing PDF with {total_pages} pages...")
        
        for page_num in range(total_pages):
            page = pdf_document[page_num]
            
            # Try simple text extraction first
            page_text = page.get_text()
            
            # Determine if page has sufficient text
            is_text_page = page_text and len(page_text.strip()) > 100
            
            if hybrid_mode and is_text_page:
                # Text-based PDF - use simple extraction
                all_text.append(page_text)
                print(f"Page {page_num + 1}: Extracted {len(page_text)} chars (text-based)")
            elif use_ocr and _TESSERACT_AVAILABLE:
                # Scanned PDF or low text - use OCR
                print(f"Page {page_num + 1}: Using OCR...")
                
                # Convert page to image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_data = pix.tobytes("ppm")
                img = Image.open(io.BytesIO(img_data))
                
                # Perform OCR
                ocr_text = ocr_page(img)
                if ocr_text:
                    all_text.append(ocr_text)
                    print(f"Page {page_num + 1}: OCR extracted {len(ocr_text)} chars")
                else:
                    print(f"Page {page_num + 1}: OCR returned no text")
            elif hybrid_mode and is_text_page:
                # Fallback: use simple extraction even if short
                all_text.append(page_text)
                print(f"Page {page_num + 1}: Extracted {len(page_text)} chars (fallback)")
        
        pdf_document.close()
        
        result = "\n\n".join(all_text) if all_text else None
        if result:
            print(f"\nTotal extracted text: {len(result)} characters from {len(all_text)} pages")
        return result
        
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None


def extract_text_from_pdf(pdf_buffer):
    """
    Main extraction function - intelligently chooses between OCR and text extraction.
    
    Args:
        pdf_buffer: BytesIO buffer containing PDF data
        
    Returns:
        str: Extracted text or None if error
    """
    # Try smart hybrid extraction
    text = extract_text_from_pdf_ocr(pdf_buffer, use_ocr=True, hybrid_mode=True)
    
    if text and len(text.strip()) > 100:
        return text
    
    # Fallback to simple extraction
    print("Falling back to simple text extraction...")
    return extract_text_from_pdf_simple(pdf_buffer)


def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=150):
    """
    Split text into chunks with overlap for FAISS indexing.
    
    Args:
        text: Text to split
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        list: List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    
    return chunks


def create_faiss_from_text(text, output_path=None):
    """
    Create and save FAISS vector store from text chunks.
    
    Args:
        text: Text to create embeddings from
        output_path: Path to save FAISS index (default: my_pdf_vectors in script directory)
        
    Returns:
        int: Number of chunks created
    """
    if not _FAISS_AVAILABLE:
        raise Exception("FAISS dependencies not available. Install langchain-openai and langchain-community")
    
    # Set default output path if not provided
    if output_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(base_dir, "my_pdf_vectors")
    
    try:
        print(f"\nCreating FAISS embeddings...")
        print(f"Text length: {len(text):,} characters")
        
        # Split text into chunks
        chunks = split_text_into_chunks(text, chunk_size=1000, chunk_overlap=150)
        print(f"Split into {len(chunks)} chunks (size: 1000, overlap: 150)")
        
        # Create embeddings
        print("Generating embeddings with OpenAI...")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise Exception("OPENAI_API_KEY not found in environment variables")
        
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        
        # Create FAISS vectorstore
        print("Building FAISS vectorstore...")
        vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
        
        # Ensure directory exists
        os.makedirs(output_path, exist_ok=True)
        
        # Save vectorstore
        print(f"Saving FAISS index to: {output_path}")
        vectorstore.save_local(output_path)
        
        print(f"✓ Created FAISS index with {len(chunks)} chunks")
        print(f"✓ Saved to: {output_path}")
        return len(chunks)
        
    except Exception as e:
        print(f"✗ Error creating FAISS index: {e}")
        import traceback
        traceback.print_exc()
        raise


def process_pdf_file(pdf_path, output_dir="data/pdfs", create_faiss=True, faiss_output=None):
    """
    Complete workflow: Extract text from PDF and optionally create FAISS index.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save extracted text
        create_faiss: Whether to create FAISS index
        faiss_output: Path to save FAISS index (default: my_pdf_vectors in script directory)
        
    Returns:
        dict: Processing results
    """
    try:
        # Read PDF file
        with open(pdf_path, 'rb') as f:
            pdf_buffer = io.BytesIO(f.read())
        
        # Extract text
        print(f"\n{'='*60}")
        print(f"Processing: {pdf_path}")
        print(f"{'='*60}")
        
        extracted_text = extract_text_from_pdf(pdf_buffer)
        
        if not extracted_text:
            return {
                "status": "error",
                "message": "Failed to extract text from PDF"
            }
        
        # Save extracted text
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
        
        print(f"\nSaved extracted text to: {txt_path}")
        
        result = {
            "status": "success",
            "text_file": txt_path,
            "text_length": len(extracted_text),
            "chunks": 0
        }
        
        # Create FAISS index if requested
        if create_faiss and _FAISS_AVAILABLE:
            try:
                chunks_count = create_faiss_from_text(extracted_text, output_path=faiss_output)
                result["chunks"] = chunks_count
                result["faiss_path"] = faiss_output if faiss_output else os.path.join(os.path.dirname(os.path.abspath(__file__)), "my_pdf_vectors")
                print(f"\n✓ FAISS index created successfully")
            except Exception as e:
                print(f"\n⚠ Warning: Could not create FAISS index: {e}")
                result["faiss_error"] = str(e)
        
        return result
        
    except Exception as e:
        print(f"Error processing PDF file: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


def process_default_pdf():
    """Process the default PDF file: Manual State 2025 (2).pdf"""
    # Get the base directory (where this script is located)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Default PDF path
    default_pdf = os.path.join(base_dir, "data", "pdfs", "Manual State 2025 (2).pdf")
    
    # Default output directory
    output_dir = os.path.join(base_dir, "data", "pdfs")
    
    # Default FAISS output path
    faiss_output = os.path.join(base_dir, "my_pdf_vectors")
    
    print("\n" + "="*70)
    print("PDF OCR Text Extraction - Default Processing")
    print("="*70)
    print(f"PDF File: {default_pdf}")
    print(f"Output Directory: {output_dir}")
    print(f"FAISS Vectors: {faiss_output}")
    print("="*70 + "\n")
    
    # Check if file exists
    if not os.path.exists(default_pdf):
        print(f"ERROR: PDF file not found at: {default_pdf}")
        print("\nPlease ensure the file exists at:")
        print("  data/pdfs/Manual State 2025 (2).pdf")
        return None
    
    # Check Tesseract availability
    tesseract_available, tesseract_msg = check_tesseract_available()
    print(f"Tesseract OCR: {tesseract_msg}\n")
    
    faiss_available = "FAISS: Available" if _FAISS_AVAILABLE else "FAISS: Not available"
    print(f"{faiss_available}\n")
    
    # Process PDF with both OCR and PyMuPDF
    print("Starting PDF extraction with hybrid mode (OCR + PyMuPDF)...")
    print("-" * 70)
    
    result = process_pdf_file(default_pdf, output_dir=output_dir, create_faiss=True, faiss_output=faiss_output)
    
    # Print summary
    print("\n" + "="*70)
    print("Processing Summary")
    print("="*70)
    print(f"Status: {result.get('status', 'unknown')}")
    
    if result.get('status') == 'success':
        print(f"✓ Text extracted: {result.get('text_length', 0):,} characters")
        print(f"✓ Saved to: {result.get('text_file', 'N/A')}")
        
        if 'chunks' in result:
            print(f"✓ FAISS chunks created: {result.get('chunks', 0)}")
            print(f"✓ FAISS index saved to: {faiss_output}")
            print(f"  - index.faiss")
            print(f"  - index.pkl")
            print("\n✓ Semantic search is now ready for LLM responses!")
        
        if 'faiss_error' in result:
            print(f"⚠ FAISS warning: {result['faiss_error']}")
    else:
        print(f"✗ Error: {result.get('message', 'Unknown error')}")
    
    print("="*70 + "\n")
    return result


def main():
    """Command-line interface for PDF processing."""
    import sys
    
    # Get base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_pdf = os.path.join(base_dir, "data", "pdfs", "Manual State 2025 (2).pdf")
    
    print("\n" + "="*70)
    print("PDF OCR Text Extraction Tool")
    print("="*70 + "\n")
    
    # Check Tesseract availability
    tesseract_available, tesseract_msg = check_tesseract_available()
    print(f"Tesseract OCR: {tesseract_msg}")
    
    if not tesseract_available:
        print("\nNote: Install Tesseract for OCR support:")
        print("  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        print("  Linux: sudo apt-get install tesseract-ocr")
        print("  Mac: brew install tesseract\n")
    
    faiss_available = "FAISS: Available" if _FAISS_AVAILABLE else "FAISS: Not available"
    print(f"{faiss_available}\n")
    
    # Check command line arguments
    if len(sys.argv) < 2:
        # No arguments - process default PDF
        print(f"No arguments provided. Processing default PDF...\n")
        result = process_default_pdf()
        if result and result.get('status') == 'success':
            sys.exit(0)
        else:
            sys.exit(1)
    
    # Custom PDF path provided
    pdf_path = sys.argv[1]
    
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    create_faiss = "--no-faiss" not in sys.argv
    
    # Determine FAISS output path
    faiss_output = os.path.join(base_dir, "my_pdf_vectors")
    if "--faiss-output" in sys.argv:
        idx = sys.argv.index("--faiss-output")
        if idx + 1 < len(sys.argv):
            faiss_output = sys.argv[idx + 1]
    
    # Process PDF
    output_dir = os.path.dirname(pdf_path) if os.path.dirname(pdf_path) else os.path.join(base_dir, "data", "pdfs")
    result = process_pdf_file(pdf_path, output_dir=output_dir, create_faiss=create_faiss, faiss_output=faiss_output)
    
    # Print summary
    print("\n" + "="*70)
    print("Processing Summary")
    print("="*70)
    print(f"Status: {result.get('status', 'unknown')}")
    
    if result.get('status') == 'success':
        print(f"✓ Text extracted: {result.get('text_length', 0):,} characters")
        print(f"✓ Saved to: {result.get('text_file', 'N/A')}")
        
        if 'chunks' in result:
            print(f"✓ FAISS chunks: {result.get('chunks', 0)}")
            print(f"✓ FAISS index: {faiss_output}")
        
        if 'faiss_error' in result:
            print(f"⚠ FAISS error: {result['faiss_error']}")
    else:
        print(f"✗ Error: {result.get('message', 'Unknown error')}")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

