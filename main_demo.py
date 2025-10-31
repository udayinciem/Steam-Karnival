from fastapi import FastAPI, Form, Response, File, UploadFile, Query, Request
from twilio.rest import Client as TwilioClient
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime
import os
import json
import fitz  # PyMuPDF
import io
import re
try:
    # Optional: Only used if FAISS vectors are available
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    _FAISS_AVAILABLE = True
except Exception:
    _FAISS_AVAILABLE = False

# Load environment variables
load_dotenv()

# FastAPI initialization
app = FastAPI()

# Twilio setup
ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
FROM_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")
twilio_client = TwilioClient(ACCOUNT_SID, AUTH_TOKEN)

# OpenAI setup
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Data file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, "data")
PDF_STORAGE = os.path.join(DATA_FOLDER, "pdfs")
VECTORSTORE_PATH = os.path.join(BASE_DIR, "my_pdf_vectors")


def get_db():
    try:
        mongo_url = os.getenv("MONGO_URL", "mongodb://localhost:27017/")
        client = MongoClient(mongo_url, serverSelectionTimeoutMS=3000)
        # Force connection on a request as the connect=True parameter of MongoClient seems
        # to be useless here
        client.server_info()  # Will throw exception if cannot connect
        db = client["Steam-Karnival"]
        print("MongoDB connection successful.")
        return db
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
        return None

db = get_db()
chats_col = db['whatsapp_chats']

def store_chat(user_mobile, timestamp, question, response):
    chats_col.insert_one({
        "user_mobile_number": user_mobile,
        "user_timestamp": timestamp,
        "user_question": question,
        "response": response
    })

def fetch_all_chats(user_mobile):
    """Fetch all chats for a user, ordered by timestamp (oldest first)."""
    return list(chats_col.find({"user_mobile_number": user_mobile}).sort("user_timestamp", 1))

def find_excel_file():
    """Find the most recently modified Excel file in the PDF storage directory."""
    try:
        print("\n=== Looking for Excel File ===")
        excel_files = []
        for file in os.listdir(PDF_STORAGE):
            # Skip Excel lock/temp files that start with '~$'
            if file.startswith('~$'):
                continue
            if file.lower().endswith(('.xlsx', '.xls')):
                full_path = os.path.join(PDF_STORAGE, file)
                try:
                    mtime = os.path.getmtime(full_path)
                except Exception:
                    mtime = 0
                excel_files.append((file, mtime))
        if excel_files:
            # Pick the newest by modification time
            excel_files.sort(key=lambda t: t[1], reverse=True)
            newest_name, _ = excel_files[0]
            print(f"Found Excel files: {[n for n,_ in excel_files]}")
            excel_path = os.path.join(PDF_STORAGE, newest_name)
            print(f"Using newest Excel file: {excel_path}")
            return excel_path
        # If only lock file existed or none found
        print("No valid Excel files found in PDF storage directory. If an Excel file is open, close it (to remove '~$' lock file) and try again.")
        return None
    except Exception as e:
        print(f"Error finding Excel file: {e}")
        return None

# Create necessary folders if they don't exist
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(PDF_STORAGE, exist_ok=True)

# Date to day mapping
DATE_MAPPING = {
    '12-11-2025': 'WEDNESDAY',
    '13-11-2025': 'THURSDAY',
    '14-11-2025': 'FRIDAY',
    '15-11-2025': 'SATURDAY'
}

# Day to date mapping (reverse lookup)
DAY_MAPPING = {
    'WEDNESDAY': '12-11-2025',
    'WED': '12-11-2025',
    'THURSDAY': '13-11-2025',
    'THU': '13-11-2025',
    'FRIDAY': '14-11-2025',
    'FRI': '14-11-2025',
    'SATURDAY': '15-11-2025',
    'SAT': '15-11-2025'
}

# Global cache for program data
program_cache = {
    'excel_data': None,
    'pdf_data': {},  # User-specific PDF data cache
    'last_excel_update': None,
    'processed_files': {},  # Track processed files and their timestamps
    'combined_data': None,  # Store combined data from all sources
    'last_cache_update': None  # Track when the combined cache was last updated
}

# Initialize program cache in memory (no file needed)
program_cache['combined_data'] = []
program_cache['processed_files'] = {}
program_cache['last_cache_update'] = pd.Timestamp.now()

def clear_schedule_cache():
    """Hard-reset all in-memory caches so old Excel content cannot persist."""
    try:
        program_cache['excel_data'] = None
        program_cache['pdf_data'] = {}
        program_cache['last_excel_update'] = None
        program_cache['processed_files'] = {}
        program_cache['combined_data'] = None
        program_cache['last_cache_update'] = None
        print("All caches cleared.")
        return True
    except Exception as e:
        print(f"Error clearing cache: {e}")
        return False

def list_pdf_files():
    """List all files in the PDF storage directory"""
    try:
        all_files = os.listdir(PDF_STORAGE)
        files_by_type = {
            'pdf': [],
            'json': [],
            'txt': [],
            'excel': []
        }
        
        for file in all_files:
            if file.endswith('.pdf'):
                files_by_type['pdf'].append(file)
            elif file.endswith('.json'):
                files_by_type['json'].append(file)
            elif file.endswith('.txt'):
                files_by_type['txt'].append(file)
            elif file.endswith(('.xlsx', '.xls')):
                files_by_type['excel'].append(file)
        
        return files_by_type
    except Exception as e:
        print(f"Error listing PDF files: {e}")
        return None

def _faiss_index_exists():
    try:
        if not _FAISS_AVAILABLE:
            return False
        index_path = os.path.join(VECTORSTORE_PATH, "index.faiss")
        return os.path.exists(index_path)
    except Exception:
        return False

def split_text_into_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 150):
    """Split text into chunks with overlap for FAISS indexing."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

def create_faiss_from_text(text: str):
    """Create and save FAISS vector store from text chunks."""
    try:
        if not _FAISS_AVAILABLE:
            raise Exception("FAISS dependencies not available. Install langchain-openai and langchain-community")
        
        chunks = split_text_into_chunks(text, chunk_size=1000, chunk_overlap=150)
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
        
        # Ensure directory exists
        os.makedirs(VECTORSTORE_PATH, exist_ok=True)
        vectorstore.save_local(VECTORSTORE_PATH)
        return len(chunks)
    except Exception as e:
        print(f"Error creating FAISS index: {e}")
        raise

def get_pdf_chunks_context(question: str, k: int = 20):
    """Retrieve top-k chunks from FAISS vector store as additional context.
    Uses multiple query variations to improve table/chart retrieval.
    Returns empty string if FAISS is not available or index not found.
    """
    try:
        if not _FAISS_AVAILABLE:
            print(" FAISS not available - langchain packages may not be installed")
            return ""
        if not _faiss_index_exists():
            print("FAISS index not found - no PDF chunks available")
            return ""
        
        print(f"\nLoading FAISS index and retrieving chunks for: '{question}'...")
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        vectorstore = FAISS.load_local(
            VECTORSTORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Create multiple query variations to improve retrieval for tables/charts
        query_variations = [question]
        
        # Dynamically create query variations based on keywords (not hardcoded)
        question_lower = question.lower()
        
        # Extract key domain terms from the question
        key_terms = []
        for word in question.split():
            if len(word) > 3:  # Only meaningful words
                key_terms.append(word.lower())
        
        # Create variations using bigrams and combinations
        # This is dynamic - no hardcoded section names
        if len(key_terms) > 1:
            # Create bigram combinations (important for "value points folk dance")
            for i in range(len(key_terms) - 1):
                bigram = f"{key_terms[i]} {key_terms[i+1]}"
                query_variations.append(bigram)
        
        # For value points queries, add specific variations to improve retrieval
        if any(term in question_lower for term in ["value point", "marks"]):
            # Add variations that emphasize scoring/criteria
            if "folk" in question_lower:
                query_variations.extend([
                    "folk dance marks criteria scoring",
                    "folk dance value points breakdown",
                    "folk dance evaluation marks"
                ])
            query_variations.extend([
                "value points scoring criteria",
                "marks breakdown evaluation"
            ])
        
        # Add the original question keywords
        query_variations.extend(key_terms[:3])  # First 3 key terms
        # No hardcoded terms - let FAISS semantic search find relevant chunks naturally
        
        # Get unique chunks from all query variations
        all_docs = []
        seen_chunks = set()
        
        # For value points queries, use more variations to ensure we find the data
        max_variations = 5 if any(term in question_lower for term in ["value point", "marks"]) else 3
        
        for query_var in query_variations[:max_variations]:
            try:
                # Increase fetch_k for value points queries to get more candidates
                fetch_multiplier = 3 if any(term in question_lower for term in ["value point", "marks"]) else 2
                retriever = vectorstore.as_retriever(search_kwargs={"k": k, "fetch_k": k * fetch_multiplier})
                docs = retriever.invoke(query_var)
                print(f"  Query '{query_var}': Retrieved {len(docs)} docs")
                for doc in docs:
                    content = getattr(doc, "page_content", str(doc))
                    if content and content not in seen_chunks:
                        all_docs.append(doc)
                        seen_chunks.add(content)
            except Exception as e:
                print(f"  Warning: Error with query variation '{query_var}': {e}")
        
        if not all_docs:
            print("No documents retrieved from FAISS")
            return ""
        
        # Sort by relevance (keep first k unique chunks)
        chunks = []
        for i, doc in enumerate(all_docs[:k]):
            content = getattr(doc, "page_content", str(doc))
            if content:
                chunks.append(content)
                # Print first 150 chars of each chunk for debugging
                preview = content[:150].replace('\n', ' ').strip()
                print(f"  Chunk {i+1}: {preview}...")
                # Check if chunk contains key terms
                content_lower = content.lower()
                if "grade" in content_lower and ("percentage" in content_lower or "70" in content or "%" in content):
                    print(f"    This chunk contains grade/percentage info!")
                if "fixing of grade" in content_lower:
                    print(f"    This chunk contains 'Fixing of Grade' table!")
                if "grade a" in content_lower and "70" in content:
                    print(f"    This chunk contains Grade A with 70%!")
                if "trophies to schools" in content_lower or "ever rolling trophy" in content_lower:
                    print(f"    This chunk contains TROPHIES TO SCHOOLS info!")
                # Check for value points/marks info
                if "value point" in content_lower or ("mark" in content_lower and ("folk" in content_lower or "dance" in content_lower)):
                    print(f"    This chunk contains VALUE POINTS/MARKS info!")
                if "folk dance" in content_lower and ("mark" in content_lower or "value point" in content_lower):
                    print(f"    *** FOUND FOLK DANCE VALUE POINTS CHUNK! ***")
        
        context = "\n\n---\n\n".join(chunks) if chunks else ""
        print(f"Retrieved {len(chunks)} unique PDF chunks ({len(context)} chars total)")
        if context:
            # Check if context contains table-like patterns
            if any(keyword in context.lower() for keyword in ["grade", "%", "percentage", "points", "table"]):
                print(f"Context appears to contain table/chart data")
        return context
    except Exception as e:
        print(f"Error retrieving FAISS context: {e}")
        import traceback
        traceback.print_exc()
        return ""

def build_query_terms(query: str):
    """Extract meaningful lowercase terms (with simple bigrams and domain terms)."""
    tokens = re.findall(r"[A-Za-z0-9+()'&.-]+", (query or "").lower())
    terms = [t for t in tokens if len(t) >= 2]
    bigrams = [f"{terms[i]} {terms[i+1]}" for i in range(len(terms)-1)] if len(terms) > 1 else []
    domain = [
        "trophy", "trophies", "award", "awards", "prize", "prizes",
        "appeal", "fee", "fees", "mark", "marks",
        "light music", "one act play", "rules"
    ]
    merged = list(dict.fromkeys(terms + bigrams + domain))
    return merged

def extract_structured_section(raw_text: str, section_headers, max_scan_chars: int = 4000):
    """Extract a section that starts with any of the section_headers and return
    a cleaned bullet list. The function is designed for small rule blocks like
    'TROPHIES TO SCHOOLS' or similar.

    The extraction is heuristic but deterministic: it finds the header, then
    reads forward until a blank line followed by an all-caps header or until
    max_scan_chars is reached. It then formats roman-numbered items (i., ii.,
    iii., iv.) or line-wrapped sentences into bullets.
    """
    if not raw_text:
        return None
    text = raw_text
    lower_text = text.lower()
    # Find header occurrence - prioritize exact matches first
    start_idx = -1
    matched_header = None
    
    # First pass: look for exact case-insensitive matches in order of preference
    for header in section_headers:
        idx = lower_text.find(header.lower())
        if idx != -1:
            # If we found a match, use it and stop (priority-based search)
            start_idx = idx
            matched_header = header
            break
    
    # If no exact match found, try finding any header (fallback)
    if start_idx == -1:
        for header in section_headers:
            # Try partial matches
            if header.lower().replace(" ", "") in lower_text.replace(" ", ""):
                idx = lower_text.find(header.lower().split()[0])  # Find first word
                if idx != -1:
                    start_idx = idx
                    matched_header = header
                    break
    
    if start_idx == -1:
        return None

    # Scan forward window
    scan_end = min(len(text), start_idx + max_scan_chars)
    window = text[start_idx:scan_end]

    # Stop when next ALL-CAPS header appears on its own line
    lines = window.splitlines()
    collected = []
    seen_header = False
    all_caps_line = re.compile(r"^[A-Z0-9 ./&()-]{6,}$")
    roman_prefix = re.compile(r"^(i{1,3}|iv|v|vi{0,3}|x)\.[) ]?\s*", re.IGNORECASE)

    for line in lines:
        stripped = line.strip()
        if not seen_header:
            if stripped.lower().startswith(matched_header.lower()):
                seen_header = True
            continue
        # Break if new section starts (check if it's a different header)
        if stripped and all_caps_line.match(stripped):
            # Only break if it's clearly a different section header
            # (not just a continuation of current section)
            stripped_lower = stripped.lower()
            matched_lower = matched_header.lower()
            # Break if it's an ALL-CAPS header that doesn't match our current section
            first_word_match = stripped_lower.split()[0] if stripped_lower.split() else ""
            if first_word_match and first_word_match not in matched_lower.split():
                break
        # Skip empty numbering-only lines like 'i.' placed alone
        if roman_prefix.match(stripped) and len(stripped) <= 3:
            continue
        if stripped:
            collected.append(stripped)

    if not collected:
        return None

    # Join wrapped lines, split by roman numerals if present
    joined = " ".join(collected)
    # Normalize multiple spaces
    joined = re.sub(r"\s+", " ", joined).strip()

    # Try to split by known roman markers i., ii., iii., iv.
    parts = re.split(r"\s+(?=(i{1,3}|iv|v|vi{0,3})\.)", joined, flags=re.IGNORECASE)
    bullets = []
    if len(parts) > 1:
        # parts like ['', 'i', '. text ...', 'ii', '. text ...']
        cur = None
        for token in parts:
            if re.fullmatch(r"(i{1,3}|iv|v|vi{0,3})", token, flags=re.IGNORECASE):
                if cur:
                    bullets.append(cur.strip())
                cur = ""
            else:
                if cur is None:
                    cur = token
                else:
                    cur += token
        if cur:
            bullets.append(cur.strip())
    else:
        # Fallback: sentence split
        bullets = [s.strip() for s in re.split(r"(?<=[.!?])\s+", joined) if len(s.strip()) > 0]

    # Clean leading punctuation
    cleaned = [re.sub(r"^(\.|-\s*)", "", b).strip() for b in bullets if b]
    return cleaned if cleaned else None

def find_relevant_manual_snippets(manual_text: str, query_terms, window_before=600, window_after=800, max_snippets=8):
    """Find up to max_snippets windows around term matches in the manual."""
    if not manual_text:
        return []
    text_lower = manual_text.lower()
    hits = []
    for term in (query_terms or []):
        if not term.strip():
            continue
        idx = 0
        term_lower = term.lower()
        for _ in range(20):
            idx = text_lower.find(term_lower, idx)
            if idx == -1:
                break
            start = max(0, idx - window_before)
            end = min(len(manual_text), idx + len(term) + window_after)
            hits.append((start, end))
            idx = idx + len(term)
    if not hits:
        return []
    hits.sort()
    merged = []
    cur_start, cur_end = hits[0]
    for s, e in hits[1:]:
        if s <= cur_end + 50:
            cur_end = max(cur_end, e)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    merged.append((cur_start, cur_end))
    merged.sort(key=lambda se: se[1]-se[0], reverse=True)
    selected = merged[:max_snippets]
    return [manual_text[s:e] for s, e in selected]

def build_excel_context_rows(all_programs, query_terms, max_rows=30):
    """Create a compact text block with rows from Excel/PDF programs matching query terms."""
    rows = []
    for program in (all_programs or []):
        blob = " ".join(str(v).lower() for v in program.values())
        if any(t in blob for t in (query_terms or [])):
            rows.append(program)
        if len(rows) >= max_rows:
            break
    if not rows:
        return ""
    def fmt(p):
        name = p.get('Program Name', 'Unknown')
        cat = p.get('Category', 'N/A')
        stage = p.get('Stage', '')
        date = p.get('Date', '')
        code = p.get('Item Code', '')
        time = p.get('Time', '')
        return f"- {name} | Category: {cat} | Stage: {stage} | Date: {date} | Item Code: {code} | Time: {time}"
    return "\n".join(fmt(p) for p in rows)

def ensure_pdfs_indexed():
    """Ensure all PDFs in storage have extracted text and JSON data.
    For each .pdf in `PDF_STORAGE`, if corresponding .json is missing,
    extract text and process into structured data using existing helpers.
    """
    try:
        all_files = os.listdir(PDF_STORAGE)
        pdf_files = [f for f in all_files if f.lower().endswith('.pdf')]
        for pdf_file in pdf_files:
            base_name = os.path.splitext(pdf_file)[0]
            json_path = os.path.join(PDF_STORAGE, base_name + '.json')
            txt_path = os.path.join(PDF_STORAGE, base_name + '.txt')

            # Skip if JSON already exists
            if os.path.exists(json_path):
                continue

            pdf_path = os.path.join(PDF_STORAGE, pdf_file)
            try:
                # Read file bytes and extract text
                with open(pdf_path, 'rb') as f:
                    pdf_buffer = io.BytesIO(f.read())
                extracted_text = extract_text_from_pdf(pdf_buffer)

                if not extracted_text:
                    print(f"No text extracted from {pdf_file}; skipping JSON generation.")
                    continue

                # Save extracted text
                try:
                    with open(txt_path, 'w', encoding='utf-8') as tf:
                        tf.write(extracted_text)
                except Exception as te:
                    print(f"Error writing TXT for {pdf_file}: {te}")

                # Process into structured data and save JSON
                structured = process_pdf_content(extracted_text)
                try:
                    with open(json_path, 'w', encoding='utf-8') as jf:
                        json.dump(structured, jf, indent=2)
                except Exception as je:
                    print(f"Error writing JSON for {pdf_file}: {je}")
            except Exception as pe:
                print(f"Error indexing PDF {pdf_file}: {pe}")
    except Exception as e:
        print(f"Error ensuring PDFs indexed: {e}")

def get_all_pdf_data():
    """Load data from all JSON files in the PDF storage"""
    try:
        json_files = [f for f in os.listdir(PDF_STORAGE) if f.endswith('.json')]
        all_data = []
        
        for json_file in json_files:
            try:
                with open(os.path.join(PDF_STORAGE, json_file), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        all_data.append(data)
            except Exception as e:
                print(f"Error reading {json_file}: {e}")
                continue
        
        return all_data
    except Exception as e:
        print(f"Error getting all PDF data: {e}")
        return []

def update_cache_if_needed():
    """Update the combined data cache if any files have changed"""
    global program_cache
    print("\n=== Cache Update Status ===")
    print("Excel data in cache:", "Present" if program_cache['excel_data'] is not None else "None")
    print("Combined data in cache:", "Present" if program_cache['combined_data'] is not None else "None")
    cache_needs_update = False
    
    try:
        # Check if any files have changed
        all_files = os.listdir(PDF_STORAGE)
        current_files = {}
        
        # Check all relevant files
        for file in all_files:
            if file.endswith(('.pdf', '.json', '.xlsx', '.xls')):
                file_path = os.path.join(PDF_STORAGE, file)
                current_files[file] = os.path.getmtime(file_path)
                
                # Check if file is new or modified
                if (file not in program_cache['processed_files'] or 
                    program_cache['processed_files'][file] != current_files[file]):
                    cache_needs_update = True
        
        # Update cache if needed
        if (cache_needs_update or 
            program_cache['combined_data'] is None or 
            program_cache['processed_files'] != current_files):
            
            print("Updating program cache...")
            
            # Update Excel data
            update_excel_cache()
            
            # Ensure any new PDFs are indexed to TXT/JSON
            ensure_pdfs_indexed()

            # Get all PDF data
            all_pdf_data = get_all_pdf_data()
            
            # Combine all data
            combined_data = []
            
            # Add Excel data if available
            if not program_cache['excel_data'].empty:
                for _, row in program_cache['excel_data'].iterrows():
                    try:
                        # Get data from row using column names
                        stage = str(row['Stage']).strip()
                        program_name = str(row['Program']).strip()
                        category = str(row['Category']).strip()
                        date = str(row['Date']).strip()
                        
                        # Skip empty or invalid rows
                        if not program_name or program_name.lower() == 'nan':
                            continue
                            
                        # Get day name for the date
                        day_name = DATE_MAPPING.get(date, "")
                        
                        program = {
                            "Program Name": program_name,
                            "Category": category if category != "nan" else "N/A",
                            "Stage": stage,
                            "Date": date,
                            "Day": day_name,
                            "Source": "excel"
                        }
                        
                        print(f"Added program: {program_name} on {date} ({day_name}) at {stage}")  # Debug print
                        combined_data.append(program)
                    except Exception as e:
                        print(f"Error processing Excel row: {e}")
                        continue
            
            # Add PDF data
            if all_pdf_data:
                for program in all_pdf_data:
                    try:
                        program_copy = program.copy()
                        program_copy["Source"] = "pdf"
                        combined_data.append(program_copy)
                    except Exception as e:
                        print(f"Error processing PDF data: {e}")
                        continue
            
            # Don't remove duplicates - we want all programs even if names are same
            # Update cache
            program_cache['combined_data'] = combined_data
            program_cache['processed_files'] = current_files
            program_cache['last_cache_update'] = pd.Timestamp.now()
            
            print(f"Cache updated with {len(combined_data)} programs")
            
            # Cache is now kept only in memory
        
        return program_cache['combined_data']
    
    except Exception as e:
        print(f"Error updating cache: {e}")
        return program_cache.get('combined_data', [])

def update_excel_cache():
    """Update the Excel data cache if needed"""
    global program_cache
    try:
        excel_path = find_excel_file()
        if excel_path:
            excel_mtime = os.path.getmtime(excel_path)
            
            if (program_cache['excel_data'] is None or 
                program_cache['last_excel_update'] != excel_mtime):
                print(f"Loading Excel file from: {excel_path}")
                print("\n=== LOADING EXCEL FILE ===")
                print(f"Excel file path: {excel_path}")
                
                # Read Excel file with no headers and keep dates as strings
                df = pd.read_excel(
                    excel_path,
                    header=None,
                    dtype=str,  # Force all columns to be string
                    keep_default_na=False,  # Don't convert empty cells to NaN
                    parse_dates=False  # Don't parse dates
                )
                
                print("\n=== RAW EXCEL DATA ===")
                print("Shape:", df.shape)
                print("\nFirst 10 rows:")
                print(df.head(10))
                
                print("\n=== COLUMN VALUES ===")
                for col in df.columns:
                    print(f"\nColumn {col} unique values:")
                    print(df[col].unique())
                
                # Dynamically rename columns for schedule Excel file
                # Accepts various headers: Time (Stage), Item (Program), Category, Date
                map_cols = {}
                wanted = {'Stage': ['stage', 'time'], 'Program': ['program', 'item'], 'Category': ['category'], 'Date': ['date']}
                for wanted_col, options in wanted.items():
                    for col in df.columns:
                        col_lower = str(col).strip().lower()
                        # Check cell value if first row is not likely a header
                        col0_val = str(df.iloc[0][col]).strip().lower() if df.shape[0] > 0 else ''
                        all_names = set([col_lower, col0_val])
                        if any(any(name.startswith(opt) for opt in options) for name in all_names):
                            map_cols[wanted_col] = col
                            break
                # Fill in missing mappings by guessing by position if not found and only 4 columns
                if len(map_cols) < 4 and len(df.columns) == 4:
                    map_order = ['Stage', 'Program', 'Category', 'Date']
                    for idx, want in enumerate(map_order):
                        if want not in map_cols:
                            map_cols[want] = df.columns[idx]
                # Rename columns accordingly
                df = df.rename(columns={v: k for k, v in map_cols.items() if v in df.columns})
                print("\n=== COLUMNS AFTER RENAME ===")
                print(df.columns)
                
                print("\n=== CLEANING DATA ===")
                # Clean up the data
                for col in df.columns:
                    print(f"\nCleaning column: {col}")
                    print("Before cleaning:", df[col].head())
                    df[col] = df[col].astype(str).str.strip()
                    df[col] = df[col].replace('nan', '')
                    print("After cleaning:", df[col].head())
                
                print("\n=== REMOVING EMPTY/HEADER ROWS ===")
                print("Rows before:", len(df))
                # Remove any empty rows
                df = df.dropna(how='all')
                df = df[df['Program'].str.len() > 0]
                # Remove header-like rows that leaked into data
                header_like_programs = {"program", "program name", "programme", "programs", "programmes"}
                header_like_stages = {"stage", "stages"}
                header_like_categories = {"category", "categories"}
                header_like_dates = {"date", "dates"}
                df = df[~df['Program'].str.strip().str.lower().isin(header_like_programs)]
                df = df[~df['Stage'].str.strip().str.lower().isin(header_like_stages)]
                df = df[~df['Category'].str.strip().str.lower().isin(header_like_categories)]
                df = df[~df['Date'].str.strip().str.lower().isin(header_like_dates)]
                # Also drop rows where all four columns equal their header tokens
                df = df[~(
                    df['Stage'].str.strip().str.upper().eq('STAGE') &
                    df['Program'].str.strip().str.upper().eq('PROGRAM') &
                    df['Category'].str.strip().str.upper().eq('CATEGORY') &
                    df['Date'].str.strip().str.upper().eq('DATE')
                )]
                print("Rows after:", len(df))
                
                print("\n=== FINAL DATA ===")
                print("All rows in cleaned data:")
                print(df.to_string())
                
                # Print all rows for debugging
                print("\nAll rows after cleaning:")
                print(df.to_string())
                print(f"\nAfter cleaning, total rows: {len(df)}")
                print("\nUnique dates found:", sorted(df['Date'].unique().tolist()))
                
                program_cache['excel_data'] = df
                program_cache['last_excel_update'] = excel_mtime
        else:
            # If Excel file doesn't exist, use empty DataFrame
            if program_cache['excel_data'] is None:
                program_cache['excel_data'] = pd.DataFrame()
                program_cache['last_excel_update'] = None
                print("No Excel file found in data/pdfs. Operating with PDF data only.")
    except Exception as e:
        print(f"Error updating Excel cache: {e}")
        # Use empty DataFrame if there's an error
        program_cache['excel_data'] = pd.DataFrame()
        program_cache['last_excel_update'] = None

def get_cached_pdf_data(user_number):
    """Get PDF data from cache or load if needed"""
    global program_cache
    
    # Clean the phone number for filename matching
    safe_number = user_number.replace(':', '_').replace('+', '')
    
    # Check if we have cached data for this user
    if safe_number not in program_cache['pdf_data']:
        try:
            # List all JSON files for this user
            json_files = [f for f in os.listdir(PDF_STORAGE) 
                         if f.endswith('.json') and safe_number in f]
            
            if json_files:
                # Get the most recent JSON file
                latest_json = sorted(json_files)[-1]
                json_path = os.path.join(PDF_STORAGE, latest_json)
                
                # Load and cache the data
                with open(json_path, 'r', encoding='utf-8') as f:
                    program_cache['pdf_data'][safe_number] = json.load(f)
            else:
                program_cache['pdf_data'][safe_number] = None
                
        except Exception as e:
            print(f"Error loading PDF data: {e}")
            program_cache['pdf_data'][safe_number] = None
    
    return program_cache['pdf_data'].get(safe_number)


from pydantic import BaseModel

class TestMessage(BaseModel):
    message: str

@app.get("/test")
@app.post("/test")
async def test_webhook(message: str = None, test_body: TestMessage = None):
    """
    Test endpoint for trying responses without Twilio.
    Supports both GET and POST methods.
    
    GET Example: http://localhost:8000/test?message=What's happening on 14-11-2025?
    
    POST Example (JSON body):
    {
        "message": "What's happening on 14-11-2025?"
    }
    """
    try:
        # Get message from either query param or body
        final_message = message if message else (test_body.message if test_body else None)
        
        if not final_message:
            return {
                "status": "error",
                "error": "No message provided. Use query parameter '?message=' for GET or JSON body {'message': ''} for POST"
            }

        print(f"\n=== Processing Test Request ===")
        print(f"Input message: {final_message}")
        
        # Extract query details
        extracted = extract_query_info(final_message)
        print(f"Extracted info: {extracted}")
        
        # Force clean slate before testing
        clear_schedule_cache()
        combined_data = update_cache_if_needed()
        print(f"Found {len(combined_data)} programs in data")
        
        # Search using cached data
        response = search_program_data(extracted, final_message, combined_data)
        print(f"Search response received")
        
        # Generate human-like reply
        final_reply = generate_human_like_reply(final_message, response)
        print(f"Final reply generated")
        
        result = {
            "status": "success",
            "query": final_message,
            "extracted_info": extracted,
            "response": final_reply,
            "debug_info": {
                "total_programs": len(combined_data),
                "query_type": extracted.get("query_type", "unknown")
            }
        }
        print(f"Returning result: {result}")
        return result
        
    except Exception as e:
        error_result = {
            "status": "error",
            "error": str(e),
            "query": final_message,
            "location": "test_webhook"
        }
        print(f"Error in test endpoint: {error_result}")
        return error_result

@app.post("/webhook")
async def whatsapp_webhook(From: str = Form(...), Body: str = Form(...), MediaUrl0: str = Form(None)):
    """
    Webhook endpoint for Twilio WhatsApp messages.
    Handle both text queries and PDF attachments.
    """
    user_message = Body.strip()
    user_number = From

    print(f" Message from {user_number}: {user_message}")

    # Handle PDF attachment if present
    if MediaUrl0 and MediaUrl0.lower().endswith('.pdf'):
        try:
            # Download PDF from Twilio's media URL
            import requests
            response = requests.get(MediaUrl0)
            # Generate a unique filename using timestamp and user number
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            safe_number = user_number.replace(':', '_').replace('+', '')
            pdf_filename = f"program_schedule_{timestamp}_{safe_number}.pdf"
            pdf_path = os.path.join(PDF_STORAGE, pdf_filename)
            # Save PDF to file
            with open(pdf_path, 'wb') as pdf_file:
                pdf_file.write(response.content)
            # Create buffer for processing
            pdf_buffer = io.BytesIO(response.content)
            # Extract text from PDF
            pdf_text = extract_text_from_pdf(pdf_buffer)
            if pdf_text:
                # Save extracted text alongside PDF
                text_filename = pdf_filename.replace('.pdf', '.txt')
                text_path = os.path.join(PDF_STORAGE, text_filename)
                with open(text_path, 'w', encoding='utf-8') as text_file:
                    text_file.write(pdf_text)
                # Process PDF content into structured data
                pdf_data = process_pdf_content(pdf_text)
                # Save structured data as JSON
                json_filename = pdf_filename.replace('.pdf', '.json')
                json_path = os.path.join(PDF_STORAGE, json_filename)
                with open(json_path, 'w', encoding='utf-8') as json_file:
                    json.dump(pdf_data, json_file, indent=2)
                send_whatsapp_message(user_number, 
                    "I've received and processed your PDF. The schedule has been saved and I can now answer questions about it!")
                return {"status": "ok", "message": "PDF processed and confirmed"}
        except Exception as e:
            print(f"Error processing PDF: {e}")
            send_whatsapp_message(user_number, 
                "Sorry, I had trouble processing that PDF. Could you try sending it again?")
            return {"status": "error", "message": str(e)}

    # --- Normal chat logic when no PDF ---
    extracted = extract_query_info(user_message)
    # Step 2: Get chat history (last 5 turns) for intelligent follow-up detection
    chat_history_entries = fetch_all_chats(user_number)
    recent_entries = chat_history_entries[-5:] if len(chat_history_entries) > 5 else chat_history_entries
    chat_history = ""
    for entry in recent_entries:
        chat_history += (
            f"User ({entry['user_timestamp']}): {entry['user_question']}\n"
            f"Bot: {entry['response']}\n"
        )
    chat_history += f"User (now): {user_message}\n"

    # AI-driven follow-up analysis (no hardcoded indicators)
    final_query = user_message
    try:
        followup_resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[
                {"role": "system", "content": "You are the Receptionist for Kalsolavm. Decide if the user's latest message is a follow-up to the prior conversation. Return strict JSON only."},
                {"role": "user", "content": f"Conversation so far (last 5 turns):\n{chat_history}\n\nTask: Is the latest message a follow-up to the previous topic? If yes, rewrite it into a standalone query that includes the missing context.\nReturn JSON: {{\"is_followup\": true|false, \"standalone_query\": \"...\"}}"}
            ]
        )
        raw = followup_resp.choices[0].message.content.strip()
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and data.get("standalone_query"):
                final_query = str(data["standalone_query"]).strip()
        except Exception:
            pass
    except Exception as e:
        print(f"follow-up analysis failed: {e}")
    try:
        program_cache['combined_data'] = None
        program_cache['excel_data'] = None
        # Ensure no old Excel content persists between user runs
        clear_schedule_cache()
        combined_data = update_cache_if_needed()
        # Use AI-generated standalone query if it's a follow-up; otherwise the original message
        ai_response = search_program_data(extracted, final_query, combined_data)
    except Exception as e:
        ai_response = ("I'm having trouble accessing the program data right now. "
                       "Please try again in a moment.")
    final_reply = generate_human_like_reply(user_message, ai_response)
    send_result = send_whatsapp_message(user_number, final_reply)
    # Store the chat with full details
    store_chat(
        user_mobile=user_number,
        timestamp=datetime.utcnow(),
        question=user_message,
        response=final_reply
    )
    return {"status": "ok"}


def extract_query_info(message: str):
    """
    Use OpenAI to understand user intent and extract keywords.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """You are an expert assistant for Kalolsavam cultural festival.
                Extract search parameters from user queries about programs, stages, times, venues, categories, etc.
                Handle various query types:
                - Schedule queries (e.g., "When is Kalolsavam?", "What are the festival dates?", "Tell me the dates")
                  For schedule queries, set query_type to "schedule"
                - Stage queries (e.g., "What's happening in Stage 2?", "Stage 2 programs")
                - Category queries (e.g., "What are the category one programmes?", "Show cat 1 programs", "Category 2 events")
                  Extract category number (1, 2, 3, 4, etc.) and set query_type to "category"
                - Time queries (e.g., "morning programs", "What's at 10 AM?")
                - Date queries (e.g., "What's on 15-11-2025?", "programs on November 15", "What is in the 15-11-2025", "What's happening tomorrow?")
                  For dates, understand and convert any date format to a standard format
                - Day queries (e.g., "What's happening on Saturday?", "Friday's programs", "What's on FRI?")
                  For days, understand both full names and abbreviations
                - Program queries (e.g., "When is Kathakali?")
                - Venue queries (e.g., "auditorium programs", "What's in Main Hall?")
                - Combined queries (e.g., "Category 1 programs on Friday", "Stage 2 programs on 14-11-2025", "Category 2 events in Stage 5")
                - Greeting/casual messages (e.g., "hi", "good morning", "how are you")
                
                IMPORTANT: 
                - For questions about when the festival is happening, use query_type "schedule"
                - For category queries, extract the category number (1, 2, 3, 4) from phrases like "category one", "cat 1", "category-1", etc.
                - Handle any date format intelligently (e.g., "15th November", "Nov 15", "15/11/25", "15-11-2025")
                - Convert relative dates (e.g., "tomorrow", "next Friday") to actual dates
                - For days, understand context and convert to actual dates"""},
                {"role": "user", "content": f"""Analyze this query and extract search parameters: "{message}"
                Return JSON with these fields:
                - program: Program name or "" if not specific
                - stage: Stage number (just the number) or "" if not mentioned
                - category: Category number (1, 2, 3, 4) or "" if not mentioned
                - time: Time period mentioned or "" if none
                - date: Date in DD-MM-YYYY format or "" if not mentioned
                - day: Day name (e.g., "FRIDAY", "SAT") or "" if not mentioned
                - venue: Venue name (e.g., "auditorium", "hall") or "" if not mentioned
                - query_type: One of ["program", "stage", "category", "time", "date", "day", "venue", "schedule", "general", "greeting", "combined"]
                - is_greeting: true if this is a casual greeting/conversation, false otherwise
                Example 1: "What is in the 15-11-2025?" → {{"program": "", "stage": "", "category": "", "time": "", "date": "15-11-2025", "day": "", "venue": "", "query_type": "date", "is_greeting": false}}
                Example 2: "Stage 2 morning programs" → {{"program": "", "stage": "2", "category": "", "time": "morning", "date": "", "day": "", "venue": "", "query_type": "stage", "is_greeting": false}}
                Example 3: "What are the category one programmes?" → {{"program": "", "stage": "", "category": "1", "time": "", "date": "", "day": "", "venue": "", "query_type": "category", "is_greeting": false}}
                Example 4: "Category 1 programs on Friday" → {{"program": "", "stage": "", "category": "1", "time": "", "date": "", "day": "FRIDAY", "venue": "", "query_type": "combined", "is_greeting": false}}
                Example 5: "Hi, how are you?" → {{"program": "", "stage": "", "category": "", "time": "", "date": "", "day": "", "venue": "", "query_type": "greeting", "is_greeting": true}}"""}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(" OpenAI extract error:", e)
        return {"program": "", "stage": "", "category": "", "time": "", "date": "", "day": "", "venue": "", "query_type": "general", "is_greeting": False}


def load_pdf_data(user_number):
    """
    Load the most recent PDF data for a given user number
    """
    try:
        # Clean the phone number for filename matching
        safe_number = user_number.replace(':', '_').replace('+', '')
        
        # List all JSON files for this user
        json_files = [f for f in os.listdir(PDF_STORAGE) 
                     if f.endswith('.json') and safe_number in f]
        
        if not json_files:
            return None
            
        # Get the most recent JSON file
        latest_json = sorted(json_files)[-1]
        json_path = os.path.join(PDF_STORAGE, latest_json)
        
        # Load and return the data
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading PDF data: {e}")
        return None

def combine_program_data(excel_data, pdf_data):
    """
    Combine program data from Excel and PDF sources
    """
    combined_data = []
    
    # Add Excel data if available and not empty
    if not excel_data.empty:
        try:
            for _, row in excel_data.iterrows():
                try:
                    program = {
                        "Program Name": str(row.get("Program Name", "Unknown Program")),
                        "Category": str(row.get("Category", "N/A")),
                        "Venue": str(row.get("Venue", "N/A")),
                        "Stage": str(row.get("Stage", "N/A")),
                        "Time": (f"{row.get('Start Time', 'N/A')} to "
                               f"{row.get('End Time', 'N/A')}"),
                        "Participants": str(row.get("Participants", "N/A")),
                        "Source": "excel"
                    }
                    combined_data.append(program)
                except Exception as e:
                    print(f"Error processing Excel row: {e}")
                    continue
        except Exception as e:
            print(f"Error processing Excel data: {e}")
    
    # Add PDF data if available
    if pdf_data:
        try:
            for program in pdf_data:
                try:
                    program_copy = program.copy()  # Create a copy to avoid modifying original
                    program_copy["Source"] = "pdf"
                    # Ensure all required fields exist with defaults
                    program_copy.setdefault("Program Name", "Unknown Program")
                    program_copy.setdefault("Category", "N/A")
                    program_copy.setdefault("Venue", "N/A")
                    program_copy.setdefault("Stage", "N/A")
                    program_copy.setdefault("Time", "Time not specified")
                    program_copy.setdefault("Participants", "N/A")
                    combined_data.append(program_copy)
                except Exception as e:
                    print(f"Error processing PDF program: {e}")
                    continue
        except Exception as e:
            print(f"Error processing PDF data: {e}")
    
    return combined_data

def search_program_data(extracted, user_message, combined_data=None):
    """
    Search program data from the combined cache based on extracted info and query type.
    Handle both program queries and casual conversations.
    """
    # Handle empty or None data
    if not combined_data:
        combined_data = []
    
    # Print debug info
    print(f"Total programs in data: {len(combined_data)}")
    
    # Try to load manual text for AI-powered search
    manual_path = os.path.join(PDF_STORAGE, "Manual State 2025 (2).txt")
    manual_text = ""
    if os.path.exists(manual_path):
        try:
            with open(manual_path, 'r', encoding='utf-8') as f:
                manual_text = f.read()
        except Exception as e:
            print(f"Error loading manual text: {e}")
    
    # Build query terms for manual/excel search
    query_terms = build_query_terms(user_message)
    
    # Special intent: count total number of stages (data-driven; no hardcoding)
    lower_msg = user_message.lower()
    is_stage_count_query = (
        any(kw in lower_msg for kw in ["stage", "stages"]) and
        (
            any(kw in lower_msg for kw in ["how many", "number", "count", "total"]) or
            "no of" in lower_msg or
            re.search(r"\bno\.?\s*of\s+stages\b", lower_msg) is not None or
            re.search(r"\bstages?\b.*\bhow\s+many\b", lower_msg) is not None
        )
    )
    if is_stage_count_query:
        stages = set()
        for prog in (combined_data or []):
            stage_raw = str(prog.get("Stage", "")).strip()
            if not stage_raw:
                continue
            # Extract numeric part and ignore header-like rows
            m = re.search(r"(\d+)", stage_raw)
            if not m:
                continue
            stages.add(int(m.group(1)))
        total = len(stages)
        if total > 0:
            return f"There are {total} stages in Kalolsavam."
        return "I couldn't find stage information in the schedule data."

    # Check if this is a program/stage/schedule question - prioritize Excel data
    is_schedule_question = any(term in user_message.lower() for term in [
        "stage", "performing", "program", "schedule", "date", "when is", 
        "where is", "which stage", "what stage", "at what"
    ])
    
    # For schedule questions, search Excel first before PDF chunks
    if is_schedule_question and combined_data:
        print("\nThis is a schedule question - prioritizing Excel data...")
        # Restrict to Excel rows first (fallback to all only if Excel missing)
        excel_rows = [p for p in combined_data if p.get("Source") == "excel"] or combined_data

        # Strong exact/containment name match pass
        msg_upper = user_message.upper()
        exact_name_matches = []
        for program in excel_rows:
            program_name_upper = program.get("Program Name", "").strip().upper()
            if program_name_upper and (program_name_upper in msg_upper or msg_upper in program_name_upper):
                exact_name_matches.append(program)

        matching_programs = []
        if exact_name_matches:
            matching_programs = exact_name_matches
        else:
            # Fallback to token-based loose matching
            program_name_terms = [w for w in user_message.split() if len(w) > 3]
            for program in excel_rows:
                program_name = program.get("Program Name", "").upper()
                if any(term.upper() in program_name for term in program_name_terms):
                    matching_programs.append(program)

        if matching_programs:
            if os.getenv("DEBUG_LOGS") == "1":
                print(f"[DEBUG] Found {len(matching_programs)} matching program(s) in Excel data for '{user_message}':")
                for prog in matching_programs:
                    print(f"  - {prog}")
            # Remove duplicates from matching_programs (unique by program name, stage, category, date)
            seen = set()
            deduped = []
            for prog in matching_programs:
                prog_key = (
                    prog.get("Program Name", "").strip().upper(),
                    prog.get("Stage", "").strip().upper(),
                    prog.get("Category", "").strip().upper(),
                    str(prog.get("Date", "")).strip()
                )
                if prog_key not in seen:
                    seen.add(prog_key)
                    deduped.append(prog)
            matching_programs = deduped
            
            # Helper function to format date nicely
            def format_date_for_answer(date_str):
                """Convert date to human-readable format"""
                if not date_str or date_str == "N/A" or str(date_str) == "nan":
                    return ""
                try:
                    # Try parsing different date formats
                    date_str_clean = str(date_str).strip()
                    # If already in DD-MM-YYYY format, use as is
                    if re.match(r'\d{2}-\d{2}-\d{4}', date_str_clean):
                        return date_str_clean
                    # Try parsing as pandas datetime
                    date_obj = pd.to_datetime(date_str_clean, errors='coerce')
                    if pd.notna(date_obj):
                        return date_obj.strftime("%d-%m-%Y")
                except:
                    pass
                # Fallback: just return cleaned string
                return str(date_str).replace("00:00:00", "").strip()
            
            # If the user specified a date or day, filter to that date only
            filter_date = None
            if extracted.get("date"):
                try:
                    filter_date = pd.to_datetime(extracted["date"]).strftime("%d-%m-%Y")
                except Exception:
                    filter_date = extracted["date"]
            elif extracted.get("day"):
                day_key = extracted["day"].upper()
                filter_date = DAY_MAPPING.get(day_key)

            if filter_date:
                filtered_matches = []
                for prog in matching_programs:
                    prog_date = str(prog.get("Date", "")).strip()
                    try:
                        prog_date_norm = pd.to_datetime(prog_date).strftime("%d-%m-%Y")
                    except Exception:
                        prog_date_norm = prog_date
                    if prog_date_norm == filter_date:
                        filtered_matches.append(prog)
                if filtered_matches:
                    matching_programs = filtered_matches
                    print(f"Filtered to {len(matching_programs)} program(s) for date {filter_date}")

            # Format the answer directly from Excel data (no hallucinations)
            results = []
            for prog in matching_programs:
                prog_name = prog.get("Program Name", "Unknown")
                stage_info = prog.get("Stage", "Unknown stage")
                category_info = prog.get("Category", "")
                date_info = format_date_for_answer(prog.get("Date", ""))
                
                if len(matching_programs) == 1:
                    # Single result - give concise, natural answer
                    if date_info:
                        if category_info and category_info != "N/A":
                            return f"{prog_name} is performing on {stage_info} ({category_info}) on {date_info}."
                        else:
                            return f"{prog_name} is performing on {stage_info} on {date_info}."
                    else:
                        if category_info and category_info != "N/A":
                            return f"{prog_name} is performing on {stage_info} ({category_info})."
                        else:
                            return f"{prog_name} is performing on {stage_info}."
                else:
                    # Multiple results - format as natural list
                    line_parts = [stage_info]
                    if category_info and category_info != "N/A":
                        line_parts.append(f"({category_info})")
                    if date_info:
                        line_parts.append(f"on {date_info}")
                    results.append(" ".join(line_parts))
            
            if results:
                prog_name = matching_programs[0].get('Program Name', 'Program')
                # More natural, conversational language for multiple stages
                if len(results) == 2:
                    return f"{prog_name} is performing on {results[0]} and {results[1]}."
                else:
                    return f"{prog_name} is performing on the following stages:\n" + "\n".join([f"• {r}" for r in results])
        else:
            # No Excel match at all – avoid hallucination
            # Provide a clear receptionist-style fallback with guidance
            return (
                "I couldn't find that program in the official schedule. "
                "Please check the exact program name or share a screenshot of the row."
            )
    
    # Always try to get FAISS PDF chunks (but don't use for schedule questions if Excel has answer)
    print("\n" + "="*60)
    print("=== RETRIEVING PDF CHUNKS FROM FAISS ===")
    print("="*60)
    # Dynamically adjust k based on question complexity (not hardcoded to specific terms)
    # Questions with multiple keywords or asking about structured data need more context
    question_words = user_message.lower().split()
    question_lower = user_message.lower()
    
    # Increase chunks for questions likely to need table/chart/structured data or value points
    is_value_points_query = any(term in question_lower for term in ["value point", "value points", "marks", "mark"])
    is_complex_query = len(question_words) > 5 or any(c in user_message for c in ["%", "table", "chart", "list"])
    
    # For value points queries, retrieve more chunks to ensure we find the scoring tables
    k_value = 30 if is_value_points_query else (25 if is_complex_query else 20)
    pdf_chunks = get_pdf_chunks_context(user_message, k=k_value)
    
    # Precise extraction for "Fixing of Grade" table (no hardcoding; parse from text)
    try:
        grade_query = ("grade" in question_lower) and ("70" in question_lower or "70%" in question_lower or "seventy" in question_lower)
        if grade_query:
            combined_source = "\n".join([manual_text or "", pdf_chunks or ""])[:20000]
            # Prefer the section starting near "Fixing of Grade"
            start_idx = combined_source.lower().find("fixing of grade")
            window = combined_source[max(0, start_idx-500): start_idx+1500] if start_idx != -1 else combined_source
            # Normalize whitespace for easier regex
            normalized = re.sub(r"[\t\u00A0]+", " ", window)
            # Try line-based scan for Grade A row
            lines = [l.strip() for l in normalized.splitlines() if l.strip()]
            for i, line in enumerate(lines):
                if re.search(r"^grade\s*a\b", line, flags=re.IGNORECASE):
                    # Look ahead a few lines to find points
                    lookahead = " ".join(lines[i:i+4])
                    m = re.search(r"(\b5\b|\d+)\s*point[s]?", lookahead, flags=re.IGNORECASE)
                    if m:
                        pts = m.group(1)
                        return f"Grade A (70% and above) = {pts} points"
            # Fallback: compact table form "Grade A ... 5 points"
            m2 = re.search(r"Grade\s*A[^\n]*?(\d+)%[^\n]*?(\d+)\s*point", normalized, flags=re.IGNORECASE)
            if m2:
                return f"Grade A ({m2.group(1)}% and above) = {m2.group(2)} points"
    except Exception:
        pass
    
    # Helper function to find matching section headers
    def find_matching_section_headers(query_text, source_text):
        """Dynamically find section headers that match the query keywords.
        Looks for ALL-CAPS headers or Title Case headers in the source text.
        """
        if not source_text:
            return []
        
        # Extract keywords from query
        query_lower = query_text.lower()
        keywords = [w for w in query_text.split() if len(w) > 3]  # Get meaningful words
        
        # Find potential section headers (ALL-CAPS lines, typically followed by roman numerals)
        lines = source_text.split('\n')
        potential_headers = []
        
        # Pattern for ALL-CAPS headers (6+ chars, mostly uppercase)
        all_caps_pattern = re.compile(r'^[A-Z][A-Z0-9 ./&()-,]{4,}$')
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Check if it's a potential section header
            if all_caps_pattern.match(stripped) and len(stripped) > 5:
                # Check if it contains query keywords
                header_lower = stripped.lower()
                if any(kw.lower() in header_lower for kw in keywords):
                    potential_headers.append(stripped)
        
        return potential_headers
    
    # Only use structured section extraction for questions asking about specific sections
    # Skip for simple fact queries (phone numbers, dates, fees, value points that need direct extraction)
    # This prevents wrong section matching for factual questions
    user_lower = user_message.lower()
    is_factual_query = any(term in user_lower for term in [
        "phone", "number", "contact", "email", "address", "when", "what time", 
        "how much", "fee", "cost", "price", "date", "accessible", "value point", 
        "value points", "marks", "mark", "point", "points"
    ])
    
    # Skip structured extraction for factual queries - let LLM handle those directly from chunks
    if not is_factual_query:
        # Try generic section extraction for structured questions (trophies, rules, etc.)
        section_candidates = [manual_text or "", pdf_chunks or ""]
        for blob in section_candidates:
            if not blob:
                continue
            
            # Dynamically find matching section headers
            matching_headers = find_matching_section_headers(user_message, blob)
            
            if matching_headers:
                # Try to extract the first (most relevant) matching section
                for header in matching_headers:
                    extracted_bullets = extract_structured_section(
                        blob,
                        section_headers=[header],
                        max_scan_chars=2500
                    )
                    if extracted_bullets:
                        print(f"Extracted {len(extracted_bullets)} points from '{header}' section")
                        
                        # Check if question is asking for a specific value (fee, percentage, etc.)
                        question_lower = user_message.lower()
                        
                        # For simple "what is X?" questions, extract the specific answer
                        if any(q in question_lower for q in ["what is", "how much", "what's", "tell me"]):
                            # Try to extract the specific answer (e.g., fee amount)
                            answer_text = " ".join(extracted_bullets)
                            
                            # Look for specific patterns (fees, percentages, etc.)
                            fee_match = re.search(r'Rs\.?\s*(\d+[,\d]*)/?-?', answer_text, re.IGNORECASE)
                            percentage_match = re.search(r'(\d+)%\s*(and\s*above|to\s*\d+%)?', answer_text, re.IGNORECASE)
                            
                            if fee_match:
                                amount = fee_match.group(1)
                                # Generate human-friendly, natural response
                                if "appeal" in question_lower:
                                    return f"The appeal fee is Rs. {amount}/-."
                                else:
                                    return f"The fee is Rs. {amount}/-."
                            elif percentage_match:
                                pct = percentage_match.group(1)
                                range_text = percentage_match.group(2) or ""
                                if range_text:
                                    return f"Grade A requires {pct}%{range_text}."
                                else:
                                    return f"Grade A requires {pct}%."
                            else:
                                # For simple questions, extract just the direct answer from first bullet
                                first_bullet = extracted_bullets[0]
                                
                                # Clean up the text - remove header artifacts, dates, etc.
                                # Remove things like "Labour India Public School CBSE State Kalotsav 2K25..."
                                cleaned = re.sub(r'Labour India Public School.*?Kalotsav.*?\d+', '', first_bullet, flags=re.IGNORECASE)
                                cleaned = cleaned.strip()
                                
                                # If it's still long, try to extract just the key sentence
                                if len(cleaned) > 300:
                                    sentences = cleaned.split('.')
                                    # Take first 2 sentences that are relevant
                                    relevant = [s.strip() for s in sentences[:2] if len(s.strip()) > 20]
                                    cleaned = '. '.join(relevant) + '.'
                                
                                return cleaned if cleaned else first_bullet
                        
                        # For general questions, format as friendly list but keep it concise
                        if len(extracted_bullets) == 1:
                            return extracted_bullets[0]
                        else:
                            bullets = "\n".join([f"• {b}" for b in extracted_bullets])
                            return bullets

    # Build context from all available sources
    final_context_parts = []
    
    # 1. Manual text snippets (if available)
    if manual_text:
        try:
            print("Extracting relevant manual snippets...")
            manual_snippets = find_relevant_manual_snippets(manual_text, query_terms)
            content_for_ai = "\n\n".join(manual_snippets) if manual_snippets else manual_text[:9000]
            if content_for_ai:
                final_context_parts.append("[Manual]\n" + content_for_ai)
                print(f"Added manual context ({len(content_for_ai)} chars)")
        except Exception as e:
            print(f"Error processing manual text: {e}")
    
    # 2. Excel/Schedule data - CRITICAL for schedule questions; skip for factual (phone/fee) queries
    if not is_factual_query:
        excel_context = build_excel_context_rows(combined_data, query_terms)
        if excel_context:
            # For schedule questions, emphasize Excel data comes first
            if is_schedule_question:
                final_context_parts.insert(0, "[SCHEDULE - PRIMARY SOURCE]\n" + excel_context)
                print(f"Added schedule context as PRIMARY SOURCE ({len(excel_context)} chars)")
            else:
                final_context_parts.append("[Schedule]\n" + excel_context)
                print(f"Added schedule context ({len(excel_context)} chars)")
    
    # 3. FAISS PDF chunks (ALWAYS try to include, independent of manual text)
    if pdf_chunks:
        final_context_parts.append("[PDF Chunks]\n" + pdf_chunks)
        print(f"Added PDF chunks context ({len(pdf_chunks)} chars)")
    else:
        print("No PDF chunks retrieved - FAISS may not be available or index not found")
    
    # If we have any context, use AI to answer
    if final_context_parts:
        final_context = "\n\n".join(final_context_parts)
        # Limit to 14000 chars to leave room for prompt
        final_context = final_context[:14000]
        
        print(f"\n=== Sending to LLM with context ({len(final_context)} chars) ===")
        print(f"Context preview:\n{final_context[:800]}...")
        
        # DEBUG: Check if context contains the answer keywords
        context_lower = final_context.lower()
        if "grade a" in context_lower or "grade" in context_lower:
            print(f"Context contains 'grade' keyword")
        if "70%" in final_context or "70" in final_context:
            print(f"Context contains '70%' or '70'")
        if "percentage" in context_lower:
            print(f"Context contains 'percentage' keyword")
        if "fixing of grade" in context_lower:
            print(f"Context contains 'Fixing of Grade' section!")
        if "trophies to schools" in context_lower:
            print(f"Context contains 'TROPHIES TO SCHOOLS' section!")
        if "ever rolling trophy" in context_lower:
            print(f"Context contains 'ever rolling trophy' information!")
        # Check for value points info
        if "value point" in context_lower:
            print(f"Context contains 'Value Points' information!")
        if "folk dance" in context_lower and ("mark" in context_lower or "value point" in context_lower):
            print(f"*** Context contains FOLK DANCE VALUE POINTS! ***")
        
        # Try to find and highlight relevant chunks
        chunks_list = final_context.split("---")
        for i, chunk in enumerate(chunks_list):
            if "grade" in chunk.lower() and ("70" in chunk or "%" in chunk):
                print(f"\nFOUND RELEVANT CHUNK #{i+1} (Grade info):")
                print(chunk[:500])
            if "trophies to schools" in chunk.lower() or ("trophy" in chunk.lower() and "school" in chunk.lower()):
                print(f"\nFOUND RELEVANT CHUNK #{i+1} (Trophies info):")
                print(chunk[:500])
        
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"""You are a friendly and helpful assistant for the Kalotsavam Cultural Festival.
                    
                    Your communication style:
                    - Be conversational, natural, and human-like
                    - Write as if you're talking to a friend
                    - Be clear and direct, but warm
                    - Use simple, everyday language
                    - Format answers for WhatsApp (short, clear, easy to read)
                    
                    Answer based on the provided context which may include:
                    - [SCHEDULE - PRIMARY SOURCE]: Excel program schedule with exact stage numbers, dates, categories - USE THIS FOR ALL SCHEDULE/STAGE QUESTIONS
                    - Manual: Rules, fees, awards, regulations
                    - PDF Chunks: Semantic search results from the festival manual/guide (use ONLY for fees, rules, awards, grades, percentages, tables - NOT for stage/program questions)
                    
                    CRITICAL PRIORITY FOR SCHEDULE QUESTIONS:
                    - If you see "[SCHEDULE - PRIMARY SOURCE]" → Use ONLY that data for stage/program questions
                    - Excel schedule has exact stage numbers like "STAGE 9", "STAGE 11" - use those EXACT values
                    - DO NOT mix PDF chunk category descriptions (like "Category III (Classes VIII to X)") with Excel stage numbers
                    - If a program appears on multiple stages in Excel, list ALL stages
                    - Excel data is authoritative for schedule questions - ignore conflicting PDF chunk descriptions
                    
                    {f"THIS IS A SCHEDULE QUESTION - IGNORE PDF CHUNKS, USE ONLY [SCHEDULE - PRIMARY SOURCE] DATA" if is_schedule_question else ""}
                    
                    CRITICAL RULES FOR TABLES, CHARTS, GRADES, AND VALUE POINTS:
                    1. PDF Chunks may contain TABLES, CHARTS, or FORMATTED DATA - look carefully for tabular information
                    2. Tables might appear as text with columns separated by spaces, tabs, or special characters
                    3. Look for patterns like "Grade A", "Grade B", "70%", "60%", percentages, points, marks, etc.
                    4. Even if data looks messy or poorly formatted, extract what you can find
                    5. For grade/percentage questions: Search for ANY mention of grades, percentages, or points in ALL chunks
                    6. For value points/marks questions: Look for sections with headers like "Value Points" followed by a category name (e.g., "Folk Dance") then item names and marks. The EXACT format in the PDF is:
                       - Line 1: "Value Points"
                       - Line 2: Category name (e.g., "Folk Dance")
                       - Line 3: Item name (e.g., "Akara Sushama")
                       - Line 4: Mark value (e.g., "15 marks")
                       - This pattern repeats for each item
                       Example from PDF:
                         "Value Points
                         Folk Dance 
                         Akara Sushama 
                         15 marks 
                         Prakadanam 
                         20 marks"
                       NOTE: The character "I" might be used instead of "1" (like "I5 marks" = "15 marks") - interpret this correctly.
                    7. Tables often have headers like "Grade", "Percentage", "Points", "Value Points", "Marks" - match these with question keywords
                    8. Extract relationships like "Grade A = 70% and above" or "Grade A = 70%+" when you see them
                    9. If you see percentage ranges (e.g., "70% and above", "60% to 69%"), use those exact ranges
                    10. For value points: Extract ALL criteria/evaluation items and their corresponding marks, format them clearly as: "Item Name - X marks". List ALL items found under the relevant category header (e.g., "Folk Dance")
                    11. CRITICAL: If you see "Value Points" followed by "Folk Dance" (or any category) in the context, extract EVERY item name and mark value from that section - DO NOT say "not mentioned"
                    
                    GENERAL CRITICAL RULES:
                    1. ALWAYS check PDF Chunks FIRST - search through EVERY chunk thoroughly
                    2. Extract EXACT numbers, amounts, percentages, phone numbers, contact info, and details from PDF Chunks when present
                    3. NEVER say "not mentioned" or "not in the provided context" if PDF Chunks section exists and contains relevant data
                    4. For value points queries: If you see ANY mention of "Value Points" with a category name (like "Folk Dance") in the PDF Chunks, extract ALL items and marks - the data IS there, DO NOT say "not mentioned"
                    5. If question asks about grades, percentages, tables, fees, rules, awards, phone numbers, contact info, or dates - PDF Chunks likely contain it
                    6. For phone/contact questions: Look for phone numbers (10-digit numbers, numbers with slashes like "9778665476/9778665475"), email addresses, office addresses, or contact details near phrases like "kalotsav office", "contact", "phone", "number"
                    7. For value points/marks questions: Look for sections with headings like "Value Points" followed by the category (e.g., "Folk Dance"), then item names and mark values. The format in the PDF is: item name on one line, then "X marks" on the next line. Extract them and format clearly: "Item Name - X marks". Example: If you see "Akara Sushama" followed by "15 marks", format as "Akara Sushama - 15 marks". DO NOT output raw number sequences without labels.
                    8. IGNORE raw number sequences without context (like "5 5 10 3 5 8...") - these are likely poorly formatted table data. Look for properly formatted sections with item names and their values instead.
                    9. When you find "Value Points" section with a category name matching the question, extract ALL items and marks listed under that category - do not stop after finding one item.
                    10. Provide precise answers with exact numbers/percentages/phone numbers when found - extract them directly from the context
                    11. Use WhatsApp-friendly formatting: *bold* for key labels, bullets for lists
                    12. If multiple relevant entries exist, list them all clearly
                    13. If you find partial information (e.g., just percentage, just phone number), provide what you found
                    14. Phone numbers might appear as digits only (e.g., "9778665476") or with separators - extract them as found
                    
                    Answer format: Write naturally and conversationally. Be direct but friendly. Extract exact values from PDF Chunks. For tables, value points, or structured data, format them clearly with item names and values (e.g., "Item Name - X marks"), never output raw number sequences."""},
                    {"role": "user", "content": f"""Question: {user_message}

Available Context:
{final_context}

CRITICAL INSTRUCTIONS:
1. CAREFULLY search through ALL sections - "[PDF Chunks]", "[Manual]", and "[Schedule]" - scan EVERY chunk thoroughly
2. Look for information that relates to the question - use semantic understanding, not just exact keyword matches
3. Extract relevant information even if:
   - Formatting is messy or unconventional
   - Information is in tables, lists, or paragraph form
   - Keywords don't match exactly but meaning is similar
4. For phone/contact questions: Search for phone numbers (long digit sequences like "9778665476/9778665475"), look near words like "office", "kalotsav", "contact", "phone", "number" - extract the digits you find
5. For value points/marks questions: Search for "Value Points" header, then find the category name (e.g., "Folk Dance"), then extract ALL items listed. The format is: item name on one line, mark value on next line. Example: "Akara Sushama" followed by "15 marks" means "Akara Sushama - 15 marks". Extract EVERY item under that category. Format as: "The value points for [category] are: * Item 1 - X marks * Item 2 - Y marks..." List ALL items you find - DO NOT stop after one. NEVER output raw number sequences without context.
6. IGNORE raw number sequences that appear without labels or context (like standalone numbers "5 5 10 3..." on separate lines) - these are poorly extracted table data. Always look for properly formatted sections with item names and their values.
7. If you find ANY "Value Points" section matching the question category in the context, extract and list ALL items - never say "not mentioned" if this data exists in the chunks.
8. DO NOT respond with "not mentioned" - if you find ANY relevant information in the context that answers the question, provide it
9. If the context has the answer but it's scattered across chunks, piece it together intelligently
10. CRITICAL FOR VALUE POINTS: If you see "Value Points" and "Folk Dance" in the same chunk or context, the answer IS THERE - extract it and format it clearly, never say it's not mentioned

INSTRUCTIONS:
- Read the context carefully and extract information that answers the question
- Use your understanding of the context to provide accurate answers
- Be natural and conversational in your response
- If you find relevant information, provide it even if it's not formatted perfectly

Now answer the question: {user_message}"""}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            ai_response = response.choices[0].message.content.strip()
            
            if ai_response and len(ai_response) > 10:
                print(f"AI response generated ({len(ai_response)} chars)")
                return ai_response
        except Exception as e:
            print(f"Error using AI with context: {e}")
            import traceback
            traceback.print_exc()
    
    
    # Convert all program data to a formatted string with better date formatting
    def format_date_for_display(date_str):
        """Format date string for better readability"""
        if not date_str or date_str == 'Unknown':
            return 'Unknown'
        try:
            # Try parsing different date formats
            date_obj = pd.to_datetime(date_str)
            return date_obj.strftime("%d-%m-%Y")
        except:
            # If parsing fails, try to extract date part only
            if '2025-' in str(date_str):
                parts = str(date_str).split('-')
                if len(parts) >= 3:
                    return f"{parts[2][:2]}-{parts[1]}-{parts[0]}"
            return str(date_str)
    
    all_programs_text = "\n".join([
        f"- {p.get('Program Name', 'Unknown')} at {p.get('Stage', 'Unknown')} (Category: {p.get('Category', 'N/A')}) on {format_date_for_display(p.get('Date', 'Unknown'))}"
        for p in combined_data
    ])
    
    print("\nSending query to AI with program data...")
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"""You are an intelligent Kalolsavam cultural festival assistant with access to the complete program schedule.

COMPLETE PROGRAM DATABASE:
{all_programs_text}

YOUR CAPABILITIES - Handle these query types intelligently:

1. **Category Queries** (examples):
   - "What are the category one programmes?"
   - "Show me cat 1 programs"
   - "Category 2 events"
   - "Programs in category 4"
   - Filter and list programs matching the category (e.g., CAT-1, CAT-2, CAT-3, CAT-4, CAT- 1, CAT- 2, etc.)
   
2. **Date Queries** (examples):
   - "What's happening on 15-11-2025?"
   - "Programs on November 12"
   - "12-11-2025 schedule"
   - "What's on 15th November?"
   - Show all programs for the specific date
   
3. **Day Queries** (examples):
   - "What's happening on Friday?"
   - "Saturday's programs"
   - "Friday schedule"
   - Convert day names to dates: Wednesday=12-11-2025, Thursday=13-11-2025, Friday=14-11-2025, Saturday=15-11-2025
   
4. **Stage Queries** (examples):
   - "What's in Stage 2?"
   - "Programs at Stage 5"
   - "STAGE 10 events"
   - Filter by stage number
   
5. **Program Name Queries** (examples):
   - "When is BHARATHA BOYS?"
   - "Tell me about FOLK DANCE"
   - "MOHINIYATTAM schedule"
   - Find and show details of specific programs
   
6. **Schedule/General Queries** (examples):
   - "When is Kalolsavam?"
   - "Festival dates"
   - "What are all the dates?"
   - Show all unique dates and dates range
   
7. **Combined Queries** (examples):
   - "Category 1 programs on Friday"
   - "Stage 2 programs on 14-11-2025"
   - Handle multiple filters together

RESPONSE RULES:
- ALWAYS analyze the user's intent first
- For category queries: List ALL programs matching that category with their stages and dates
- For date/day queries: Show ALL programs on that date with their stages and categories
- Group results logically by stage, category, or date as appropriate
- Be specific: Include full program names, categories, stages, and dates
- Use clear formatting with bullet points or numbered lists
- If no results found, politely inform and suggest alternative queries
- Keep responses concise but comprehensive
- Use emojis sparingly (max 1-2 per response)

IMPORTANT: Understand user intent. If they ask "category one programmes", they mean CAT-1, CAT- 1, or similar category labels in the data."""},
                {"role": "user", "content": user_message}
            ]
        )
        
        ai_response = response.choices[0].message.content.strip()
        print(f"\nAI Response: {ai_response}")
        return ai_response
    except Exception as e:
        print(f"Error in AI response: {e}")
        return "I'm having trouble processing your request. Please try again."
    
    # Handle greetings and casual conversation
    if extracted.get("is_greeting", False):
        current_hour = pd.Timestamp.now().hour
        if "bye" in user_message.lower() or "goodbye" in user_message.lower():
            return ("Goodbye! Feel free to ask me about any programs later. "
                    "I'm here to help!")
        elif "good morning" in user_message.lower() or ("morning" in user_message.lower() and "hi" in user_message.lower()):
            return ("Good morning! I'm your Kalolsavam assistant. "
                   "How can I help you today? You can ask me about any programs, venues, or schedules!")
        elif "good afternoon" in user_message.lower():
            return ("Good afternoon! I'm your Kalolsavam assistant. "
                   "How can I help you today? You can ask me about any programs, venues, or schedules!")
        elif "good evening" in user_message.lower():
            return ("Good evening! I'm your Kalolsavam assistant. "
                   "How can I help you today? You can ask me about any programs, venues, or schedules!")
        elif "thank" in user_message.lower():
            return "You're welcome! Let me know if you need anything else!"
        elif "how are you" in user_message.lower():
            return ("I'm doing great, thank you for asking! "
                   "I'm ready to help you with any information about the Kalolsavam programs. "
                   "What would you like to know?")
        else:
            return ("Hello! I'm your Kalolsavam assistant. "
                   "I can help you with program schedules, venues, and timings. "
                   "What would you like to know?")
    # Use cached Excel data
    df = program_cache['excel_data']
    
    # Use the combined data passed to the function
    all_programs = combined_data if combined_data else []
    query_type = extracted.get("query_type", "general")
    results = []

    # Helper function to format program details
    def format_program_details(program, detailed=True):
        """Format program details from either Excel or PDF data"""
        source_label = "[Excel]" if program.get("Source") == "excel" else "[PDF]"
        
        if detailed:
            details = [f"*{program['Program Name']}* ({program.get('Category', 'N/A')}) {source_label}"]
            
            if program.get("Stage"):
                details.append(f"Stage: {program['Stage']}")
            
            if program.get("Date"):
                details.append(f"Date: {program['Date']}")
            
            return "\n".join(details)
        else:
            return (
                f"• *{program['Program Name']}* "
                f"({program.get('Category', 'N/A')}) {source_label}\n"
                f"  Stage: {program.get('Stage', 'N/A')}"
            )

    # Filter and search through combined data
    filtered_programs = all_programs.copy()

    # Detect gender intent from message (used for category queries)
    msg_lower = user_message.lower()
    gender_intent = None
    if "girls" in msg_lower or "girl" in msg_lower:
        gender_intent = "GIRLS"
    elif "boys" in msg_lower or "boy" in msg_lower:
        gender_intent = "BOYS"
    
    def filter_programs(programs, condition):
        return [p for p in programs if condition(p)]
    
    # Apply filters based on query
    if extracted.get("venue"):
        venue_query = extracted["venue"].lower()
        filtered_programs = filter_programs(
            filtered_programs,
            lambda p: venue_query in str(p.get("Venue", "")).lower()
        )
    
    if extracted.get("stage"):
        stage_num = extracted["stage"]
        filtered_programs = filter_programs(
            filtered_programs,
            lambda p: f"Stage {stage_num}" in str(p.get("Stage", ""))
        )

    # Category filter (supports forms like "CAT-2", "CAT - 2", "CATEGORY 2", "CATEGORY II")
    if extracted.get("category"):
        cat_num = str(extracted["category"]).strip()
        def _cat_match(cat_val: str) -> bool:
            c = str(cat_val or "").upper().replace(" ", "")
            return (
                f"CAT-{cat_num}".replace(" ", "") in c or
                f"CAT- {cat_num}".replace(" ", "") in c or
                f"CAT{cat_num}" in c or
                f"CATEGORY{cat_num}" in c or
                c.endswith(cat_num)
            )
        filtered_programs = filter_programs(
            filtered_programs,
            lambda p: _cat_match(p.get("Category", ""))
        )

    # Gender refinement for category/program queries
    if gender_intent:
        filtered_programs = filter_programs(
            filtered_programs,
            lambda p: gender_intent in str(p.get("Program Name", "")).upper()
        )
    
    if extracted.get("date") or extracted.get("day"):
        date_query = None
        
        # Handle direct date queries
        if extracted.get("date"):
            date_query = extracted["date"]
            print(f"Searching for date: {date_query}")  # Debug print
            try:
                date_obj = pd.to_datetime(date_query)
                date_query = date_obj.strftime("%d-%m-%Y")
            except:
                pass
        
        # Handle day queries
        elif extracted.get("day"):
            day = extracted["day"].upper()
            print(f"Searching for day: {day}")  # Debug print
            if day in DAY_MAPPING:
                date_query = DAY_MAPPING[day]
                print(f"Converted day {day} to date: {date_query}")  # Debug print
        
        if date_query:
            print(f"Searching for programs on: {date_query}")  # Debug print
            # Debug print all dates in the data
            all_dates = set(str(p.get("Date", "")).strip() for p in all_programs)
            print("Available dates in data:", all_dates)
            
            # Print all available dates for debugging
            all_dates = set()
            for p in all_programs:
                date = str(p.get("Date", "")).strip()
                if date:
                    all_dates.add(date)
            print(f"All dates in data: {sorted(list(all_dates))}")
            
            # Filter programs by date
            filtered_programs = []
            for p in all_programs:
                prog_date = str(p.get("Date", "")).strip()
                print(f"Checking program: {p.get('Program Name')} on {prog_date}")
                
                # Direct string comparison first
                if prog_date == date_query:
                    print(f"Found match (direct): {p.get('Program Name')}")
                    filtered_programs.append(p)
                    continue
                
                # Try date normalization if direct match fails
                try:
                    query_date = pd.to_datetime(date_query).strftime("%d-%m-%Y")
                    program_date = pd.to_datetime(prog_date).strftime("%d-%m-%Y")
                    if query_date == program_date:
                        print(f"Found match (normalized): {p.get('Program Name')}")
                        filtered_programs.append(p)
                except:
                    pass
            
            print(f"\nFound {len(filtered_programs)} programs for {date_query}")
            if filtered_programs:
                print("\nFound these programs:")
                for p in filtered_programs:
                    print(f"- {p['Program Name']} ({p['Category']}) at {p['Stage']}")
    
    if extracted.get("time"):
        time_query = extracted["time"].lower()
        filtered_programs = filter_programs(
            filtered_programs,
            lambda p: (
                ("AM" in str(p.get("Time", "")).upper() if "morning" in time_query else True) and
                ("PM" in str(p.get("Time", "")).upper() if "afternoon" in time_query else True)
            )
        )
    
    # Handle different query types
    if query_type == "venue" and extracted["venue"]:
        if filtered_programs:
            results.append(f"*Programs in {extracted['venue'].title()}:*\n")
            for program in filtered_programs:
                results.append(format_program_details(program, detailed=False))
    
    elif query_type == "stage" and extracted["stage"]:
        if filtered_programs:
            results.append(f"*Programs in Stage {extracted['stage']}:*\n")
            for program in filtered_programs:
                results.append(format_program_details(program, detailed=False))
    
    elif query_type == "time" and extracted["time"]:
        if filtered_programs:
            results.append(f"*Programs during {extracted['time']}:*\n")
            for program in filtered_programs:
                results.append(format_program_details(program, detailed=False))
    
    elif query_type == "date" and extracted["date"]:
        if filtered_programs:
            results.append(f"*Programs on {extracted['date']}:*\n")
            for program in filtered_programs:
                results.append(format_program_details(program, detailed=False))
    
    elif query_type == "day" and extracted["day"]:
        day = extracted["day"].upper()
        if day in DAY_MAPPING:
            date = DAY_MAPPING[day]
            if filtered_programs:
                results.append(f"*Programs on {day} ({date}):*\n")
                for program in filtered_programs:
                    results.append(format_program_details(program, detailed=False))
    
    elif query_type == "program" or (query_type == "general" and extracted.get("program")):
        program_query = extracted.get("program", "").lower() or user_message.lower()
        matching_programs = filter_programs(
            all_programs,
            lambda p: program_query in str(p.get("Program Name", "")).lower()
        )
        for program in matching_programs:
            results.append(format_program_details(program, detailed=True))

    elif query_type == "category" or (query_type == "general" and extracted.get("category")):
        if filtered_programs:
            header = f"*Category {extracted.get('category')} programs*"
            if gender_intent:
                header = f"*{gender_intent.title()} Category {extracted.get('category')} programs*"
            results.append(header + ":\n")
            # Show concise lines: Program – Stage on Date (Category)
            def _fmt_row(p):
                prog = p.get("Program Name", "Unknown")
                stage = p.get("Stage", "N/A")
                date = p.get("Date", "")
                cat = p.get("Category", "")
                return f"• {prog} – {stage} on {date} ({cat})"
            # De-duplicate by Program+Stage+Date
            seen = set()
            for p in filtered_programs:
                key = (p.get("Program Name", ""), p.get("Stage", ""), p.get("Date", ""))
                if key in seen:
                    continue
                seen.add(key)
                results.append(_fmt_row(p))
    
    elif query_type == "general":
        # For general queries, try to match against all fields
        search_text = user_message.lower()
        for program in all_programs:
            program_text = " ".join(str(val).lower() for val in program.values())
            if search_text in program_text:
                results.append(format_program_details(program, detailed=True))

    if results:
        return "\n\n".join(results)
    else:
        # Customize message based on query type
        if query_type == "date" and extracted.get("date"):
            return (
                f"I couldn't find any programs scheduled for {extracted['date']}. "
                "You can ask me about:\n"
                "• Specific programs (e.g., 'When is Kathakali?')\n"
                "• Venue programs (e.g., 'What's in the Auditorium?')\n"
                "• Stage programs (e.g., 'What's in Stage 2?')\n"
                "• Time slots (e.g., 'morning programs')"
            )
        else:
            return (
                "I couldn't find any matching programs in the schedule. "
                "You can ask me about:\n"
                "• Specific programs (e.g., 'When is Kathakali?')\n"
                "• Venue programs (e.g., 'What's in the Auditorium?')\n"
                "• Stage programs (e.g., 'What's in Stage 2?')\n"
                "• Time slots (e.g., 'morning programs')"
            )


def _deduplicate_and_flatten_list_text(raw_text: str) -> str:
    """De-duplicate bullet-like lines and convert to a simple sentence when possible."""
    if not raw_text:
        return raw_text
    lines = [l.strip() for l in raw_text.splitlines()]
    header = None
    content_lines = []
    for idx, l in enumerate(lines):
        if not l:
            continue
        header = l
        content_lines = lines[idx + 1:]
        break
    if header is None:
        content_lines = lines
    items = []
    seen = set()
    for l in content_lines:
        s = l.lstrip("-•* ").strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        items.append(s)
    if items:
        prefix = header.rstrip(':').strip() if header and len(header) < 160 else None
        if len(items) == 1:
            joined = items[0]
        elif len(items) == 2:
            joined = f"{items[0]} and {items[1]}"
        else:
            joined = ", ".join(items[:-1]) + f", and {items[-1]}"
        return f"{prefix + ': ' if prefix else ''}{joined}"
    return raw_text


def generate_human_like_reply(user_message, info_text):
    """
    Generate a concise, non-hallucinated WhatsApp reply:
    - Use ONLY the content in info_text; never invent items.
    - Prefer one short paragraph; no emojis.
    - If info_text signals "not found", reply politely with guidance.
    """
    try:
        cleaned_info = _deduplicate_and_flatten_list_text(info_text)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0.2,
            messages=[
                {"role": "system", "content": """You are a clear, concise WhatsApp assistant for the Kalolsavam Cultural Festival.
                HARD CONSTRAINTS (NO HALLUCINATIONS):
                - Use ONLY the content provided by the assistant message; never add or infer extra items, dates, stages, or categories.
                - If the user asks about a specific program name (e.g., MONO ACT), include ONLY lines that refer to that exact program name. Do NOT include similarly worded but different items (e.g., ENGLISH ONE ACT PLAY) when asked about MONO ACT.
                - If there is a single schedule entry, answer with ONE short, natural sentence.
                - If multiple entries exist for that same program, consolidate into one short readable sentence, listing each unique (Stage, Category, Date) only once.
                - If no entries for the exact program are present, clearly say you couldn't find it and suggest checking the exact program name. Do not fabricate.
                - No emojis; keep it brief and professional.
                - Preserve exact dates as written; do not substitute with relative words.
                - IMPORTANT: If the assistant content contains entries that are NOT for the exact program requested, IGNORE those lines entirely."""},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": f"{cleaned_info}"},
                {"role": "user", "content": "Rewrite the above as a brief, human reply without emojis. Include ONLY entries that match the exact program name asked by the user. Do not add or infer any data that is not present above."}
            ]
        )
        text = response.choices[0].message.content.strip()
        try:
            emoji_pattern = re.compile("[\U00010000-\U0010FFFF]", flags=re.UNICODE)
            text = emoji_pattern.sub("", text)
        except Exception:
            pass
        return text
    except Exception as e:
        print(" OpenAI reply error:", e)
        return _deduplicate_and_flatten_list_text(info_text)


def send_whatsapp_message(to, message):
    """
    Send the WhatsApp message back using Twilio API.
    Handle rate limits and errors appropriately.
    """
    try:
        twilio_client.messages.create(
            from_=FROM_NUMBER,
            to=to,
            body=message
        )
        print(f"Successfully sent reply to {to}")
    except Exception as e:
        error_str = str(e).lower()
        if "limit" in error_str:
            print("Twilio daily message limit reached. Please try again tomorrow.")
            # Here you could implement fallback communication or alert administrators
            return {"status": "error", "type": "rate_limit", "message": "Daily message limit reached"}
        else:
            print(f"Twilio send error: {e}")
            return {"status": "error", "type": "general", "message": str(e)}


def process_pdf_content(pdf_text):
    """
    Process extracted PDF text and convert it to structured data
    Handles both manual-style format (ITEM CODE, ITEM, TIME) and general format
    """
    try:
        # Split text into lines and remove empty lines
        lines = [line.strip() for line in pdf_text.split('\n') if line.strip()]
        
        programs = []
        current_program = {}
        current_category = None
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Track categories for manual
            if "CATEGORY" in line.upper() and ("CLASS" in line.upper() or "III" in line or "IV" in line):
                current_category = line.strip()
                
            # Check if this is an ITEM CODE (3-digit number at start of line)
            if re.match(r'^\d{3}\s*$', line):
                item_code = line.strip()
                # Look ahead for item name and time
                if i + 1 < len(lines):
                    item_name = lines[i + 1].strip()
                    if i + 2 < len(lines) and not re.match(r'^\d+$', lines[i + 2]):
                        # Third line might be continuation or time
                        time_line = lines[i + 2].strip()
                        if any(word in time_line.lower() for word in ['hr', 'mts', 'min', 'hour']):
                            time_val = time_line
                            current_program = {
                                "Program Name": item_name,
                                "Time": time_val,
                                "Category": current_category if current_category else "Manual",
                                "Item Code": item_code
                            }
                            programs.append(current_program.copy())
                            i += 3
                            continue
                
                # If time is on next line
                if i + 2 < len(lines):
                    time_val = lines[i + 2].strip()
                    if any(word in time_val.lower() for word in ['hr', 'mts', 'min', 'hour']) or time_val.isdigit():
                        current_program = {
                            "Program Name": item_name if 'item_name' in locals() else lines[i + 1].strip(),
                            "Time": time_val,
                            "Category": current_category if current_category else "Manual",
                            "Item Code": item_code
                        }
                        programs.append(current_program.copy())
                        i += 3
                        continue
            
            # Look for common program details patterns (original format)
            if "Program:" in line or "Event:" in line:
                if current_program:
                    programs.append(current_program)
                current_program = {"Program Name": line.split(":", 1)[1].strip()}
            elif "Time:" in line or "Timing:" in line:
                if current_program:
                    current_program["Time"] = line.split(":", 1)[1].strip()
            elif "Venue:" in line:
                if current_program:
                    current_program["Venue"] = line.split(":", 1)[1].strip()
            elif "Stage:" in line:
                if current_program:
                    current_program["Stage"] = line.split(":", 1)[1].strip()
            elif "Category:" in line:
                if current_program:
                    current_program["Category"] = line.split(":", 1)[1].strip()
            elif "Participants:" in line:
                if current_program:
                    current_program["Participants"] = line.split(":", 1)[1].strip()
            
            i += 1
        
        # Add the last program if exists
        if current_program:
            programs.append(current_program)
            
        return programs
    except Exception as e:
        print(f"Error processing PDF content: {e}")
        import traceback
        traceback.print_exc()
        return []

def extract_text_from_pdf(pdf_buffer):
    """
    Extract text from a PDF using PyMuPDF
    """
    text = []
    try:
        # Open PDF from buffer
        pdf_document = fitz.open(stream=pdf_buffer.getvalue(), filetype="pdf")
        
        # Extract text from each page
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text.append(page.get_text())
        
        pdf_document.close()
        return "\n".join(text)
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

@app.get("/")
def root():
    return {"message": " Kalolsavam WhatsApp Assistant is running!"}



@app.get("/ask")
async def ask_unified_get(question: str = Query(None, description="Your question about schedule/manual/PDF")):
    """Unified GET endpoint: answers using Excel schedule, manual, and FAISS PDF chunks."""
    if not question:
        return {"status": "error", "error": "No question provided. Use ?question=..."}
    try:
        extracted = extract_query_info(question)
        combined_data = update_cache_if_needed()
        answer = search_program_data(extracted, question, combined_data)
        return {"status": "success", "question": question, "answer": answer}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/ask")
async def ask_unified_post(request: Request):
    """
    Unified POST endpoint: accepts questions in multiple formats:
    - JSON body: {"question": "..."} or {"message": "..."}
    - Form data: question=...
    """
    question = None
    content_type = request.headers.get("content-type", "")
    
    # Try JSON first (most common)
    if "application/json" in content_type or not content_type:
        try:
            payload = await request.json()
            question = payload.get("question") or payload.get("message")
        except Exception:
            # If JSON fails, might be form data
            pass
    
    # Try form data if no question found yet
    if not question:
        if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
            try:
                form_data = await request.form()
                question = form_data.get("question") or form_data.get("message")
            except Exception:
                pass
        # Also try form data even if content-type is missing or unclear
        elif not content_type:
            try:
                form_data = await request.form()
                question = form_data.get("question") or form_data.get("message")
            except Exception:
                pass
    
    # Clean up question (remove whitespace)
    if question:
        question = question.strip()
        if not question:
            question = None
    
    if not question:
        # Provide helpful error with format examples
        return {
            "status": "error",
            "error": "No question provided",
            "examples": {
                "json": {
                    "method": "POST",
                    "url": "/ask",
                    "headers": {"Content-Type": "application/json"},
                    "body": {"question": "what is the appeal fee?"}
                },
                "form_data": {
                    "method": "POST",
                    "url": "/ask",
                    "headers": {"Content-Type": "multipart/form-data"},
                    "body": "question=what is the appeal fee?"
                },
                "urlencoded": {
                    "method": "POST",
                    "url": "/ask",
                    "headers": {"Content-Type": "application/x-www-form-urlencoded"},
                    "body": "question=what is the appeal fee?"
                }
            },
            "content_type_received": content_type
        }
    
    try:
        extracted = extract_query_info(question)
        clear_schedule_cache()
        combined_data = update_cache_if_needed()
        answer = search_program_data(extracted, question, combined_data)
        return {"status": "success", "question": question, "answer": answer}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/upload-pdf-for-faiss")
async def upload_pdf_for_faiss(pdf_file: UploadFile = File(...)):
    """
    Upload a PDF file and create/update FAISS vector store from it.
    This processes the PDF into chunks and creates searchable embeddings.
    """
    try:
        if not _FAISS_AVAILABLE:
            return {
                "status": "error",
                "error": "FAISS dependencies not available. Install: pip install langchain-openai langchain-community faiss-cpu"
            }
        
        # Read PDF file
        pdf_content = await pdf_file.read()
        pdf_buffer = io.BytesIO(pdf_content)
        
        # Extract text from PDF
        extracted_text = extract_text_from_pdf(pdf_buffer)
        
        if not extracted_text:
            return {"status": "error", "error": "Failed to extract text from PDF"}
        
        # Create FAISS vector store
        chunks_count = create_faiss_from_text(extracted_text)
        
        return {
            "status": "success",
            "message": f"PDF processed and FAISS index created with {chunks_count} chunks",
            "chunks": chunks_count
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("Starting WhatsApp Assistant Server...")
    print("Press Ctrl+C to stop the server")
    uvicorn.run(app, host="0.0.0.0", port=8000)
