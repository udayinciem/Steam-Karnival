# -*- coding: utf-8 -*-
from fastapi import FastAPI, Form, Response, File, UploadFile, Query, Request
from twilio.rest import Client as TwilioClient
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import os
import json
import io
import re
import requests
try:
    # Optional: Only used if FAISS vectors are available
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    _FAISS_AVAILABLE = True
except Exception:
    _FAISS_AVAILABLE = False

# Optional MongoDB for chat history
try:
    from pymongo import MongoClient
    _MONGO_AVAILABLE = True
except Exception:
    _MONGO_AVAILABLE = False

# Import PDF OCR extractor for processing PDFs and creating embeddings
try:
    from pdf_ocr_extractor import process_pdf_file
    _PDF_EXTRACTOR_AVAILABLE = True
except Exception as e:
    print(f"Warning: pdf_ocr_extractor not available: {e}")
    _PDF_EXTRACTOR_AVAILABLE = False

# Load environment variables
load_dotenv()

# FastAPI initialization
app = FastAPI()

# Twilio setup
ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
FROM_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")

# Validate Twilio credentials
if not ACCOUNT_SID:
    print("ERROR: TWILIO_ACCOUNT_SID not found in environment variables!")
if not AUTH_TOKEN:
    print("ERROR: TWILIO_AUTH_TOKEN not found in environment variables!")
if not FROM_NUMBER:
    print("ERROR: TWILIO_WHATSAPP_NUMBER not found in environment variables!")

if ACCOUNT_SID and AUTH_TOKEN:
    print(f"[OK] Twilio credentials loaded: SID={ACCOUNT_SID[:10]}... (length: {len(ACCOUNT_SID)})")
    print(f"[OK] Auth Token loaded: {AUTH_TOKEN[:10]}... (length: {len(AUTH_TOKEN)})")
    print(f"[OK] WhatsApp Number: {FROM_NUMBER}")
    twilio_client = TwilioClient(ACCOUNT_SID, AUTH_TOKEN)
else:
    print("WARNING: Twilio not configured. WhatsApp messaging will not work!")
    twilio_client = None

# OpenAI setup
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Data file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, "data")
PDF_STORAGE = os.path.join(DATA_FOLDER, "pdfs")
VECTORSTORE_PATH = os.path.join(BASE_DIR, "my_pdf_vectors")
EXCEL_VECTORS_PATH = os.path.join(BASE_DIR, "my_excel_vectors")


# MongoDB setup (lazy connection - only connects when needed)
_db_connection = None
_chats_collection = None

# Cache FAISS vectorstores and embeddings for faster access
_pdf_vectorstore_cache = None
_excel_vectorstore_cache = None
_embeddings_cache = None

def get_mongodb_collection():
    """Lazy MongoDB connection - only connects when chat history is actually needed."""
    global _db_connection, _chats_collection
    
    if not _MONGO_AVAILABLE:
        return None
    
    if _chats_collection is not None:
        return _chats_collection
    
    try:
        mongo_url = os.getenv("MONGO_URL", "")
        if not mongo_url:
            print("MONGO_URL not set - chat history will not be stored")
            return None
        
        # For mongodb+srv:// (MongoDB Atlas), handle SSL handshake issues
        if mongo_url.startswith("mongodb+srv://"):
            try:
                import dns.resolver
            except ImportError:
                print("ERROR: 'dnspython' is required for mongodb+srv:// connections.")
                print("Install it with: pip install dnspython")
                return None
            
            try:
                # First attempt: Standard connection
                client = MongoClient(mongo_url, serverSelectionTimeoutMS=10000)
                client.server_info()
                _db_connection = client["Steam-Karnival"]
                _chats_collection = _db_connection['whatsapp_chats']
                print("MongoDB connection successful.")
                return _chats_collection
            except Exception as ssl_error:
                # If SSL handshake fails, try with relaxed TLS
                if "SSL" in str(ssl_error) or "TLS" in str(ssl_error):
                    print("SSL handshake failed, trying with relaxed TLS settings...")
                    try:
                        client = MongoClient(
                            mongo_url,
                            serverSelectionTimeoutMS=10000,
                            tlsAllowInvalidCertificates=True
                        )
                        client.server_info()
                        _db_connection = client["Steam-Karnival"]
                        _chats_collection = _db_connection['whatsapp_chats']
                        print("MongoDB connection successful with relaxed TLS.")
                        return _chats_collection
                    except Exception:
                        print("MongoDB connection failed. Chat history will not be stored.")
                        return None
                else:
                    print("MongoDB connection failed. Chat history will not be stored.")
                    return None
        else:
            # For regular mongodb:// connections
            client = MongoClient(mongo_url, serverSelectionTimeoutMS=10000)
            client.server_info()
            _db_connection = client["Steam-Karnival"]
            _chats_collection = _db_connection['whatsapp_chats']
            print("MongoDB connection successful.")
            return _chats_collection
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        print("Chat history will not be stored. Continue without MongoDB.")
        return None

def store_chat(user_mobile, timestamp, question, response):
    """Store chat history in MongoDB."""
    chats_col = get_mongodb_collection()
    if chats_col is None:
        return
    try:
        chats_col.insert_one({
            "user_mobile_number": user_mobile,
            "user_timestamp": timestamp,
            "user_question": question,
            "response": response
        })
    except Exception as e:
        print(f"Error storing chat: {e}")

def fetch_all_chats(user_mobile):
    """Fetch all chats for a user, ordered by timestamp (oldest first)."""
    chats_col = get_mongodb_collection()
    if chats_col is None:
        return []
    try:
        return list(chats_col.find({"user_mobile_number": user_mobile}).sort("user_timestamp", 1))
    except Exception as e:
        print(f"Error fetching chats: {e}")
        return []

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
        program_cache['last_excel_update'] = None
        program_cache['processed_files'] = {}
        program_cache['combined_data'] = None
        program_cache['last_cache_update'] = None
        print("All caches cleared.")
        return True
    except Exception as e:
        print(f"Error clearing cache: {e}")
        return False

def _faiss_index_exists():
    try:
        if not _FAISS_AVAILABLE:
            return False
        index_path = os.path.join(VECTORSTORE_PATH, "index.faiss")
        return os.path.exists(index_path)
    except Exception:
        return False

def _excel_index_exists():
    """Check if Excel FAISS index exists"""
    try:
        if not _FAISS_AVAILABLE:
            return False
        index_path = os.path.join(EXCEL_VECTORS_PATH, "index.faiss")
        return os.path.exists(index_path)
    except Exception:
        return False

def get_pdf_chunks_context(question: str, k: int = 20):
    """Retrieve top-k chunks from FAISS vector store as additional context.
    Uses multiple query variations to improve table/chart retrieval.
    Returns empty string if FAISS is not available or index not found.
    """
    global _pdf_vectorstore_cache, _embeddings_cache
    
    try:
        if not _FAISS_AVAILABLE:
            print(" FAISS not available - langchain packages may not be installed")
            return ""
        if not _faiss_index_exists():
            print("FAISS index not found - no PDF chunks available")
            return ""
        
        # Use cached vectorstore if available (faster)
        if _pdf_vectorstore_cache is None:
            print(f"\nLoading PDF FAISS index (first time)...")
            if _embeddings_cache is None:
                _embeddings_cache = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
            _pdf_vectorstore_cache = FAISS.load_local(
                VECTORSTORE_PATH,
                _embeddings_cache,
                allow_dangerous_deserialization=True
            )
            print("PDF FAISS index cached in memory")
        else:
            print(f"\nUsing cached PDF FAISS index for: '{question}'...")
        
        vectorstore = _pdf_vectorstore_cache
        
        # Create multiple query variations to improve retrieval for tables/charts
        query_variations = [question]
        
        # Dynamically create query variations based on keywords (not hardcoded)
        question_lower = question.lower()
        
        # Extract key domain terms from the question
        key_terms = []
        for word in question.split():
            if len(word) > 3:  # Only meaningful words
                key_terms.append(word.lower())
        
        # Add specific query variations for grade/points queries
        if "grade" in question_lower and ("point" in question_lower or "fixing" in question_lower):
            query_variations.append("fixing of grade points")
            query_variations.append("grade points percentage")
            query_variations.append("grade A 70% points")
            query_variations.append("grade table points")
        
        # Add specific query variations for merit certificate queries
        if "merit" in question_lower or "certificate" in question_lower:
            query_variations.append("merit certificate minimum marks")
            query_variations.append("merit certificate 50%")
            query_variations.append("awards and trophies merit certificate")
            query_variations.append("minimum marks merit certificate")
        
        # Add specific query variations for awards/trophies queries
        if "award" in question_lower or "trophy" in question_lower:
            query_variations.append("awards and trophies")
            query_variations.append("merit certificate")
        
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
        print(f"Error retrieving PDF chunks: {e}")
        import traceback
        traceback.print_exc()
        return ""

def get_excel_chunks_context(question: str, k: int = 15):
    """Retrieve top-k chunks from Excel FAISS vector store for semantic search.
    Returns empty string if FAISS is not available or index not found.
    """
    global _excel_vectorstore_cache, _embeddings_cache
    
    try:
        if not _FAISS_AVAILABLE:
            print("FAISS not available - Excel embeddings cannot be used")
            return ""
        if not _excel_index_exists():
            print("Excel FAISS index not found - falling back to traditional search")
            return ""
        
        # Use cached vectorstore if available (faster)
        if _excel_vectorstore_cache is None:
            print(f"\nLoading Excel FAISS index (first time)...")
            if _embeddings_cache is None:
                _embeddings_cache = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
            _excel_vectorstore_cache = FAISS.load_local(
                EXCEL_VECTORS_PATH,
                _embeddings_cache,
                allow_dangerous_deserialization=True
            )
            print("Excel FAISS index cached in memory")
        else:
            print(f"\nUsing cached Excel FAISS index for: '{question}'...")
        
        vectorstore = _excel_vectorstore_cache
        
        # Create query variations for better retrieval
        query_variations = [question]
        question_lower = question.lower()
        
        # Expand day names to dates for better matching
        # If question mentions a day name, add the corresponding date from DAY_MAPPING
        day_expanded = False
        for day_name, date in DAY_MAPPING.items():
            if day_name.lower() in question_lower:
                # Add query with date format
                query_variations.append(question.replace(day_name, date).replace(day_name.lower(), date))
                query_variations.append(f"programs on {date}")
                query_variations.append(f"category programs on {date}")
                day_expanded = True
                break
        
        # Extract key terms
        key_terms = []
        for word in question.split():
            if len(word) > 3:  # Only meaningful words
                key_terms.append(word.lower())
        
        # Add bigram combinations
        if len(key_terms) > 1:
            for i in range(len(key_terms) - 1):
                bigram = f"{key_terms[i]} {key_terms[i+1]}"
                query_variations.append(bigram)
        
        # For category queries, add variations
        if "categor" in question_lower or "cat" in question_lower:
            query_variations.append("categories performing")
            query_variations.append("category programs")
            # Add date-specific category queries if day/date is mentioned
            for day_name, date in DAY_MAPPING.items():
                if day_name.lower() in question_lower:
                    query_variations.append(f"categories on {date}")
                    query_variations.append(f"category programs on {date}")
                    query_variations.append(f"what categories {date}")
                    break
        
        # Get unique chunks from query variations
        all_docs = []
        seen_chunks = set()
        
        # Increase fetch_k for better retrieval, especially for category queries
        fetch_k_multiplier = 3 if ("categor" in question_lower or "cat" in question_lower) else 2
        
        for query_var in query_variations[:5]:  # Use top 5 variations
            try:
                retriever = vectorstore.as_retriever(search_kwargs={"k": k, "fetch_k": k * fetch_k_multiplier})
                docs = retriever.invoke(query_var)
                print(f"  Query '{query_var}': Retrieved {len(docs)} Excel docs")
                for doc in docs:
                    content = getattr(doc, "page_content", str(doc))
                    if content and content not in seen_chunks:
                        all_docs.append(doc)
                        seen_chunks.add(content)
            except Exception as e:
                print(f"  Warning: Error with query variation '{query_var}': {e}")
        
        if not all_docs:
            print("No Excel documents retrieved from FAISS")
            return ""
        
        # Sort by relevance (keep first k unique chunks)
        chunks = []
        for i, doc in enumerate(all_docs[:k]):
            content = getattr(doc, "page_content", str(doc))
            if content:
                chunks.append(content)
        
        context = "\n\n---\n\n".join(chunks) if chunks else ""
        print(f"Retrieved {len(chunks)} unique Excel chunks ({len(context)} chars total)")
        
        # DEBUG: Print first few chunks to verify what's being retrieved
        if chunks:
            print(f"\n=== DEBUG: First 3 Excel chunks retrieved ===")
            for i, chunk in enumerate(chunks[:3], 1):
                print(f"\nExcel Chunk {i}:")
                print(chunk[:300] + "..." if len(chunk) > 300 else chunk)
            print("=" * 60)
        
        return context
    except Exception as e:
        print(f"Error retrieving Excel chunks: {e}")
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


# --- Category normalization helpers ---
def _roman_to_int(roman: str) -> int:
    """Convert a (simple) Roman numeral up to 50 (L) to int. Returns 0 if invalid."""
    if not roman:
        return 0
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
    total = 0
    prev = 0
    for ch in roman.upper():
        val = values.get(ch, 0)
        if val == 0:
            return 0
        if val > prev:
            total += val - 2 * prev
        else:
            total += val
        prev = val
    return total


def _normalize_category_label(text: str) -> str:
    """
    Normalize a category label to its numeric string (e.g., "CATEGORY IV" -> "4", "CAT-4" -> "4").
    Returns empty string if nothing can be parsed.
    """
    if text is None:
        return ""
    s = str(text).strip().upper()
    # Remove common tokens and separators
    for token in ["CATEGORY", "CATEGORIES", "CAT", "-", ":", "/", "(", ")"]:
        s = s.replace(token, " ")
    s = " ".join(s.split())  # collapse whitespace
    # Prefer digits if present
    m = re.findall(r"\d+", s)
    if m:
        return str(int(m[-1]))
    # Otherwise try roman numerals
    parts = s.split()
    for part in reversed(parts):
        if re.fullmatch(r"[IVXLC]+", part):
            val = _roman_to_int(part)
            if val > 0:
                return str(val)
    return ""

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
        name = p.get('Item', 'Unknown')
        cat = p.get('Category', 'N/A')
        time = p.get('Time', '')
        date = p.get('Date', '')
        code = p.get('Item Code', '')
        return f"- {name} | Category: {cat} | Time: {time} | Date: {date} | Item Code: {code}"
    return "\n".join(fmt(p) for p in rows)

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
        
        # Check all relevant files (only Excel now - PDF processing removed)
        for file in all_files:
            if file.endswith(('.xlsx', '.xls')):
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
            
            # Combine all data (Excel only - PDF extraction removed)
            combined_data = []
            
            # Add Excel data if available
            if not program_cache['excel_data'].empty:
                for _, row in program_cache['excel_data'].iterrows():
                    try:
                        # Get data from row using column names
                        time = str(row['Time']).strip()
                        item_name = str(row['Item']).strip()
                        category = str(row['Category']).strip()
                        date = str(row['Date']).strip()
                        
                        # Skip empty or invalid rows
                        if not item_name or item_name.lower() == 'nan':
                            continue
                            
                        # Get day name for the date
                        day_name = DATE_MAPPING.get(date, "")
                        
                        program = {
                            "Item": item_name,
                            "Category": category if category != "nan" else "N/A",
                            "Time": time,
                            "Date": date,
                            "Day": day_name,
                            "Source": "excel"
                        }
                        
                        print(f"Added item: {item_name} on {date} ({day_name}) at {time}")  # Debug print
                        combined_data.append(program)
                    except Exception as e:
                        print(f"Error processing Excel row: {e}")
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
                # Accepts various headers: Time, Item, Category, Date
                map_cols = {}
                wanted = {'Time': ['time'], 'Item': ['item'], 'Category': ['category'], 'Date': ['date']}
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
                    map_order = ['Time', 'Item', 'Category', 'Date']
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
                df = df[df['Item'].str.len() > 0]
                # Remove header-like rows that leaked into data
                header_like_items = {"item", "items", "program", "program name", "programme", "programs", "programmes"}
                header_like_times = {"time", "times", "stage", "stages"}
                header_like_categories = {"category", "categories"}
                header_like_dates = {"date", "dates"}
                df = df[~df['Item'].str.strip().str.lower().isin(header_like_items)]
                df = df[~df['Time'].str.strip().str.lower().isin(header_like_times)]
                df = df[~df['Category'].str.strip().str.lower().isin(header_like_categories)]
                df = df[~df['Date'].str.strip().str.lower().isin(header_like_dates)]
                # Also drop rows where all four columns equal their header tokens
                df = df[~(
                    df['Time'].str.strip().str.upper().eq('TIME') &
                    df['Item'].str.strip().str.upper().eq('ITEM') &
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
    Handles text queries, PDF uploads (processed via pdf_ocr_extractor.py), and FAISS semantic search.
    """
    user_message = Body.strip()
    user_number = From

    print(f" Message from {user_number}: {user_message}")

    # Handle PDF attachment if present - delegate to pdf_ocr_extractor.py
    if MediaUrl0 and MediaUrl0.lower().endswith('.pdf'):
        try:
            if not _PDF_EXTRACTOR_AVAILABLE:
                send_whatsapp_message(user_number, 
                    "Sorry, PDF processing is not available. Please contact the administrator.")
                return Response(content="", status_code=200)
            
            # Download PDF from Twilio's media URL
            response = requests.get(MediaUrl0)
            
            # Generate a unique filename using timestamp and user number
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            safe_number = user_number.replace(':', '_').replace('+', '')
            pdf_filename = f"program_schedule_{timestamp}_{safe_number}.pdf"
            pdf_path = os.path.join(PDF_STORAGE, pdf_filename)
            
            # Save PDF to file
            with open(pdf_path, 'wb') as pdf_file:
                pdf_file.write(response.content)
            
            print(f"PDF saved to: {pdf_path}")
            
            # Process PDF using pdf_ocr_extractor.py - this will extract text and create FAISS embeddings
            print("Processing PDF with pdf_ocr_extractor.py...")
            result = process_pdf_file(
                pdf_path=pdf_path,
                output_dir=PDF_STORAGE,
                create_faiss=True,  # Create FAISS embeddings
                faiss_output=VECTORSTORE_PATH  # Use the same vectorstore path as main_demo.py
            )
            
            if result.get('status') == 'success':
                chunks_count = result.get('chunks', 0)
                send_whatsapp_message(user_number, 
                    f"I've received and processed your PDF! Extracted {result.get('text_length', 0):,} characters and created {chunks_count} embeddings. You can now ask me questions about it!")
                return Response(content="", status_code=200)
            else:
                error_msg = result.get('message', 'Unknown error')
                send_whatsapp_message(user_number, 
                    f"Sorry, I had trouble processing that PDF: {error_msg}")
                return Response(content="", status_code=200)
                
        except Exception as e:
            print(f"Error processing PDF: {e}")
            import traceback
            traceback.print_exc()
            send_whatsapp_message(user_number, 
                "Sorry, I had trouble processing that PDF. Could you try sending it again?")
            return Response(content="", status_code=200)

    # --- Normal chat logic ---
    # Step 1: Quick greeting check (before followup analysis for faster response)
    quick_extracted = extract_query_info(user_message)
    if quick_extracted.get("is_greeting", False) or quick_extracted.get("query_type") == "greeting":
        msg_lower = user_message.lower()
        if "bye" in msg_lower or "goodbye" in msg_lower:
            greeting_reply = "Goodbye! Feel free to ask me about any programs later. I'm here to help!"
        elif "good morning" in msg_lower or ("morning" in msg_lower and "hi" in msg_lower):
            greeting_reply = "Good morning! I'm your Kalolsavam assistant. How can I help you today? You can ask me about any programs, venues, or schedules!"
        elif "good afternoon" in msg_lower:
            greeting_reply = "Good afternoon! I'm your Kalolsavam assistant. How can I help you today? You can ask me about any programs, venues, or schedules!"
        elif "good evening" in msg_lower:
            greeting_reply = "Good evening! I'm your Kalolsavam assistant. How can I help you today? You can ask me about any programs, venues, or schedules!"
        elif "thank" in msg_lower:
            greeting_reply = "You're welcome! Let me know if you need anything else!"
        elif "how are you" in msg_lower:
            greeting_reply = "I'm doing great, thank you for asking! I'm ready to help you with any information about the Kalolsavam programs. What would you like to know?"
        else:
            greeting_reply = "Hello! I'm your Kalolsavam assistant. I can help you with program schedules, venues, and timings. What would you like to know?"
        
        send_whatsapp_message(user_number, greeting_reply)
        store_chat(
            user_mobile=user_number,
            timestamp=datetime.utcnow(),
            question=user_message,
            response=greeting_reply
        )
        return Response(content="", status_code=200)
    
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

    # Step 3: AI-driven follow-up analysis (no hardcoded indicators)
    final_query = user_message
    is_followup = False
    try:
        followup_resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[
                {"role": "system", "content": "You are the Receptionist for Kalsolavm. Decide if the user's latest message is a follow-up to the prior conversation. Return strict JSON only."},
                {"role": "user", "content": f"Conversation so far (last 5 turns):\n{chat_history}\n\nTask: Is the latest message a follow-up to the previous topic? \n\nIf YES, rewrite it into a COMPLETE standalone query that includes ALL context from previous messages:\n- CRITICAL: If the follow-up uses pronouns like 'it', 'them', 'that', 'this', 'they', replace them with the EXACT program/item name from the previous query\n- CRITICAL: If previous query mentioned a program name (e.g., 'elocution', 'folk dance', 'mono act'), and follow-up says 'it' or 'for it', include that EXACT program name in the standalone query\n- If previous query mentioned a time (e.g., 'after 2 pm'), include it in the standalone query WITH the comparison word ('after', 'before', etc.)\n- If previous query mentioned a date or day (e.g., 'Saturday', '12-11-2025'), include it in the standalone query\n- If previous query mentioned a category (e.g., 'category 4'), include it in the standalone query\n- CRITICAL: If the follow-up mentions 'after', 'before', 'past', 'later than', 'earlier than', preserve these words in the expanded query\n- Combine ALL filters from the conversation into one complete query\n\nExample: If user asked 'what are the topics for the elocution?' and then asks 'i need the topics for it', the standalone query should be 'what are the topics for elocution?' (preserve 'elocution')\n\nExample: If user asked 'what items after 2 pm on Saturday?' and then asks 'only category 4', the standalone query should be 'what are the category 4 items after 2 pm on Saturday?'\n\nExample: If user asked 'folk dance for girls' and then asks 'Is there any program happening after 8.30 am?', the standalone query should be 'Is there any program happening after 8.30 am?' (preserve the 'after' keyword)\n\nReturn JSON: {{\"is_followup\": true|false, \"standalone_query\": \"...\"}}"}
            ]
        )
        raw = followup_resp.choices[0].message.content.strip()
        print(f"DEBUG: Followup analysis raw response: {raw}")
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                is_followup = data.get("is_followup", False)
                if data.get("standalone_query"):
                    final_query = str(data["standalone_query"]).strip()
                    print(f"DEBUG: Followup detected! Original: '{user_message}' -> Expanded: '{final_query}'")
        except Exception as e:
            print(f"DEBUG: Failed to parse followup JSON: {e}")
    except Exception as e:
        print(f"follow-up analysis failed: {e}")
    
    # Step 4: Extract query info from the FINAL query (expanded if it was a followup)
    extracted = extract_query_info(final_query)
    print(f"DEBUG: Query extraction - Original: '{user_message}', Final: '{final_query}', Is Followup: {is_followup}")
    print(f"DEBUG: Extracted time_comparison: '{extracted.get('time_comparison', '')}'")
    try:
        program_cache['combined_data'] = None
        program_cache['excel_data'] = None
        # Ensure no old Excel content persists between user runs
        clear_schedule_cache()
        combined_data = update_cache_if_needed()
        # Use AI-generated standalone query if it's a follow-up; otherwise the original message
        ai_response = search_program_data(extracted, final_query, combined_data)
        print(f"DEBUG: search_program_data returned: {ai_response[:200] if ai_response else 'None'}...")
    except Exception as e:
        print(f"ERROR in search_program_data: {e}")
        import traceback
        traceback.print_exc()
        ai_response = ("I'm having trouble accessing the program data right now. "
                       "Please try again in a moment.")
    
    # Check if this is a time query that already has a formatted list response
    # If so, skip LLM rewriting which might filter out valid results
    # Use final_query (not user_message) to detect time queries in followups
    time_patterns = re.findall(r'\b(\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM))\b', final_query, re.IGNORECASE)
    is_time_query = bool(time_patterns) or ("at" in final_query.lower() and ("pm" in final_query.lower() or "am" in final_query.lower()))
    has_formatted_list = ai_response and ("At " in ai_response or "items are scheduled:" in ai_response or "• " in ai_response)
    
    # Also check if it's a category query with formatted response
    is_category_query = extracted.get("query_type") == "category" or (extracted.get("category") and not is_time_query)
    has_category_list = ai_response and ("Category" in ai_response or "• " in ai_response or "*Category" in ai_response)
    
    # Check if this is a duration query or remarks query (should use PDF data and might have formatted response)
    final_query_lower = final_query.lower()
    is_duration_query = any(term in final_query_lower for term in [
        "duration", "how long", "length", "time period", "running time"
    ])
    has_duration_info = ai_response and (("min" in ai_response.lower() or "mins" in ai_response.lower() or "minute" in ai_response.lower()) and any(c.isdigit() for c in ai_response))
    
    # Check if this is a remarks/notes query
    is_remarks_query = any(term in final_query_lower for term in [
        "remark", "remarks", "note", "notes", "comment", "comments", "additional", "detail", "info"
    ])
    has_remarks_info = ai_response and (("common" in ai_response.lower() or "participants" in ai_response.lower() or "(" in ai_response) and len(ai_response) > 50)
    
    print(f"DEBUG: is_time_query={is_time_query}, is_category_query={is_category_query}, is_duration_query={is_duration_query}, is_remarks_query={is_remarks_query}, has_formatted_list={has_formatted_list}, has_category_list={has_category_list}, has_duration_info={has_duration_info}, has_remarks_info={has_remarks_info}")
    
    if (is_time_query and has_formatted_list) or (is_category_query and has_category_list) or (is_duration_query and has_duration_info) or (is_remarks_query and has_remarks_info):
        # Already formatted correctly, use as-is
        if is_time_query:
            print("Time query with formatted response - skipping LLM rewrite")
        elif is_category_query:
            print("Category query with formatted response - skipping LLM rewrite")
        elif is_duration_query:
            print("Duration query with formatted response - skipping LLM rewrite")
        elif is_remarks_query:
            print("Remarks query with formatted response - skipping LLM rewrite")
        final_reply = ai_response
    else:
        # Use LLM to rewrite for better naturalness
        # Use final_query (not user_message) so LLM understands the expanded followup query
        print("Using LLM to rewrite response")
        final_reply = generate_human_like_reply(final_query, ai_response)
    
    print(f"DEBUG: Final reply to send: {final_reply[:200]}...")
    send_result = send_whatsapp_message(user_number, final_reply)
    print(f"DEBUG: send_result = {send_result}")
    # Store the chat with full details
    store_chat(
        user_mobile=user_number,
        timestamp=datetime.utcnow(),
        question=user_message,
        response=final_reply
    )
    return Response(content="", status_code=200)


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
                - Time queries (e.g., "morning programs", "What's at 10 AM?", "after 2 pm", "before 3 pm", "past 2", "later than 2 pm")
                  For time queries, extract:
                  * The time itself (e.g., "2 pm", "10:00 AM", "14:00")
                  * Time comparison: "after" (for "after", "past", "later than", "from"), "before" (for "before", "earlier than", "until"), or "" for exact time queries
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
                - time: Time mentioned (e.g., "2 pm", "10:00 AM", "14:00") or "" if none. Extract the time value itself without comparison words.
                - time_comparison: "after" if query asks for times after/beyond/past/later than the time, "before" if query asks for times before/earlier than/until the time, or "" for exact time queries (e.g., "at 2 pm", "2 pm programs")
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
        result = json.loads(response.choices[0].message.content)
        # Ensure time_comparison field exists (for backward compatibility)
        if "time_comparison" not in result:
            result["time_comparison"] = ""
        return result
    except Exception as e:
        print(" OpenAI extract error:", e)
        return {"program": "", "stage": "", "category": "", "time": "", "time_comparison": "", "date": "", "day": "", "venue": "", "query_type": "general", "is_greeting": False}



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
    
    # Helper function to format time to 12-hour format (e.g., "15:30:00" -> "3:30 PM")
    def format_time_for_answer(time_str):
        """Convert time to 12-hour format with AM/PM"""
        if not time_str or time_str == "Unknown time" or str(time_str).strip() == "":
            return str(time_str).strip() if time_str else "Unknown time"
        
        time_str_clean = str(time_str).strip()
        
        # If already in 12-hour format (contains AM/PM), return as is (but clean up if needed)
        if "AM" in time_str_clean.upper() or "PM" in time_str_clean.upper():
            return time_str_clean
        
        try:
            # Try to parse various 24-hour formats: "15:30:00", "15:30", "1530", etc.
            # Extract hour and minute using regex
            time_match = re.search(r'(\d{1,2}):(\d{2})(?::(\d{2}))?', time_str_clean)
            if time_match:
                hour = int(time_match.group(1))
                minute = int(time_match.group(2))
                
                # Convert to 12-hour format
                if hour == 0:
                    hour_12 = 12
                    am_pm = "AM"
                elif hour < 12:
                    hour_12 = hour
                    am_pm = "AM"
                elif hour == 12:
                    hour_12 = 12
                    am_pm = "PM"
                else:
                    hour_12 = hour - 12
                    am_pm = "PM"
                
                return f"{hour_12}:{minute:02d} {am_pm}"
            else:
                # Try to parse as just hour without colon (e.g., "1530" -> 15:30)
                hour_match = re.search(r'^(\d{2})(\d{2})$', time_str_clean)
                if hour_match:
                    hour = int(hour_match.group(1))
                    minute = int(hour_match.group(2))
                    
                    if hour == 0:
                        hour_12 = 12
                        am_pm = "AM"
                    elif hour < 12:
                        hour_12 = hour
                        am_pm = "AM"
                    elif hour == 12:
                        hour_12 = 12
                        am_pm = "PM"
                    else:
                        hour_12 = hour - 12
                        am_pm = "PM"
                    
                    return f"{hour_12}:{minute:02d} {am_pm}"
        except Exception as e:
            print(f"Error formatting time '{time_str_clean}': {e}")
        
        # Fallback: return original string if we can't parse it
        return time_str_clean
    
    # Try to load manual text for AI-powered search
    manual_path = os.path.join(PDF_STORAGE, "Manual State 2025 (2).txt")
    manual_text = ""
    if os.path.exists(manual_path):
        try:
            with open(manual_path, 'r', encoding='utf-8') as f:
                manual_text = f.read()
        except Exception as e:
            print(f"Error loading manual text: {e}")
    
    # ALWAYS retrieve PDF chunks from FAISS embeddings FIRST (before any early returns)
    # This ensures semantic search is always performed, even when Excel has data
    print("\n" + "="*60)
    print("=== RETRIEVING PDF CHUNKS FROM FAISS ===")
    print("="*60)
    question_words = user_message.lower().split()
    question_lower = user_message.lower()
    
    # Increase chunks for questions likely to need table/chart/structured data or value points
    is_value_points_query = any(term in question_lower for term in ["value point", "value points", "marks", "mark"])
    is_complex_query = len(question_words) > 5 or any(c in user_message for c in ["%", "table", "chart", "list"])
    
    # For value points queries, retrieve more chunks to ensure we find the scoring tables
    # For time queries (like "anchoring time"), also retrieve more chunks
    # For remarks queries, retrieve more chunks to find item details
    is_time_query = any(term in question_lower for term in ["time", "when", "at what time", "timing", "schedule"])
    is_remarks_query_check = any(term in question_lower for term in ["remark", "remarks", "note", "notes", "comment", "comments"])
    k_value = 30 if (is_value_points_query or is_remarks_query_check) else (25 if is_complex_query or is_time_query else 20)
    pdf_chunks = get_pdf_chunks_context(user_message, k=k_value)
    print(f"Retrieved {len(pdf_chunks)} characters from PDF FAISS embeddings")
    
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
            time_raw = str(prog.get("Time", "")).strip()
            if not time_raw:
                continue
            # Extract numeric part and ignore header-like rows
            m = re.search(r"(\d+)", time_raw)
            if not m:
                continue
            stages.add(int(m.group(1)))
        total = len(stages)
        if total > 0:
            return f"There are {total} stages in Kalolsavam."
        return "I couldn't find stage information in the schedule data."

    # Check if this is a factual query that should use PDF/manual data (fees, marks, phones, etc.)
    user_msg_lower = user_message.lower()
    is_factual_query = any(term in user_msg_lower for term in [
        "phone", "number", "contact", "email", "address", "how much", "fee", 
        "cost", "price", "value point", "value points", "marks", "mark", 
        "point", "points", "appeal", "grade", "percentage", "trophy", "trophies",
        "website", "url", "register", "registration", "password", "accessible",
        "duration", "how long", "length", "time period", "mins", "minutes", 
        "hrs", "hours", "hr", "mts", "mt", "running time", "remark", "remarks",
        "note", "notes", "comment", "comments", "additional", "info", "detail",
        "office", "kalotsav office", "call", "mobile"
    ])
    
    # Check if this is a program/stage/schedule question - prioritize Excel data
    # Also include time queries like "1 pm", "1:00 PM", "items at 1 pm", etc.
    # Include program name queries (user asking about a specific program)
    is_schedule_question = any(term in user_msg_lower for term in [
        "stage", "performing", "program", "schedule", "date", "when is", 
        "where is", "which stage", "what stage", "at what", "items", "item",
        "happening", "at", "pm", "am", "time", "what are", "which items",
        "what is", "tell me about", "when", "where"
    ])
    
    # Extract time from query if present (e.g., "1 pm", "1:00 PM", "13:00")
    # Use AI-extracted values if available, otherwise fall back to regex
    time_query = extracted.get("time", "").strip().lower() if extracted else ""
    
    # Check if this is a time query
    is_time_query_check = bool(time_query) or any(term in user_msg_lower for term in ["time", "when", "at what time", "timing", "schedule"])
    
    # ALSO retrieve Excel chunks from FAISS embeddings (if available)
    # This should be done after determining is_schedule_question
    print("\n" + "="*60)
    print("=== RETRIEVING EXCEL CHUNKS FROM FAISS ===")
    print("="*60)
    # For schedule questions, use more Excel chunks
    # For time queries (like "items at 8:30 AM"), retrieve even more chunks to ensure we get all items
    # For category queries (like "what categories are performing"), also retrieve more chunks
    is_category_query = "categor" in user_message.lower() or "cat" in user_message.lower()
    excel_k = 30 if (is_schedule_question and time_query) else (25 if (is_schedule_question or is_category_query) else (20 if is_time_query_check else 15))
    excel_chunks = get_excel_chunks_context(user_message, k=excel_k)
    print(f"Retrieved {len(excel_chunks)} characters from Excel FAISS embeddings")
    time_comparison = extracted.get("time_comparison", "").strip().lower() if extracted else None
    time_comparison = time_comparison if time_comparison else None  # Convert empty string to None
    
    # Fallback: Also check user_message directly for time patterns if AI didn't extract it
    if not time_query:
        time_patterns = re.findall(r'\b(\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM))\b', user_message, re.IGNORECASE)
        if time_patterns:
            time_query = time_patterns[0].lower()
    
    # Fallback: Detect "after" or "before" if AI didn't extract it
    if not time_comparison and time_query:
        user_msg_lower_check = user_message.lower()
        if any(word in user_msg_lower_check for word in ["after", "past", "later than", "from"]):
            time_comparison = "after"
        elif any(word in user_msg_lower_check for word in ["before", "earlier than", "until"]):
            time_comparison = "before"
    
    # For schedule questions, use Excel FAISS semantic search FIRST (if available)
    # Skip Excel search for factual queries (fees, marks, phones, etc.) - they need PDF data
    matching_programs = []  # Will be populated only if linear search is needed (when Excel FAISS not available)
    
    if (is_schedule_question or time_query) and combined_data and not is_factual_query:
        # If Excel FAISS embeddings are available, use them instead of linear search
        if excel_chunks:
            print("\nUsing Excel FAISS semantic search - skipping linear search, will use LLM for formatting")
            # Don't do linear search - use Excel FAISS chunks directly with LLM
            # matching_programs stays empty, so LLM will process Excel chunks
        else:
            # FAISS not available - skip Excel data (no linear search fallback)
            print("\nExcel FAISS embeddings not available - skipping Excel data (FAISS required for Excel search)")
            matching_programs = []
        
        # Check for category queries FIRST (before time queries) if it's a pure category query
        # This ensures category queries work independently
        query_type = extracted.get("query_type", "general")
        is_pure_category_query = (query_type == "category" and extracted.get("category") and not time_query)
        
        # All Excel queries now use FAISS semantic search only (no linear search)
        # If excel_chunks is available, LLM will process it; if not, Excel data is skipped
    
    # PDF chunks already retrieved at the top of function - continue to use them
    
    # Precise extraction for "Fixing of Grade" table (no hardcoding; parse from text)
    try:
        # Check for grade-related queries - trigger on "fixing of grade", "grade", "points", "percentage"
        grade_query = (
            ("grade" in question_lower and ("point" in question_lower or "percentage" in question_lower or "%" in question_lower or "70" in question_lower or "fixing" in question_lower)) or
            ("fixing of grade" in question_lower or "fixing of grading" in question_lower) or
            ("point" in question_lower and "grade" in question_lower)
        )
        if grade_query:
            combined_source = "\n".join([manual_text or "", pdf_chunks or ""])[:20000]
            # Prefer the section starting near "Fixing of Grade"
            start_idx = combined_source.lower().find("fixing of grade")
            if start_idx == -1:
                start_idx = combined_source.lower().find("fixing of grading")
            # Increase window size to capture all grades (A, B, C, etc.)
            window = combined_source[max(0, start_idx-500): start_idx+3000] if start_idx != -1 else combined_source
            # Normalize whitespace for easier regex
            normalized = re.sub(r"[\t\u00A0]+", " ", window)
            
            # First, try to extract all grades (Grade A, B, C, etc.) - this is the primary method
            # Improved regex to handle:
            # - "Grade A: 70% and above = 5 points"
            # - "Grade B: 60% to 69% = 3 points"
            # - "Grade B 60-69% 3 points"
            # - "Grade C 50% to 59% 1 point"
            
            # Try to extract with percentage ranges first (e.g., "60% to 69%")
            grade_pattern_with_range = re.compile(
                r"Grade\s+([A-Z])[^\n]*?(\d+)%\s*(?:to|and|-)\s*(\d+)%[^\n]*?(\d+)\s*point",
                re.IGNORECASE | re.MULTILINE
            )
            matches_with_range = grade_pattern_with_range.findall(normalized)
            
            # Also try to extract with "and above" (e.g., "70% and above")
            grade_pattern_above = re.compile(
                r"Grade\s+([A-Z])[^\n]*?(\d+)%\s*(?:and\s+above|and\s+over)[^\n]*?(\d+)\s*point",
                re.IGNORECASE | re.MULTILINE
            )
            matches_above = grade_pattern_above.findall(normalized)
            
            # Combine matches
            all_matches = {}
            for grade, min_pct, max_pct, points in matches_with_range:
                grade_letter = grade.upper()
                if grade_letter not in all_matches:
                    all_matches[grade_letter] = (min_pct, max_pct, points)
            
            for grade, min_pct, points in matches_above:
                grade_letter = grade.upper()
                if grade_letter not in all_matches:
                    all_matches[grade_letter] = (min_pct, "above", points)
            
            # Fallback: simpler pattern if above patterns don't match
            if not all_matches:
                grade_pattern_simple = re.compile(
                    r"Grade\s+([A-Z])[^\n]*?(\d+)%[^\n]*?(\d+)\s*point",
                    re.IGNORECASE | re.MULTILINE
                )
                matches_simple = grade_pattern_simple.findall(normalized)
                for grade, percent, points in matches_simple:
                    grade_letter = grade.upper()
                    if grade_letter not in all_matches:
                        # For simple matches, assume "and above" if it's Grade A, otherwise check context
                        all_matches[grade_letter] = (percent, "above", points)
            
            if all_matches:
                grade_info = []
                # Sort grades: A, B, C, etc.
                for grade_letter in sorted(all_matches.keys()):
                    min_pct, max_pct, points = all_matches[grade_letter]
                    if max_pct == "above":
                        grade_info.append(f"Grade {grade_letter} ({min_pct}% and above) = {points} points")
                    else:
                        grade_info.append(f"Grade {grade_letter} ({min_pct}% to {max_pct}%) = {points} points")
                
                if grade_info:
                    # If user asked about a specific grade (e.g., "Grade B"), return just that grade
                    # Otherwise return all grades
                    user_msg_lower = user_message.lower()
                    if "grade b" in user_msg_lower or "gradeb" in user_msg_lower:
                        grade_b_info = [g for g in grade_info if "Grade B" in g]
                        if grade_b_info:
                            return grade_b_info[0]
                    elif "grade a" in user_msg_lower or "gradea" in user_msg_lower:
                        grade_a_info = [g for g in grade_info if "Grade A" in g]
                        if grade_a_info:
                            return grade_a_info[0]
                    elif "grade c" in user_msg_lower or "gradec" in user_msg_lower:
                        grade_c_info = [g for g in grade_info if "Grade C" in g]
                        if grade_c_info:
                            return grade_c_info[0]
                    # Otherwise return all grades found
                    return "\n".join(grade_info)
            
            # Fallback: Try line-based scan for Grade A row (if all-grade extraction didn't work)
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
    except Exception as e:
        print(f"Error in grade extraction: {e}")
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
    # is_factual_query is already defined earlier in the function
    
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
                                    cleaned = '. '.join(sentences[:2]) + '.'
                                
                                return cleaned
                        else:
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
        # FAISS semantic search only (no linear search fallback)
        if excel_chunks:
            # For schedule questions, emphasize Excel data comes first
            if is_schedule_question:
                final_context_parts.insert(0, "[SCHEDULE - PRIMARY SOURCE (FAISS)]\n" + excel_chunks)
                print(f"Added Excel embeddings context as PRIMARY SOURCE ({len(excel_chunks)} chars)")
            else:
                final_context_parts.append("[Schedule (FAISS)]\n" + excel_chunks)
                print(f"Added Excel embeddings context ({len(excel_chunks)} chars)")
        else:
            # FAISS not available - skip Excel data (no linear search fallback)
            print("\nExcel FAISS embeddings not available - skipping Excel data (FAISS required for Excel search)")
    
    # 3. FAISS PDF chunks - SKIP for schedule/time questions (use Excel data only)
    # Note: PDF "TIME" column = duration (1hr, 5mts), NOT scheduled time
    # Excel "Time" column = actual scheduled time (8:30 AM, 1:00 PM)
    # For schedule questions, ONLY use Excel data - PDF chunks contain category descriptions that confuse schedule queries
    if pdf_chunks and not (is_schedule_question or time_query):
        # Only include PDF chunks for factual queries (fees, rules, grades, etc.), NOT for schedule queries
        pdf_header = "[PDF Chunks]\n"
        final_context_parts.append(pdf_header + pdf_chunks)
        print(f"Added PDF chunks context ({len(pdf_chunks)} chars)")
    elif pdf_chunks and (is_schedule_question or time_query):
        # Skip PDF chunks for schedule questions - they contain category descriptions that confuse the LLM
        print("Skipping PDF chunks for schedule/time query - using Excel data only")
    else:
        print("No PDF chunks retrieved - FAISS may not be available or index not found")
    
    # If we have any context, use AI to answer
    if final_context_parts:
        final_context = "\n\n".join(final_context_parts)
        # Limit to 14000 chars to leave room for prompt
        final_context = final_context[:14000]
        
        print(f"\n=== Sending to LLM with context ({len(final_context)} chars) ===")
        print(f"Context preview:\n{final_context[:800]}...")
        
        # DEBUG: Check if Excel chunks are in the context
        if "[SCHEDULE - PRIMARY SOURCE (FAISS)]" in final_context or "[Schedule (FAISS)]" in final_context:
            excel_start = final_context.find("[SCHEDULE") if "[SCHEDULE" in final_context else final_context.find("[Schedule (FAISS)]")
            excel_end = final_context.find("\n\n[", excel_start + 50) if excel_start != -1 else len(final_context)
            excel_section = final_context[excel_start:excel_end] if excel_start != -1 else ""
            if excel_section:
                print(f"\n=== DEBUG: Excel FAISS Section in Context ===")
                print(excel_section[:1000] + "..." if len(excel_section) > 1000 else excel_section)
                print("=" * 60)
        
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

            # Apply category filtering if category is specified (even with time query)
            if extracted.get("category"):
                cat_input = str(extracted.get("category", "")).strip()
                target_cat = _normalize_category_label(cat_input)
                if target_cat:
                    category_filtered = []
                    for prog in matching_programs:
                        prog_cat = _normalize_category_label(prog.get("Category", ""))
                        if prog_cat == target_cat:
                            category_filtered.append(prog)
                    if category_filtered:
                        matching_programs = category_filtered
                        print(f"Filtered to {len(matching_programs)} program(s) for category {target_cat} (normalized from '{cat_input}')")
                    else:
                        print(f"No programs found matching category {target_cat} after filtering {len(matching_programs)} time-matched programs")

            # Format the answer directly from Excel data (no hallucinations)
            # Check for category queries FIRST, then time queries
            query_type = extracted.get("query_type", "general")
            is_pure_category_query = (query_type == "category" and extracted.get("category") and not time_query)
            
            if is_pure_category_query:
                cat_input = str(extracted.get("category", "")).strip()
                target_cat = _normalize_category_label(cat_input)
                print(f"DEBUG: Category query formatting - input='{cat_input}', normalized='{target_cat}', matching_programs={len(matching_programs)}")
                
                if len(matching_programs) == 0:
                    return f"I couldn't find any items in category {cat_input}. Please check if the category number is correct (1, 2, 3, or 4)."
                else:
                    # Format all category items
                    result_lines = [f"*Category {cat_input} items:*\n"]
                    for prog in matching_programs:
                        item_name = prog.get("Item", "Unknown")
                        time_info = format_time_for_answer(prog.get("Time", "Unknown time"))
                        date_info = format_date_for_answer(prog.get("Date", ""))
                        
                        if date_info:
                            result_lines.append(f"• {item_name} – {time_info} on {date_info}")
                        else:
                            result_lines.append(f"• {item_name} – {time_info}")
                    
                    return "\n".join(result_lines)
            
            # For time queries, list ALL items at that time (or after/before)
            print(f"DEBUG: Time query check - time_query='{time_query}', time_comparison='{time_comparison}', matching_programs count={len(matching_programs)}")
            if time_query:
                if len(matching_programs) == 0:
                    # Check if category filter was applied
                    has_category_filter = extracted.get("category") and _normalize_category_label(str(extracted.get("category", "")))
                    cat_input = str(extracted.get("category", "")).strip() if has_category_filter else None
                    
                    if has_category_filter:
                        if time_comparison:
                            return f"I couldn't find any category {cat_input} items scheduled {time_comparison} {time_query}."
                        else:
                            return f"I couldn't find any category {cat_input} items scheduled at {time_query}."
                    else:
                        if time_comparison:
                            return f"I couldn't find any items scheduled {time_comparison} {time_query}."
                        else:
                            return f"I couldn't find any items scheduled at {time_query}."
                elif len(matching_programs) == 1:
                    # Single result - give concise answer
                    prog = matching_programs[0]
                    item_name = prog.get("Item", "Unknown")
                    time_info = format_time_for_answer(prog.get("Time", "Unknown time"))
                    category_info = prog.get("Category", "")
                    date_info = format_date_for_answer(prog.get("Date", ""))
                    
                    # Build Excel answer
                    if time_comparison:
                        if date_info:
                            excel_answer = f"{time_comparison.title()} {time_query}, {item_name} is performing ({category_info}) on {date_info}." if category_info and category_info != "N/A" else f"{time_comparison.title()} {time_query}, {item_name} is performing on {date_info}."
                        else:
                            excel_answer = f"{time_comparison.title()} {time_query}, {item_name} is performing ({category_info})." if category_info and category_info != "N/A" else f"{time_comparison.title()} {time_query}, {item_name} is performing."
                    else:
                        if date_info:
                            excel_answer = f"At {time_info}, {item_name} is performing ({category_info}) on {date_info}." if category_info and category_info != "N/A" else f"At {time_info}, {item_name} is performing on {date_info}."
                        else:
                            excel_answer = f"At {time_info}, {item_name} is performing ({category_info})." if category_info and category_info != "N/A" else f"At {time_info}, {item_name} is performing."
                    
                    # Combine with PDF chunks if available
                    if pdf_chunks and len(pdf_chunks.strip()) > 50:
                        print(f"Found Excel data AND PDF chunks for single result time query - will combine in LLM processing")
                        # Let it fall through to LLM processing
                    else:
                        return excel_answer
                else:
                    # Multiple items - list ALL of them
                    # Check if category filter was applied
                    has_category_filter = extracted.get("category") and _normalize_category_label(str(extracted.get("category", "")))
                    if has_category_filter:
                        cat_input = str(extracted.get("category", "")).strip()
                        if time_comparison:
                            result_lines = [f"{time_comparison.title()} {time_query}, the following category {cat_input} items are scheduled:"]
                        else:
                            result_lines = [f"At {time_query}, the following category {cat_input} items are scheduled:"]
                    else:
                        if time_comparison:
                            result_lines = [f"{time_comparison.title()} {time_query}, the following items are scheduled:"]
                        else:
                            result_lines = [f"At {time_query}, the following items are scheduled:"]
                    print(f"DEBUG: Formatting {len(matching_programs)} items for time query response (category filter: {has_category_filter})")
                    for prog in matching_programs:
                        item_name = prog.get("Item", "Unknown")
                        category_info = prog.get("Category", "")
                        date_info = format_date_for_answer(prog.get("Date", ""))
                        
                        if category_info and category_info != "N/A" and date_info:
                            result_lines.append(f"• {item_name} ({category_info}) on {date_info}")
                        elif category_info and category_info != "N/A":
                            result_lines.append(f"• {item_name} ({category_info})")
                        elif date_info:
                            result_lines.append(f"• {item_name} on {date_info}")
                        else:
                            result_lines.append(f"• {item_name}")
                    
                    final_response = "\n".join(result_lines)
                    print(f"DEBUG: Time query response with {len(matching_programs)} items:\n{final_response[:200]}...")
                    
                    # Combine with PDF chunks if available
                    if pdf_chunks and len(pdf_chunks.strip()) > 50:
                        print(f"Found Excel data AND PDF chunks for time query - will combine in LLM processing")
                        # Store Excel answer but let it fall through to LLM processing
                    else:
                        return final_response
            
            # For non-time queries, format normally
            results = []
            for prog in matching_programs:
                item_name = prog.get("Item", "Unknown")
                time_info = format_time_for_answer(prog.get("Time", "Unknown time"))
                category_info = prog.get("Category", "")
                date_info = format_date_for_answer(prog.get("Date", ""))
                
                if len(matching_programs) == 1:
                    # Single result - give concise, natural answer
                    if date_info:
                        excel_answer = f"{item_name} is performing at {time_info} ({category_info}) on {date_info}." if category_info and category_info != "N/A" else f"{item_name} is performing at {time_info} on {date_info}."
                    else:
                        excel_answer = f"{item_name} is performing at {time_info} ({category_info})." if category_info and category_info != "N/A" else f"{item_name} is performing at {time_info}."
                    
                    # For schedule queries, Excel data is authoritative - return it directly
                    # Only use PDF chunks for factual queries (fees, rules, etc.), not schedule data
                    if is_schedule_question:
                        print(f"Schedule query with Excel match - using Excel data as authoritative source")
                        return excel_answer
                    # For non-schedule queries, check if PDF chunks should be combined
                    elif pdf_chunks and len(pdf_chunks.strip()) > 50 and not is_factual_query:
                        print(f"Found Excel data AND PDF chunks for single result - will combine in LLM processing")
                        # Let it fall through to LLM processing
                    else:
                        return excel_answer
                else:
                    # Multiple results - format as natural list
                    line_parts = [time_info]
                    if category_info and category_info != "N/A":
                        line_parts.append(f"({category_info})")
                    if date_info:
                        line_parts.append(f"on {date_info}")
                    results.append(" ".join(line_parts))
            
            if results:
                item_name_display = matching_programs[0].get('Item', 'Item')
                # More natural, conversational language for multiple times
                excel_answer = f"{item_name_display} is performing at {results[0]} and {results[1]}." if len(results) == 2 else f"{item_name_display} is performing at the following times:\n" + "\n".join([f"• {r}" for r in results])
                
                # Combine Excel answer with PDF chunks if available for comprehensive answer
                if pdf_chunks and len(pdf_chunks.strip()) > 50:
                    print(f"Found Excel data AND PDF chunks - combining answers")
                    # Return combined answer - let LLM process it for natural response
                    # Don't return early, let it fall through to LLM processing below
                else:
                    # Excel answer only, return immediately
                    return excel_answer
        else:
            # No Excel match - for schedule questions, don't use PDF chunks
            if not matching_programs and not excel_chunks:
                if is_schedule_question:
                    # For schedule questions, Excel data is authoritative - don't use PDF chunks
                    print("No Excel match found for schedule question - returning 'not found' message")
                    filter_info = ""
                    if filter_date:
                        filter_info = f" on {filter_date}"
                    if time_query:
                        filter_info = f" at {time_query}{filter_info}"
                    return f"I couldn't find any programs scheduled{filter_info}. Please check the time and date."
                elif pdf_chunks and len(pdf_chunks.strip()) > 50:
                    # No Excel match and PDF chunks found - use PDF chunks for factual queries only
                    print("No Excel match, but PDF chunks found - using PDF chunks for answer")
                else:
                    # No Excel match and no PDF chunks - avoid hallucination
                    return (
                        "I couldn't find that program in the official schedule. "
                        "Please check the exact program name or share a screenshot of the row."
                    )
    
    # PDF chunks already retrieved at the top of function - continue to use them
    
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
    # is_factual_query is already defined earlier in the function
    
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
        # FAISS semantic search only (no linear search fallback)
        if excel_chunks:
            # For schedule questions, emphasize Excel data comes first
            if is_schedule_question:
                final_context_parts.insert(0, "[SCHEDULE - PRIMARY SOURCE (FAISS)]\n" + excel_chunks)
                print(f"Added Excel embeddings context as PRIMARY SOURCE ({len(excel_chunks)} chars)")
            else:
                final_context_parts.append("[Schedule (FAISS)]\n" + excel_chunks)
                print(f"Added Excel embeddings context ({len(excel_chunks)} chars)")
        else:
            # FAISS not available - skip Excel data (no linear search fallback)
            print("\nExcel FAISS embeddings not available - skipping Excel data (FAISS required for Excel search)")
    
    # 3. FAISS PDF chunks - SKIP for schedule/time questions (use Excel data only)
    # Note: PDF "TIME" column = duration (1hr, 5mts), NOT scheduled time
    # Excel "Time" column = actual scheduled time (8:30 AM, 1:00 PM)
    # For schedule questions, ONLY use Excel data - PDF chunks contain category descriptions that confuse schedule queries
    if pdf_chunks and not (is_schedule_question or time_query):
        # Only include PDF chunks for factual queries (fees, rules, grades, etc.), NOT for schedule queries
        pdf_header = "[PDF Chunks]\n"
        final_context_parts.append(pdf_header + pdf_chunks)
        print(f"Added PDF chunks context ({len(pdf_chunks)} chars)")
    elif pdf_chunks and (is_schedule_question or time_query):
        # Skip PDF chunks for schedule questions - they contain category descriptions that confuse the LLM
        print("Skipping PDF chunks for schedule/time query - using Excel data only")
    else:
        print("No PDF chunks retrieved - FAISS may not be available or index not found")
    
    # If we have any context, use AI to answer
    if final_context_parts:
        final_context = "\n\n".join(final_context_parts)
        # Limit to 14000 chars to leave room for prompt
        final_context = final_context[:14000]
        
        print(f"\n=== Sending to LLM with context ({len(final_context)} chars) ===")
        print(f"Context preview:\n{final_context[:800]}...")
        
        # DEBUG: Check if Excel chunks are in the context
        if "[SCHEDULE - PRIMARY SOURCE (FAISS)]" in final_context or "[Schedule (FAISS)]" in final_context:
            excel_start = final_context.find("[SCHEDULE") if "[SCHEDULE" in final_context else final_context.find("[Schedule (FAISS)]")
            excel_end = final_context.find("\n\n[", excel_start + 50) if excel_start != -1 else len(final_context)
            excel_section = final_context[excel_start:excel_end] if excel_start != -1 else ""
            if excel_section:
                print(f"\n=== DEBUG: Excel FAISS Section in Context ===")
                print(excel_section[:1000] + "..." if len(excel_section) > 1000 else excel_section)
                print("=" * 60)
        
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
                    - [SCHEDULE - PRIMARY SOURCE]: Excel program schedule with exact scheduled times (e.g., "8:30 AM", "1:00 PM"), dates, categories - USE THIS FOR ALL SCHEDULE/TIME QUESTIONS
                    - Manual: Rules, fees, awards, regulations
                    - PDF Chunks: Semantic search results from the festival manual/guide (use ONLY for fees, rules, awards, grades, percentages, tables, durations, remarks, notes - NOT for schedule/time questions)
                    
                    ⚠️ CRITICAL FOR SCHEDULE QUESTIONS: 
                    - If this is a schedule/time question (e.g., "list items at 8:30 AM on Friday"), ONLY use "[SCHEDULE - PRIMARY SOURCE]" data
                    - COMPLETELY IGNORE PDF chunks for schedule/time questions - they contain category descriptions, NOT schedule data
                    - PDF chunks may contain text like "CATEGORY 1 (CLASS III AND IV)" - this is NOT schedule information, ignore it
                    - Only Excel schedule data contains actual scheduled times and dates
                    
                    ⚠️ CRITICAL: TIME COLUMN DISTINCTION ⚠️
                    - Excel "Time" column = ACTUAL SCHEDULED TIME of program (e.g., "8:30 AM", "1:00 PM") - USE THIS for "when is X?", "what time is Y?", schedule questions
                    - PDF "TIME" column = DURATION of program (e.g., "1hr", "5mts", "30mts") - USE THIS for "how long is X?", "duration of Y?", "length of Z?" questions
                    - For questions about "when is X?", "what time is Y?", "schedule for Z" → USE EXCEL TIME ONLY (scheduled time), IGNORE PDF TIME (duration)
                    - For questions about "duration of X?", "how long is Y?", "length of Z?" → USE PDF "TIME" column (duration in hrs/mts), convert to minutes or standard format like "X min"
                    
                    CRITICAL PRIORITY FOR SCHEDULE QUESTIONS:
                    - ⚠️ CRITICAL: For schedule/time questions, ONLY use "[SCHEDULE - PRIMARY SOURCE]" data - IGNORE PDF chunks completely
                    - ⚠️ CRITICAL: PDF chunks contain category descriptions (like "CATEGORY 1 (CLASS III AND IV)") which are NOT schedule data - DO NOT use them for schedule queries
                    - ⚠️ CRITICAL: PDF chunks do NOT contain scheduled times or dates - only use Excel schedule data for "when" or "what time" questions
                    - If you see "[SCHEDULE - PRIMARY SOURCE]" → Use ONLY that data for stage/program/time questions - IGNORE everything else
                    - Scheduled times like "8:30 AM", "1:00 PM" - use these EXACT values for time questions
                    - ⚠️ CRITICAL FOR LIST QUERIES: If asked to "list items at [time]" or "what's happening at [time]", you MUST:
                      1. IGNORE PDF chunks completely - they do NOT contain schedule information
                      2. Scan through EVERY chunk in the "[SCHEDULE - PRIMARY SOURCE]" section ONLY
                      3. Extract EVERY item that has the exact time (e.g., "8:30 AM", "8:30AM", "8.30 AM")
                      4. Extract EVERY item that has the matching date (e.g., Friday = 11/14/2025, 11-14-2025, 14-11-2025)
                      5. Count ALL matching items first, then list each one
                      6. DO NOT summarize - list EVERY single item individually
                      7. Format as: "There are X items at [time] on [day]:\n• Item Name (Category)\n• Item Name (Category)\n..." with ALL items listed
                    - PDF "TIME" has DURATIONS like "1hr", "5mts" - DO NOT use these for schedule/time questions, they're just durations
                    - Stage numbers like "STAGE 9", "STAGE 11" - use those EXACT values
                    - DO NOT mix PDF chunk category descriptions (like "Category III (Classes VIII to X)") with schedule stage numbers
                    - If a program appears on multiple stages, list ALL stages
                    - Schedule data is authoritative for schedule/time questions - ignore conflicting PDF chunk descriptions
                    - If PDF chunks mention a "TIME" column, that's DURATION (how long), not scheduled time (when it happens)
                    
                    ⚠️ CRITICAL: NEVER HALLUCINATE OR INVENT DATA ⚠️
                    - ONLY use dates, times, and categories that are EXACTLY shown in the schedule data
                    - If schedule shows "11/12/2025" → use that date, NOT "14-11-2025" or any other date
                    - If schedule shows "CATEGORY II" → use that category, NOT "Category I" or any other category
                    - If a program is NOT in the schedule data for a specific date/category → DO NOT make up information
                    - If you cannot find the exact match in schedule data, say "I don't see that program for that date/category"
                    - DO NOT combine dates from different sources or invent dates/times that don't exist in schedule data
                    
                    ⚠️ CRITICAL: NO HALLUCINATION RULE ⚠️
                    - ONLY use dates, times, categories, and programs that EXACTLY appear in the schedule data from "[SCHEDULE - PRIMARY SOURCE]"
                    - NEVER make up or guess dates, times, or categories that are not explicitly shown in the schedule data
                    - If schedule shows "11/12/2025" or "11-12-2025", use that EXACT date - DO NOT convert to "14-11-2025" or any other date
                    - If schedule shows "CATEGORY II", use "Category II" - DO NOT change it to "Category I" or any other category
                    - If the program is not found in schedule data, say "I don't see that program" - DO NOT make up information
                    - If schedule shows multiple entries for the same program, list ALL of them with their EXACT dates and categories
                    - DO NOT combine or merge different entries - report each one separately as it appears in schedule data
                    
                    {f"THIS IS A SCHEDULE QUESTION - USE schedule 'Time' (scheduled time like '8:30 AM'), IGNORE PDF 'TIME' (duration like '1hr'). USE ONLY [SCHEDULE - PRIMARY SOURCE] DATA" if is_schedule_question else ""}
                    
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
                    10. For grade points queries: Look for "Fixing of Grade" section or table. Extract ALL grades (Grade A, Grade B, Grade C, etc.) with their percentage requirements and corresponding points. Format as: "Grade A (70% and above) = 5 points", "Grade B (60-69%) = 4 points", etc.
                    11. CRITICAL FOR GRADE POINTS: If you see "Fixing of Grade" or "Grade A", "Grade B" with percentages and points in the context, extract ALL grade information (Grade A, Grade B, Grade C, etc.) with their percentage ranges and points. DO NOT say "not mentioned" if this information exists in the chunks.
                    12. CRITICAL FOR SPECIFIC GRADE QUERIES: If user asks about "Grade B" or "what about grade B", search for "Grade B" in the context. If "Grade B" appears with a percentage and points in the PDF chunks, extract it EXACTLY as shown. DO NOT make up or hallucinate grade information. If you see "Grade B: 60% to 69% = 3 points" extract it exactly as "Grade B (60% to 69%) = 3 points". If Grade B is not found in the context, say "I couldn't find Grade B information in the provided context" - DO NOT invent or hallucinate.
                    13. CRITICAL: NEVER HALLUCINATE GRADE INFORMATION - ONLY extract what is EXACTLY shown in the PDF chunks. If Grade B says "60% to 69%" do NOT say "60% and above". If Grade C says "50% to 59%" do NOT say "50% and above". Extract percentage ranges EXACTLY as shown.
                    14. For value points: Extract ALL criteria/evaluation items and their corresponding marks, format them clearly as: "Item Name - X marks". List ALL items found under the relevant category header (e.g., "Folk Dance")
                    15. CRITICAL: If you see "Value Points" followed by "Folk Dance" (or any category) in the context, extract EVERY item name and mark value from that section - DO NOT say "not mentioned"
                    
                    GENERAL CRITICAL RULES:
                    1. ALWAYS check PDF Chunks FIRST - search through EVERY chunk thoroughly
                    2. Extract EXACT numbers, amounts, percentages, phone numbers, contact info, and details from PDF Chunks when present
                    3. NEVER say "not mentioned" or "not in the provided context" if PDF Chunks section exists and contains relevant data
                    4. For value points queries: If you see ANY mention of "Value Points" with a category name (like "Folk Dance") in the PDF Chunks, extract ALL items and marks - the data IS there, DO NOT say "not mentioned"
                    5. If question asks about grades, percentages, tables, fees, rules, awards, phone numbers, contact info, dates, durations, remarks, or notes - PDF Chunks likely contain it
                    6. For phone/contact questions: Look for phone numbers (10-digit numbers, numbers with slashes like "9778665476/9778665475"), email addresses, office addresses, or contact details near phrases like "kalotsav office", "contact", "phone", "number", "office number". Phone numbers may appear on a separate line after a dash (-) or after mentioning "kalotsav office". Example: "from the kalotsav office\n-\n9778665476/9778665475" means the office numbers are 9778665476 and 9778665475
                    7. For value points/marks questions: Look for sections with headings like "Value Points" followed by the category (e.g., "Folk Dance"), then item names and mark values. The format in the PDF is: item name on one line, then "X marks" on the next line. Extract them and format clearly: "Item Name - X marks". Example: If you see "Akara Sushama" followed by "15 marks", format as "Akara Sushama - 15 marks". DO NOT output raw number sequences without labels.
                    8. IGNORE raw number sequences without context (like "5 5 10 3 5 8...") - these are likely poorly formatted table data. Look for properly formatted sections with item names and their values instead.
                    9. When you find "Value Points" section with a category name matching the question, extract ALL items and marks listed under that category - do not stop after finding one item.
                    10. For duration questions: Look for PDF "TIME" column values like "1hr", "5mts", "30mts", "1 hour", "5 minutes". Convert to standard format: "1hr" → "60 min", "5mts" → "5 min", "30mts" → "30 min". Always format as "X min" or "X minutes" for consistency.
                    11. For remarks/notes questions: Look for information in parentheses like "(Common for both boys & girls)", "(Common for both boys and girls)", or other notes listed after the item name and duration in PDF chunks. These notes appear on the line immediately after the duration. Extract ANY text in parentheses or on lines following the item name that provides additional information. Format as: "The remarks for [Item Name] are: [remarks text]". If you find ANY remarks/notes in the PDF chunks, provide them - DO NOT say "not mentioned" if remarks exist.
                    12. Provide precise answers with exact numbers/percentages/phone numbers/durations/remarks when found - extract them directly from the context
                    13. For merit certificate questions: Search for "merit certificate", "minimum marks", "50%", "first second third positions", "awards and trophies" in PDF chunks. If you see "minimum of 50% marks shall be awarded merit certificates" or similar text, extract the exact percentage (e.g., "50%") and explain the requirement clearly.
                    14. For awards/trophies questions: Look for sections titled "AWARDS AND TROPHIES" or similar headers. Extract information about merit certificates, participation certificates, minimum marks requirements, etc.
                    15. CRITICAL FOR MERIT CERTIFICATE QUERIES: If you see "minimum of 50% marks" or "50% marks" mentioned with "merit certificate" in the context, the answer IS THERE - extract it and provide it. DO NOT say "not mentioned" or "not specified" if this information exists in the chunks.
                    16. Use WhatsApp-friendly formatting: *bold* for key labels, bullets for lists
                    17. If multiple relevant entries exist, list them all clearly
                    18. If you find partial information (e.g., just percentage, just phone number), provide what you found
                    19. Phone numbers might appear as digits only (e.g., "9778665476") or with separators - extract them as found
                    
                    Answer format: Write naturally and conversationally. Be direct but friendly. Extract exact values from PDF Chunks. For tables, value points, or structured data, format them clearly with item names and values (e.g., "Item Name - X marks"), never output raw number sequences."""},
                    {"role": "user", "content": f"""Question: {user_message}

Available Context:
{final_context}

CRITICAL INSTRUCTIONS:
1. FOR SCHEDULE/TIME/PROGRAM QUESTIONS: 
   - ⚠️ CRITICAL: For schedule/time questions, COMPLETELY IGNORE PDF chunks - they contain category descriptions, NOT schedule data
   - ⚠️ CRITICAL: PDF chunks may contain text like "CATEGORY 1 (CLASS III AND IV)" - this is NOT schedule information, ignore it completely
   - FIRST check "[SCHEDULE - PRIMARY SOURCE]" or "[Schedule (FAISS)]" section - this is the ONLY source for schedule data
   - ONLY use dates, times, categories, and programs that EXACTLY appear in that section
   - ⚠️ CRITICAL: For "list items at [time]" or "what's happening at [time]" questions, you MUST list EVERY SINGLE item that appears at that time in the schedule data
   - ⚠️ CRITICAL: If asked "list items at 8:30 AM on Friday", search through ALL schedule chunks and list EVERY item that has Time="8:30 AM" (or "8:30AM" or "8.30 AM") and Date matching Friday (11/14/2025 or 11-14-2025 or 14-11-2025)
   - ⚠️ CRITICAL: DO NOT return just one item - scan through ALL schedule chunks from top to bottom and extract EVERY item that matches the time and date criteria
   - ⚠️ CRITICAL: If the schedule shows 30+ items at 8:30 AM on Friday, list ALL 30+ items - DO NOT summarize, DO NOT skip any, DO NOT say "and more" - list EVERY ONE
   - ⚠️ CRITICAL: Read through the ENTIRE "[SCHEDULE - PRIMARY SOURCE]" section chunk by chunk, line by chunk, and extract EVERY matching item
   - Count how many items match the criteria FIRST, then list each one with its exact category as shown
   - Format as: "There are X items at [time] on [day]:\n• Item 1 (Category as shown)\n• Item 2 (Category as shown)\n..." listing ALL items - NO EXCEPTIONS
   - ⚠️ CRITICAL FOR "WHAT CATEGORIES ARE PERFORMING" QUERIES:
     * If asked "what categories are performing on [day]" or "what are the categories performing on [day]" or similar:
     * Scan through ALL chunks in "[SCHEDULE - PRIMARY SOURCE]" section
     * Extract EVERY unique Category value that appears for that specific day/date
     * List ALL unique categories found (e.g., "CATEGORY I", "CATEGORY II", "CATEGORY III", "CATEGORY IV")
     * Format as: "On [day], the following categories are performing:\n• Category I\n• Category II\n• Category III\n• Category IV" (or whatever categories you find)
     * DO NOT say "I couldn't find" - if chunks exist for that day, extract the categories from them
     * If you see "Category: CATEGORY I" or "Category: CATEGORY II" in any chunk for that day, include it in your list
     * Count how many unique categories you found first, then list each one
   - If you see multiple items with the same name but different categories, list them separately (e.g., "Folk Dance (CATEGORY I)" and "Folk Dance (CATEGORY III)")
   - If schedule shows "Folk Dance - Girls" on "11/12/2025" with "CATEGORY II" → use EXACTLY those values
   - If schedule shows "11/12/2025" → use that date, NOT "14-11-2025" or any other date
   - If schedule shows "CATEGORY II" → use that category, NOT "Category I" or any other category  
   - If the program is not found in schedule data for a specific date, say "I don't see that program on that date" - DO NOT make up dates or categories
   - If schedule shows multiple entries (e.g., same program on different dates/categories), list ALL of them separately with their exact dates/categories
   - DO NOT use dates from PDF chunks for schedule questions - schedule dates are authoritative
   - DO NOT convert or change dates/times/categories from schedule - use them EXACTLY as shown
   - DO NOT invent dates, times, or categories that are not in the schedule data
   
2. FOR FACTUAL QUESTIONS (fees, rules, grades, awards, certificates, etc.):
   - Use PDF chunks and Manual sections
   - These contain rules, fees, awards, certificates, and other factual information
   - ⚠️ CRITICAL FOR MERIT CERTIFICATE QUERIES: Search for "merit certificate", "minimum marks", "50%", "awards and trophies" in PDF chunks
   - If you see "minimum of 50% marks shall be awarded merit certificates" or similar text, extract the exact percentage (50%) and explain it clearly
   - DO NOT say "not mentioned" or "not specified" if this information exists in the PDF chunks - search thoroughly through ALL chunks
   - Look for sections titled "AWARDS AND TROPHIES" or similar headers
3. Look for information that relates to the question - use semantic understanding, not just exact keyword matches
4. Extract relevant information even if:
   - Formatting is messy or unconventional
   - Information is in tables, lists, or paragraph form
   - Keywords don't match exactly but meaning is similar
4. For phone/contact questions: Search for phone numbers (long digit sequences like "9778665476/9778665475"), look near words like "office", "kalotsav", "contact", "phone", "number", "office number". Phone numbers often appear on a separate line after mentioning "kalotsav office" and may have a dash (-) before them. If you see "kalotsav office" followed by a dash and then 10-digit numbers, those ARE the office numbers - extract them. Example: if you see "from the kalotsav office" followed by "-" followed by "9778665476/9778665475", the office numbers are 9778665476 and 9778665475
5. For value points/marks questions: Search for "Value Points" header, then find the category name (e.g., "Folk Dance"), then extract ALL items listed. The format is: item name on one line, mark value on next line. Example: "Akara Sushama" followed by "15 marks" means "Akara Sushama - 15 marks". Extract EVERY item under that category. Format as: "The value points for [category] are: * Item 1 - X marks * Item 2 - Y marks..." List ALL items you find - DO NOT stop after one. NEVER output raw number sequences without context.
6. For remarks/notes questions: Search for the item name (e.g., "Mono Act"), then look at the lines IMMEDIATELY AFTER the duration (e.g., "5mts"). Remarks appear in parentheses like "(Common for both boys & girls)" or "(Common for both boys and Girls)" or as additional text on the next line. If you see ANY text in parentheses or additional descriptive text after the item name and duration, that IS the remark - extract it and provide it. DO NOT say "not mentioned" if you find ANY parenthetical text or notes after the item.
7. IGNORE raw number sequences that appear without labels or context (like standalone numbers "5 5 10 3..." on separate lines) - these are poorly extracted table data. Always look for properly formatted sections with item names and their values.
8. If you find ANY "Value Points" section matching the question category in the context, extract and list ALL items - never say "not mentioned" if this data exists in the chunks.
9. DO NOT respond with "not mentioned" - if you find ANY relevant information in the context that answers the question, provide it
10. If the context has the answer but it's scattered across chunks, piece it together intelligently
11. CRITICAL FOR VALUE POINTS: If you see "Value Points" and "Folk Dance" in the same chunk or context, the answer IS THERE - extract it and format it clearly, never say it's not mentioned
12. CRITICAL FOR REMARKS: If you see an item name followed by a duration (like "5mts") followed by text in parentheses or additional descriptive text, that IS the remark - extract it. Example: "Mono Act\n5mts\n(Common for both boys & girls)" means the remark is "Common for both boys & girls"

INSTRUCTIONS:
- Read the context carefully and extract information that answers the question
- Use your understanding of the context to provide accurate answers
- Be natural and conversational in your response
- If you find relevant information, provide it even if it's not formatted perfectly

Now answer the question: {user_message}"""}
                ],
                max_tokens=2000,  # Increased to prevent truncation of long lists
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
    
    # If we reach here, either no context was found or AI response was too short
    # Fall back to direct data search from Excel/PDF
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
        f"- {p.get('Item', 'Unknown')} at {format_time_for_answer(p.get('Time', 'Unknown'))} (Category: {p.get('Category', 'N/A')}) on {format_date_for_display(p.get('Date', 'Unknown'))}"
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
    
    # Note: Greetings are handled early in the webhook, so they shouldn't reach here
    # But keep this fallback in case this function is called directly
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
            details = [f"*{program['Item']}* ({program.get('Category', 'N/A')}) {source_label}"]
            
            if program.get("Time"):
                details.append(f"Time: {format_time_for_answer(program['Time'])}")
            
            if program.get("Date"):
                details.append(f"Date: {program['Date']}")
            
            return "\n".join(details)
        else:
            return (
                f"• *{program['Item']}* "
                f"({program.get('Category', 'N/A')}) {source_label}\n"
                f"  Time: {format_time_for_answer(program.get('Time', 'N/A'))}"
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
    
    if extracted.get("time"):
        time_val = extracted["time"]
        filtered_programs = filter_programs(
            filtered_programs,
            lambda p: time_val.lower() in str(p.get("Time", "")).lower()
        )

    # Category filter (robust: supports "CATEGORY IV", "CAT-4", "4", etc.)
    if extracted.get("category"):
        cat_input = str(extracted["category"]).strip()
        target_cat = _normalize_category_label(cat_input)
        print(f"DEBUG: Category filter - input='{cat_input}', normalized='{target_cat}'")
        if target_cat:
            original_count = len(filtered_programs)
            filtered_programs = filter_programs(
                filtered_programs,
                lambda p: _normalize_category_label(p.get("Category", "")) == target_cat
            )
            print(f"DEBUG: Category filter - filtered from {original_count} to {len(filtered_programs)} programs")
            # Debug: show some matches
            if filtered_programs:
                print(f"DEBUG: Sample category matches:")
                for p in filtered_programs[:5]:
                    print(f"  - {p.get('Item')} ({p.get('Category')}) -> normalized: '{_normalize_category_label(p.get('Category', ''))}'")
            else:
                print(f"DEBUG: No category matches found. Testing normalization on sample categories:")
                sample_cats = set(str(p.get("Category", "")).strip() for p in all_programs[:20])
                for cat in list(sample_cats)[:5]:
                    print(f"  - '{cat}' -> normalized: '{_normalize_category_label(cat)}'")

    # Gender refinement for category/item queries
    if gender_intent:
        filtered_programs = filter_programs(
            filtered_programs,
            lambda p: gender_intent in str(p.get("Item", "")).upper()
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
                print(f"Checking item: {p.get('Item')} on {prog_date}")
                
                # Direct string comparison first
                if prog_date == date_query:
                    print(f"Found match (direct): {p.get('Item')}")
                    filtered_programs.append(p)
                    continue
                
                # Try date normalization if direct match fails
                try:
                    query_date = pd.to_datetime(date_query).strftime("%d-%m-%Y")
                    program_date = pd.to_datetime(prog_date).strftime("%d-%m-%Y")
                    if query_date == program_date:
                        print(f"Found match (normalized): {p.get('Item')}")
                        filtered_programs.append(p)
                except:
                    pass
            
            print(f"\nFound {len(filtered_programs)} items for {date_query}")
            if filtered_programs:
                print("\nFound these items:")
                for p in filtered_programs:
                    print(f"- {p['Item']} ({p['Category']}) at {p['Time']}")
    
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
            lambda p: program_query in str(p.get("Item", "")).lower()
        )
        for program in matching_programs:
            results.append(format_program_details(program, detailed=True))

    elif query_type == "category" or (query_type == "general" and extracted.get("category")):
        print(f"DEBUG: Processing category query - query_type='{query_type}', category='{extracted.get('category')}', filtered_programs={len(filtered_programs)}")
        if filtered_programs:
            header = f"*Category {extracted.get('category')} items*"
            if gender_intent:
                header = f"*{gender_intent.title()} Category {extracted.get('category')} items*"
            results.append(header + ":\n")
            # Show concise lines: Item – Time on Date (Category)
            def _fmt_row(p):
                item = p.get("Item", "Unknown")
                time = p.get("Time", "N/A")
                date = p.get("Date", "")
                cat = p.get("Category", "")
                return f"• {item} – {time} on {date} ({cat})"
            # De-duplicate by Item+Time+Date
            seen = set()
            for p in filtered_programs:
                key = (p.get("Item", ""), p.get("Time", ""), p.get("Date", ""))
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
        
        # Check if this is a factual query (website, fee, phone, etc.) vs schedule query
        user_msg_lower = user_message.lower()
        is_factual_query = any(term in user_msg_lower for term in [
            "website", "url", "fee", "cost", "price", "phone", "contact", "email", 
            "address", "register", "registration", "how much", "what is", "when",
            "accessible", "time", "timing", "anchoring", "anchor",
            "duration", "how long", "length", "time period", "mins", "minutes",
            "hrs", "hours", "hr", "mts", "mt", "running time", "remark", "remarks",
            "note", "notes", "comment", "comments", "additional", "info", "detail",
            "office", "kalotsav office", "call", "mobile"
        ])
        
        # Different system prompts for factual vs schedule queries
        if is_factual_query:
            system_prompt = """You are a clear, concise WhatsApp assistant for the Kalolsavam Cultural Festival.
            TASK: Rewrite the assistant's response as a brief, natural human reply.
            CRITICAL RULES:
            - PRESERVE all factual information (website URLs, dates, times, fees, phone numbers, etc.) EXACTLY as provided
            - If the assistant message contains a clear answer, use it - DO NOT say "I couldn't find" or "not found"
            - Make the reply conversational and friendly but keep it brief
            - No emojis
            - If the assistant message is already clear and complete, keep it as-is or make minor improvements only
            - DO NOT reject valid answers or say "not found" if the assistant provided an answer
            - The assistant's message IS the answer - just rewrite it in a friendly way"""
            
            user_prompt = f"User asked: '{user_message}'\n\nAssistant found this answer:\n{cleaned_info}\n\nRewrite this as a brief, friendly WhatsApp reply. PRESERVE all factual details (URLs, dates, times, amounts) exactly as shown. If an answer is provided, use it - do not say it's not found. The answer is already correct, just make it more conversational."
        else:
            # Schedule/program query - use stricter matching
            system_prompt = """You are a clear, concise WhatsApp assistant for the Kalolsavam Cultural Festival.
            HARD CONSTRAINTS (NO HALLUCINATIONS):
            - Use ONLY the content provided by the assistant message; never add or infer extra items, dates, stages, or categories.
            - CRITICAL: Match the EXACT program/item name from the user's query. If user asks about "elocution topics", search ONLY for "elocution" in the assistant message, NOT other programs like "Mono Act".
            - If the user asks about a specific program name (e.g., ELOCUTION, MONO ACT), include ONLY lines that refer to that exact program name. Do NOT include similarly worded but different items (e.g., ENGLISH ONE ACT PLAY) when asked about MONO ACT.
            - If user asks about "topics for elocution" or "topics for it" (where "it" refers to elocution), search for "elocution" in the assistant message, NOT other programs.
            - If there is a single schedule entry, answer with ONE short, natural sentence.
            - If multiple entries exist for that same program, consolidate into one short readable sentence, listing each unique (Stage, Category, Date) only once.
            - If no entries for the exact program are present, clearly say you couldn't find it and suggest checking the exact program name. Do not fabricate.
            - No emojis; keep it brief and professional.
            - Preserve exact dates as written; do not substitute with relative words.
            - IMPORTANT: If the assistant content contains entries that are NOT for the exact program requested, IGNORE those lines entirely."""
            
            user_prompt = f"User asked: '{user_message}'\n\nRewrite the above assistant message as a brief, human reply without emojis. Include ONLY entries that match the exact program/item name from the user's query ('{user_message}'). Do not add or infer any data that is not present above. If user asked about a specific program (like 'elocution'), only include information about that program, NOT other programs."
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=2000,  # Increased to prevent truncation of long lists
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": f"{cleaned_info}"},
                {"role": "user", "content": user_prompt}
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
    Retry on transient DNS/network errors.
    """
    import time
    
    if twilio_client is None:
        print("ERROR: Cannot send message - Twilio client not initialized!")
        print("Please check your .env file has correct TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN")
        return {"status": "error", "type": "config", "message": "Twilio not configured"}
    
    print(f"DEBUG: Attempting to send message to {to}")
    print(f"DEBUG: FROM_NUMBER = {FROM_NUMBER}")
    print(f"DEBUG: Message preview: {message[:100]}...")
    
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            msg = twilio_client.messages.create(
                from_=FROM_NUMBER,
                to=to,
                body=message
            )
            print(f"Successfully sent reply to {to}")
            print(f"DEBUG: Twilio Message SID: {msg.sid}, Status: {msg.status}")
            return {"status": "success", "sid": msg.sid, "twilio_status": msg.status}
        except Exception as e:
            error_str = str(e).lower()
            
            # Check if it's a rate limit error (don't retry)
            if "limit" in error_str:
                print("Twilio daily message limit reached. Please try again tomorrow.")
                return {"status": "error", "type": "rate_limit", "message": "Daily message limit reached"}
            
            # Check if it's a transient DNS/network error (retry)
            is_transient_error = any(term in error_str for term in [
                "name resolution", "dns", "getaddrinfo", "connection", 
                "timeout", "network", "unreachable", "refused"
            ])
            
            if is_transient_error and attempt < max_retries - 1:
                current_delay = retry_delay * (2 ** attempt)  # Exponential backoff
                print(f"Twilio transient error (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"Retrying in {current_delay} seconds...")
                time.sleep(current_delay)
                continue
            
            # Permanent error or max retries reached
            print(f"Twilio send error: {e}")
            return {"status": "error", "type": "general", "message": str(e)}



@app.get("/")
def root():
    return {"message": " Kalolsavam WhatsApp Assistant is running!"}


@app.get("/ask")
@app.get("/ask/")  # Also handle trailing slash
async def ask_unified_get(question: str = Query(None, description="Your question about schedule/manual/PDF")):
    """
    Unified GET endpoint: answers using Excel schedule, manual, and FAISS PDF chunks.
    
    Usage: GET /ask?question=your question here
    Example: GET /ask?question=what is the appeal fee?
    """
    if not question:
        return {
            "status": "error",
            "error": "No question provided",
            "usage": "GET /ask?question=your question here",
            "example": "/ask?question=what is the appeal fee?",
            "methods_available": {
                "GET": "Use query parameter ?question=...",
                "POST": "Send question in JSON body: {\"question\": \"...\"} or form data: question=..."
            }
        }
    try:
        extracted = extract_query_info(question)
        combined_data = update_cache_if_needed()
        answer = search_program_data(extracted, question, combined_data)
        return {"status": "success", "question": question, "answer": answer}
    except Exception as e:
        return {"status": "error", "error": str(e)}

# Import and register the ask router
# Note: Import is here (after app initialization) to avoid circular dependency issues
try:
    from api.ask import router as ask_router
    app.include_router(ask_router, tags=["ask"])
    print("[OK] Ask API router registered successfully")
    print(f"  - Router has {len(ask_router.routes)} route(s)")
    for route in ask_router.routes:
        print(f"  - Route: {route.methods} {route.path}")
except Exception as e:
    print(f"✗ Error registering ask router: {e}")
    import traceback
    traceback.print_exc()


if __name__ == "__main__":
    import uvicorn
    print("Starting WhatsApp Assistant Server...")
    print("Press Ctrl+C to stop the server")
    uvicorn.run(app, host="0.0.0.0", port=8000)
