# -*- coding: utf-8 -*-
"""
Excel Embedding Creator
Creates FAISS embeddings from Excel files for fast semantic search.
Similar to pdf_ocr_extractor.py but for Excel data.
"""

import pandas as pd
import os
from dotenv import load_dotenv

try:
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    _FAISS_AVAILABLE = True
except Exception as e:
    print(f"FAISS dependencies not available: {e}")
    print("Please install: pip install langchain-openai langchain-community faiss-cpu")
    _FAISS_AVAILABLE = False

# Load environment variables
load_dotenv()

# Default paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, "data")
PDF_STORAGE = os.path.join(DATA_FOLDER, "pdfs")
EXCEL_VECTORS_DIR = os.path.join(BASE_DIR, "my_excel_vectors")


def find_excel_file():
    """Find the newest Excel file in the data/pdfs directory"""
    if not os.path.exists(PDF_STORAGE):
        return None
    
    excel_files = []
    for file in os.listdir(PDF_STORAGE):
        # Skip Excel lock/temp files that start with '~$'
        if file.startswith('~$'):
            continue
        if file.lower().endswith(('.xlsx', '.xls')):
            file_path = os.path.join(PDF_STORAGE, file)
            try:
                mtime = os.path.getmtime(file_path)
                excel_files.append((file, mtime))
            except Exception:
                mtime = 0
                excel_files.append((file, mtime))
    
    if excel_files:
        excel_files.sort(key=lambda t: t[1], reverse=True)
        newest_name, _ = excel_files[0]
        excel_path = os.path.join(PDF_STORAGE, newest_name)
        print(f"Found Excel file: {excel_path}")
        return excel_path
    
    return None


def load_and_clean_excel(excel_path):
    """
    Load Excel file and clean the data, similar to main_demo.py's update_excel_cache()
    Returns a list of dictionaries with program data
    """
    print(f"\n=== Loading Excel File ===")
    print(f"Reading from: {excel_path}")
    
    # Read Excel file
    df = pd.read_excel(
        excel_path,
        header=None,
        dtype=str,  # Force all columns to be string
        keep_default_na=False,  # Don't convert empty cells to NaN
        parse_dates=False  # Don't parse dates
    )
    
    print(f"Raw Excel shape: {df.shape}")
    
    # Dynamically rename columns for schedule Excel file
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
    
    # Rename columns
    df = df.rename(columns={v: k for k, v in map_cols.items() if v in df.columns})
    print(f"Columns after rename: {df.columns.tolist()}")
    
    # Clean up the data
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace('nan', '')
    
    # Remove empty/header rows
    df = df.dropna(how='all')
    if 'Item' in df.columns:
        df = df[df['Item'].str.len() > 0]
        
        # Remove header-like rows
        header_like_items = {"item", "items", "program", "program name", "programme", "programs", "programmes"}
        header_like_times = {"time", "times", "stage", "stages"}
        header_like_categories = {"category", "categories"}
        header_like_dates = {"date", "dates"}
        
        if 'Item' in df.columns:
            df = df[~df['Item'].str.strip().str.lower().isin(header_like_items)]
        if 'Time' in df.columns:
            df = df[~df['Time'].str.strip().str.lower().isin(header_like_times)]
        if 'Category' in df.columns:
            df = df[~df['Category'].str.strip().str.lower().isin(header_like_categories)]
        if 'Date' in df.columns:
            df = df[~df['Date'].str.strip().str.lower().isin(header_like_dates)]
        
        # Drop rows where all columns equal their header tokens
        if all(col in df.columns for col in ['Time', 'Item', 'Category', 'Date']):
            df = df[~(
                df['Time'].str.strip().str.upper().eq('TIME') &
                df['Item'].str.strip().str.upper().eq('ITEM') &
                df['Category'].str.strip().str.upper().eq('CATEGORY') &
                df['Date'].str.strip().str.upper().eq('DATE')
            )]
    
    print(f"After cleaning: {len(df)} rows")
    
    # Convert to list of dictionaries
    programs = []
    for _, row in df.iterrows():
        try:
            time = str(row.get('Time', '')).strip()
            item_name = str(row.get('Item', '')).strip()
            category = str(row.get('Category', '')).strip()
            date = str(row.get('Date', '')).strip()
            
            # Skip empty rows
            if not item_name or item_name.lower() == 'nan':
                continue
            
            program = {
                "Item": item_name,
                "Category": category if category != "nan" else "N/A",
                "Time": time,
                "Date": date
            }
            programs.append(program)
        except Exception as e:
            print(f"Error processing row: {e}")
            continue
    
    print(f"Total programs extracted: {len(programs)}")
    return programs


def programs_to_text_chunks(programs):
    """
    Convert program dictionaries to text chunks for embedding.
    Each program becomes a chunk with all its information.
    Includes day names for better semantic matching (e.g., "Friday" matches with "11/14/2025").
    """
    from datetime import datetime
    
    chunks = []
    for program in programs:
        # Create a descriptive text chunk for each program
        item = program.get('Item', 'Unknown')
        category = program.get('Category', 'N/A')
        time = program.get('Time', 'N/A')
        date = program.get('Date', 'N/A')
        
        # Try to extract day name from date
        day_name = ""
        if date and date != 'N/A':
            try:
                # Try different date formats
                date_formats = [
                    "%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%Y/%m/%d",
                    "%m-%d-%Y", "%m/%d/%Y", "%d-%m-%y", "%d/%m/%y"
                ]
                date_obj = None
                for fmt in date_formats:
                    try:
                        date_obj = datetime.strptime(str(date).strip(), fmt)
                        break
                    except:
                        continue
                
                if date_obj:
                    day_name = date_obj.strftime("%A")  # Full day name (Monday, Tuesday, etc.)
            except:
                pass
        
        # Create a comprehensive text chunk with day name for better semantic search
        chunk_parts = [
            f"Program: {item}",
            f"Category: {category}",
            f"Scheduled Time: {time}",
            f"Date: {date}"
        ]
        
        if day_name:
            chunk_parts.append(f"Day: {day_name}")
        
        # Add natural language description with day name
        if day_name:
            chunk_parts.append(f"Event: {item} is scheduled to perform in category {category} at {time} on {day_name} ({date}).")
        else:
            chunk_parts.append(f"Event: {item} is scheduled to perform in category {category} at {time} on {date}.")
        
        # Add category-focused description for better category queries
        chunk_parts.append(f"Category {category} programs include {item}.")
        
        chunk_text = "\n".join(chunk_parts)
        chunks.append(chunk_text)
    
    return chunks


def create_excel_embeddings(excel_path=None, output_path=None):
    """
    Create FAISS embeddings from Excel file.
    
    Args:
        excel_path: Path to Excel file (if None, will search for newest Excel in data/pdfs)
        output_path: Path to save FAISS index (default: my_excel_vectors in script directory)
    
    Returns:
        dict: Status and information about the created embeddings
    """
    if not _FAISS_AVAILABLE:
        return {
            'status': 'error',
            'message': 'FAISS dependencies not available. Install langchain-openai and langchain-community'
        }
    
    try:
        # Find Excel file if not provided
        if excel_path is None:
            excel_path = find_excel_file()
            if not excel_path:
                return {
                    'status': 'error',
                    'message': 'No Excel file found in data/pdfs directory'
                }
        
        if not os.path.exists(excel_path):
            return {
                'status': 'error',
                'message': f'Excel file not found: {excel_path}'
            }
        
        # Set default output path if not provided
        if output_path is None:
            output_path = EXCEL_VECTORS_DIR
        
        print(f"\n{'='*60}")
        print("=== CREATING EXCEL EMBEDDINGS ===")
        print(f"{'='*60}")
        print(f"Excel file: {excel_path}")
        print(f"Output directory: {output_path}")
        
        # Load and clean Excel data
        programs = load_and_clean_excel(excel_path)
        
        if not programs:
            return {
                'status': 'error',
                'message': 'No valid programs found in Excel file'
            }
        
        # Convert programs to text chunks
        print(f"\nConverting {len(programs)} programs to text chunks...")
        chunks = programs_to_text_chunks(programs)
        print(f"Created {len(chunks)} text chunks")
        
        # Create embeddings
        print("\nGenerating embeddings with OpenAI...")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {
                'status': 'error',
                'message': 'OPENAI_API_KEY not found in environment variables'
            }
        
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        
        # Create FAISS vectorstore
        print("Building FAISS vectorstore...")
        vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
        
        # Ensure directory exists
        os.makedirs(output_path, exist_ok=True)
        
        # Save vectorstore
        print(f"Saving FAISS index to: {output_path}")
        vectorstore.save_local(output_path)
        
        print(f"\n{'='*60}")
        print("✓ SUCCESS: Excel embeddings created!")
        print(f"✓ Total chunks: {len(chunks)}")
        print(f"✓ Saved to: {output_path}")
        print(f"{'='*60}\n")
        
        return {
            'status': 'success',
            'chunks_created': len(chunks),
            'output_path': output_path,
            'excel_file': excel_path,
            'message': f'Successfully created {len(chunks)} embeddings from Excel file'
        }
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            'status': 'error',
            'message': f'Error creating embeddings: {str(e)}'
        }


if __name__ == "__main__":
    """
    Run this script to create Excel embeddings.
    Usage: python excel_embedding_creator.py
    """
    print("Excel Embedding Creator")
    print("=" * 60)
    
    result = create_excel_embeddings()
    
    if result['status'] == 'success':
        print(f"\n✓ Embeddings created successfully!")
        print(f"  - Chunks: {result['chunks_created']}")
        print(f"  - Location: {result['output_path']}")
    else:
        print(f"\n✗ Error: {result['message']}")
        exit(1)

