from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pandas as pd
import openai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# FastAPI initialization
app = FastAPI()

# OpenAI setup
openai.api_key = os.getenv("OPENAI_API_KEY")

# Excel file path
EXCEL_FILE = "data/pdfs/PROGRAMME SCHEDULE.xlsx"

def load_excel_data():
    """Load Excel data and convert it to a formatted string for LLM"""
    try:
        print("\n=== Loading Excel File ===")
        print(f"Reading from: {EXCEL_FILE}")
        
        if not os.path.exists(EXCEL_FILE):
            print(f"ERROR: Excel file not found at {EXCEL_FILE}")
            return "No schedule data available.", []
            
        # Read Excel file with explicit settings
        df = pd.read_excel(
            EXCEL_FILE,
            dtype=str,  # Force all columns to be string
            keep_default_na=False,  # Don't convert empty cells to NaN
            na_filter=False  # Don't filter any values as NA
        )
        
        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]
        
        # Map column names to standard names
        column_mapping = {
            'Time': 'Time',
            'Item': 'Item',
            'Category': 'Category',
            'Date': 'Date'
        }
        
        # Try to find matching columns
        for col in df.columns:
            col_lower = col.lower()
            if 'time' in col_lower:
                column_mapping[col] = 'Time'
            elif 'item' in col_lower or 'program' in col_lower or 'event' in col_lower:
                column_mapping[col] = 'Item'
            elif 'category' in col_lower or 'type' in col_lower:
                column_mapping[col] = 'Category'
            elif 'date' in col_lower or 'day' in col_lower:
                column_mapping[col] = 'Date'
                
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Clean data
        for col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            # For Date column, extract only the date part (YYYY-MM-DD)
            if col == 'Date' and ' ' in df[col].iloc[0] if len(df) > 0 else False:
                df[col] = df[col].str.split(' ').str[0]
            df[col] = df[col].replace(['nan', 'None', 'NaN', ''], 'N/A')
            
        # Create program list
        programs = []
        for _, row in df.iterrows():
            program = {
                'Time': row.get('Time', 'N/A'),
                'Item': row.get('Item', 'N/A'),
                'Category': row.get('Category', 'N/A'),
                'Date': row.get('Date', 'N/A')
            }
            if not all(v == 'N/A' for v in program.values()):
                programs.append(program)
                
        # Create formatted string for LLM
        excel_content = "PROGRAM SCHEDULE:\n\n"
        
        # Group by time and date
        programs_by_time = {}
        for prog in programs:
            time = prog['Time']
            if time not in programs_by_time:
                programs_by_time[time] = {}
            
            date = prog['Date']
            if date not in programs_by_time[time]:
                programs_by_time[time][date] = []
            
            programs_by_time[time][date].append(prog)
            
        # Format the content
        for time in sorted(programs_by_time.keys()):
            excel_content += f"\n⏰ {time}:\n"
            for date in sorted(programs_by_time[time].keys()):
                excel_content += f"  📅 {date}:\n"
                for prog in sorted(programs_by_time[time][date], key=lambda x: x['Item']):
                    excel_content += f"    • {prog['Item']} ({prog['Category']})\n"
                    
        return excel_content, programs
    except Exception as e:
        print(f"Error loading Excel data: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading Excel data: {str(e)}")

class QuestionRequest(BaseModel):
    question: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "which stage BHARATHA BOYS is performing on 12-11-2025"
            }
        }

@app.get("/ask")
async def ask_about_schedule_get(
    question: str = Query(None, description="The question to ask about the schedule")
):
    """Ask questions about the schedule using GET method"""
    try:
        if not question:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "No question provided",
                    "usage": {
                        "GET": "/ask?question=your question here",
                        "POST": {
                            "url": "/ask",
                            "body": {"question": "your question here"},
                            "headers": {"Content-Type": "application/json"}
                        }
                    }
                }
            )
        return await process_question(question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_about_schedule_post(
    request_body: dict
):
    """Ask questions about the schedule using POST method"""
    try:
        # Try to get question from either "question" or "message" field
        final_question = request_body.get('question') or request_body.get('message')
        
        if not final_question:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "No question provided",
                    "expected": "Either 'question' or 'message' field in request body",
                    "received": request_body
                }
            )
            
        return await process_question(final_question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def process_question(final_question: str):
    """Process the question and return the answer"""
    try:
        # Load Excel data
        excel_content, programs = load_excel_data()
        
        # Calculate statistics
        times = sorted(set(p['Time'] for p in programs if p['Time'] != 'N/A'))
        dates = sorted(set(p['Date'] for p in programs if p['Date'] != 'N/A'))
        categories = sorted(set(p['Category'] for p in programs if p['Category'] != 'N/A'))
        
        # Create prompt for OpenAI
        prompt = f"""You are a knowledgeable assistant for the Kalolsavam Cultural Festival. Here is the complete program data:

FESTIVAL STATISTICS:
Total Times: {len(times)}
Total Programs: {len(programs)}
Festival Dates: {', '.join(dates)}
Available Times: {', '.join(times)}
Program Categories: {', '.join(categories)}

DETAILED PROGRAM SCHEDULE:
{excel_content}

Current Question: "{final_question}"

RESPONSE GUIDELINES:
1. For questions about numbers (times, items, etc.):
   - ALWAYS start with "There are X [items] in total" or "The total number of [items] is X"
   - Then list ALL items in a clear format
   - End with any relevant details about the items
   Example: 
   "There are 3 times in total at the festival:
   ⏰ 1:00 PM
   ⏰ 2:00 PM
   ⏰ 3:00 PM"

2. For specific program queries:
   - Give a direct answer about the time, date, and category
   Example: "BHARATHA BOYS is performing at 1:00 PM on 12-11-2025 (Category 3)"

3. For date/schedule questions:
   - IMPORTANT: Count ALL programs for that date in the schedule above
   - IMPORTANT: List EVERY SINGLE program for that date - DO NOT SKIP ANY
   - Group by stage
   - Include category information
   - Example: If asked "who all are performing on friday", search for "2025-11-14" in the schedule and list ALL 41 programs

4. Response Format:
   - Start with a direct answer including the exact count
   - List EVERY program found, ensuring you don't miss any
   - Use emojis appropriately (🎭 for stages, 📅 for dates)
   - Use bullet points for lists
   - End with a friendly note
   
5. CRITICAL: When listing programs for a specific date, you MUST count and include ALL of them. Do not return a partial list."""

        # Get response from OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """You are a helpful assistant that provides information about the Kalolsavam Cultural Festival program schedule. 
                
                CRITICAL RULES:
                1. When asked about programs on a specific date, you MUST list ALL programs for that date
                2. Count the programs in the schedule data carefully
                3. Do NOT return partial lists - ensure you include every single program
                4. When asked about numbers (like 'how many stages'), ALWAYS:
                   - Start with the exact number in a complete sentence
                   - Then list all items if appropriate
                   - Finally, provide any additional details
                
                Example for "how many times are there?":
                "There are 3 times in total: 1:00 PM, 2:00 PM, and 3:00 PM."
                
                Example for "who all are performing on friday?":
                "There are 41 programs scheduled for Friday, 2025-11-14. Here they are:
                [list ALL 41 programs grouped by stage]"
                """},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract and return the response
        answer = response['choices'][0]['message']['content'].strip()
        
        return {
            "question": final_question,
            "answer": answer,
            "source": "Excel Schedule"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Excel Schedule API is running"}

@app.get("/programs")
def get_all_programs():
    """Get all programs from the Excel file"""
    try:
        _, programs = load_excel_data()
        return {"programs": programs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("Starting Excel Schedule API...")
    print(f"Excel file path: {os.path.abspath(EXCEL_FILE)}")
    uvicorn.run(app, host="0.0.0.0", port=8001)