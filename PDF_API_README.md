# PDF Question-Answering API

This document describes the new PDF question-answering feature for the Manual State 2025 PDF.

## Overview

The system now includes a `pdf.py` module and API endpoints that allow you to ask questions about the content in the "Manual State 2025 (2).pdf" file located in `data/pdfs/`.

## Features

- Extract text from PDF files using PyMuPDF
- Cache extracted text for faster subsequent queries
- Answer questions using OpenAI's GPT model
- Get summaries of the PDF content
- RESTful API endpoints with GET and POST support

## API Endpoints

### 1. Ask a Question (GET)

**Endpoint:** `GET /ask-pdf`

**Parameters:**
- `question` (query parameter): Your question about the PDF

**Example:**
```bash
curl "http://localhost:8000/ask-pdf?question=What are the main topics covered in this manual?"
```

**Response:**
```json
{
  "status": "success",
  "question": "What are the main topics covered in this manual?",
  "answer": "Based on the Manual State 2025, the main topics include..."
}
```

### 2. Ask a Question (POST)

**Endpoint:** `POST /ask-pdf`

**Parameters:**
- `question` (form data): Your question about the PDF

**Example:**
```bash
curl -X POST "http://localhost:8000/ask-pdf" -d "question=What is the purpose of this manual?"
```

### 3. Get PDF Summary (GET)

**Endpoint:** `GET /pdf-summary`

**Description:** Returns a concise summary of the PDF content

**Example:**
```bash
curl "http://localhost:8000/pdf-summary"
```

**Response:**
```json
{
  "status": "success",
  "summary": "The Manual State 2025 provides comprehensive information about..."
}
```

### 4. Get PDF Summary (POST)

**Endpoint:** `POST /pdf-summary`

**Description:** Returns a concise summary of the PDF content (POST method)

**Example:**
```bash
curl -X POST "http://localhost:8000/pdf-summary"
```

## How It Works

1. **Text Extraction**: When first accessed, the system extracts all text from the PDF using PyMuPDF
2. **Caching**: The extracted text is saved to `data/pdfs/manual_text_cache.txt` for faster subsequent access
3. **Question Answering**: Uses OpenAI's GPT-3.5-turbo to answer questions based on the PDF content
4. **Context Management**: Automatically handles long documents by using the most relevant sections

## Usage Examples

### Python Example

```python
import requests

# Ask a question
response = requests.get(
    "http://localhost:8000/ask-pdf",
    params={"question": "What are the key regulations mentioned?"}
)
print(response.json()["answer"])

# Get summary
response = requests.get("http://localhost:8000/pdf-summary")
print(response.json()["summary"])
```

### JavaScript/Node.js Example

```javascript
// Ask a question
const response = await fetch(
  "http://localhost:8000/ask-pdf?question=What are the main topics?"
);
const data = await response.json();
console.log(data.answer);

// Get summary
const summaryResponse = await fetch("http://localhost:8000/pdf-summary");
const summaryData = await summaryResponse.json();
console.log(summaryData.summary);
```

### cURL Examples

```bash
# Ask a specific question
curl "http://localhost:8000/ask-pdf?question=What is the purpose of this document?"

# Ask about dates
curl "http://localhost:8000/ask-pdf?question=What are the important dates?"

# Get a summary
curl "http://localhost:8000/pdf-summary"
```

## Configuration

Make sure you have the following environment variables set in your `.env` file:

```
OPENAI_API_KEY=your_openai_api_key_here
```

## File Structure

```
.
├── pdf.py                    # PDF processing module
├── main.py                   # Main API server (with PDF endpoints)
├── example_pdf_usage.py      # Example usage
└── data/
    └── pdfs/
        ├── Manual State 2025 (2).pdf  # Source PDF
        └── manual_text_cache.txt     # Cached text (generated)
```

## Testing

Run the server:

```bash
python main.py
```

Then test with the example script:

```bash
python example_pdf_usage.py
```

Or use curl to test endpoints:

```bash
# Test question endpoint
curl "http://localhost:8000/ask-pdf?question=What is this manual about?"

# Test summary endpoint
curl "http://localhost:8000/pdf-summary"
```

## Troubleshooting

### PDF not found
- Make sure "Manual State 2025 (2).pdf" exists in `data/pdfs/`
- Check the file path in `pdf.py`

### OpenAI API errors
- Verify your `OPENAI_API_KEY` is set correctly
- Check that you have API credits available
- Review API usage limits

### Import errors
- Ensure all dependencies in `requirements.txt` are installed
- Run: `pip install -r requirements.txt`

## Performance

- First query may take longer as it extracts and caches the PDF text
- Subsequent queries use the cached text and are faster
- The cache file is created at `data/pdfs/manual_text_cache.txt`
- To refresh the cache, delete the cache file and restart the server

## License

This is part of the WhatsApp Integration project.

