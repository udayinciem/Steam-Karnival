"""
PDF Processing Module for Manual State 2025
Extracts text from PDF and provides question-answering capabilities
"""

import os
import fitz  # PyMuPDF
import openai
from dotenv import load_dotenv
import json
from typing import Dict, List, Optional

# Load environment variables
load_dotenv()

# OpenAI setup
openai.api_key = os.getenv("OPENAI_API_KEY")

# Get the path to the PDF
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "data", "pdfs", "Manual State 2025 (2).pdf")


class PDFQuestionAnswerer:
    """Handles PDF text extraction and question-answering"""
    
    def __init__(self, pdf_path: str = PDF_PATH):
        self.pdf_path = pdf_path
        self.pdf_text = None
        self._cache_file = os.path.join(BASE_DIR, "data", "pdfs", "manual_text_cache.txt")
        
    def extract_text_from_pdf(self) -> Optional[str]:
        """
        Extract text from the PDF file
        Returns the extracted text or None if extraction fails
        """
        try:
            # Check if we have cached text
            if os.path.exists(self._cache_file):
                print("Loading cached PDF text...")
                with open(self._cache_file, 'r', encoding='utf-8') as f:
                    self.pdf_text = f.read()
                    return self.pdf_text
            
            # Extract text from PDF
            print(f"Extracting text from PDF: {self.pdf_path}")
            
            if not os.path.exists(self.pdf_path):
                print(f"Error: PDF file not found at {self.pdf_path}")
                return None
            
            text_parts = []
            pdf_document = fitz.open(self.pdf_path)
            
            print(f"PDF has {pdf_document.page_count} pages")
            
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                text = page.get_text()
                text_parts.append(f"\n--- Page {page_num + 1} ---\n{text}")
                print(f"Extracted text from page {page_num + 1}")
            
            pdf_document.close()
            
            self.pdf_text = "\n".join(text_parts)
            
            # Cache the extracted text
            os.makedirs(os.path.dirname(self._cache_file), exist_ok=True)
            with open(self._cache_file, 'w', encoding='utf-8') as f:
                f.write(self.pdf_text)
            
            print("PDF text extraction completed")
            return self.pdf_text
            
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return None
    
    def get_pdf_text(self) -> Optional[str]:
        """Get PDF text, extracting if necessary"""
        if self.pdf_text is None:
            self.extract_text_from_pdf()
        return self.pdf_text
    
    def answer_question(self, question: str, context_window: int = 4000) -> str:
        """
        Answer questions about the PDF content using OpenAI
        
        Args:
            question: The user's question
            context_window: Maximum characters to include from the PDF text
            
        Returns:
            A detailed answer based on the PDF content
        """
        try:
            # Get the PDF text
            pdf_text = self.get_pdf_text()
            
            if not pdf_text:
                return "Sorry, I couldn't read the PDF file. Please check if the file exists."
            
            # Limit the context to avoid token limits
            if len(pdf_text) > context_window:
                # Use the first and last parts of the text to maintain context
                half_window = context_window // 2
                truncated_text = pdf_text[:half_window] + "\n[... content truncated ...]\n" + pdf_text[-half_window:]
            else:
                truncated_text = pdf_text
            
            # Create the prompt
            system_prompt = """You are a helpful assistant that answers questions based on the "Manual State 2025" document.
            Answer questions accurately and in detail based ONLY on the information provided in the document.
            If the information is not in the document, say so clearly.
            Be conversational and helpful. Use clear formatting and structure your answers well."""
            
            user_prompt = f"""Based on the following document, please answer this question: {question}

Here is the document content:
{truncated_text}

Please provide a detailed and helpful answer."""
            
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content.strip()
            return answer
            
        except Exception as e:
            print(f"Error answering question: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def get_summary(self) -> str:
        """
        Get a summary of the PDF content
        
        Returns:
            A summary of the PDF
        """
        try:
            pdf_text = self.get_pdf_text()
            
            if not pdf_text:
                return "Could not generate summary. PDF file not found or could not be read."
            
            # Use only the first 8000 characters for summary to avoid token limits
            text_for_summary = pdf_text[:8000]
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates concise summaries of documents."},
                    {"role": "user", "content": f"Please provide a concise summary of the following document:\n\n{text_for_summary}"}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generating summary: {str(e)}"


# Global instance
_qa_instance = None

def get_qa_instance() -> PDFQuestionAnswerer:
    """Get or create the global QA instance"""
    global _qa_instance
    if _qa_instance is None:
        _qa_instance = PDFQuestionAnswerer()
    return _qa_instance

def answer_question(question: str) -> str:
    """
    Convenience function to answer a question about the PDF
    
    Args:
        question: The user's question
        
    Returns:
        The answer as a string
    """
    qa = get_qa_instance()
    return qa.answer_question(question)

def get_manual_summary() -> str:
    """
    Get a summary of the manual
    
    Returns:
        Summary as a string
    """
    qa = get_qa_instance()
    return qa.get_summary()

