from fastapi import Request, APIRouter, Query
import json
import re

# Create a router for the ask endpoints
router = APIRouter()


@router.post("/ask")
@router.post("/ask/")  # Also handle trailing slash
async def ask_unified_post(request: Request):
    """
    Unified POST endpoint: accepts questions in multiple formats.
    Works exactly like /webhook but via API.
    
    - JSON body: {"question": "...", "user_id": "..."} or {"message": "...", "user_id": "..."}
    - Form data: question=...&user_id=...
    
    Optional: user_id - If provided, enables followup detection using chat history
    """
    # Lazy imports to avoid circular dependency issues
    from main_demo import (
        extract_query_info,
        clear_schedule_cache,
        update_cache_if_needed,
        search_program_data,
        fetch_all_chats,
        store_chat,
        generate_human_like_reply,
        client,
        program_cache
    )
    from datetime import datetime
    
    user_message = None
    user_id = None
    content_type = request.headers.get("content-type", "")
    
    # Try JSON first (most common)
    if "application/json" in content_type or not content_type:
        try:
            payload = await request.json()
            user_message = payload.get("question") or payload.get("message")
            user_id = payload.get("user_id") or payload.get("user_number") or payload.get("user_mobile")
        except Exception:
            # If JSON fails, might be form data
            pass
    
    # Try form data if no question found yet
    if not user_message:
        if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
            try:
                form_data = await request.form()
                user_message = form_data.get("question") or form_data.get("message")
                user_id = form_data.get("user_id") or form_data.get("user_number") or form_data.get("user_mobile")
            except Exception:
                pass
        # Also try form data even if content-type is missing or unclear
        elif not content_type:
            try:
                form_data = await request.form()
                user_message = form_data.get("question") or form_data.get("message")
                user_id = form_data.get("user_id") or form_data.get("user_number") or form_data.get("user_mobile")
            except Exception:
                pass
    
    # Clean up message (remove whitespace)
    if user_message:
        user_message = user_message.strip()
        if not user_message:
            user_message = None
    
    if not user_message:
        # Provide helpful error with format examples
        return {
            "status": "error",
            "error": "No question provided",
            "examples": {
                "json": {
                    "method": "POST",
                    "url": "/ask",
                    "headers": {"Content-Type": "application/json"},
                    "body": {"question": "what is the appeal fee?", "user_id": "optional_for_followup"}
                },
                "form_data": {
                    "method": "POST",
                    "url": "/ask",
                    "headers": {"Content-Type": "multipart/form-data"},
                    "body": "question=what is the appeal fee?&user_id=optional"
                },
                "urlencoded": {
                    "method": "POST",
                    "url": "/ask",
                    "headers": {"Content-Type": "application/x-www-form-urlencoded"},
                    "body": "question=what is the appeal fee?&user_id=optional"
                }
            },
            "content_type_received": content_type,
            "note": "Include 'user_id' to enable followup detection based on chat history"
        }
    
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
        
        # Store greeting in database if user_id provided
        if user_id:
            try:
                store_chat(
                    user_mobile=user_id,
                    timestamp=datetime.utcnow(),
                    question=user_message,
                    response=greeting_reply
                )
            except Exception as e:
                print(f"DEBUG: Failed to store greeting: {e}")
        
        return {
            "status": "success",
            "question": user_message,
            "answer": greeting_reply,
            "is_greeting": True
        }
    
    # Step 2: Get chat history (last 5 turns) for intelligent follow-up detection
    # Only if user_id is provided
    chat_history_entries = []
    if user_id:
        try:
            chat_history_entries = fetch_all_chats(user_id)
        except Exception as e:
            print(f"DEBUG: Failed to fetch chat history: {e}")
            chat_history_entries = []
    
    recent_entries = chat_history_entries[-5:] if len(chat_history_entries) > 5 else chat_history_entries
    chat_history = ""
    for entry in recent_entries:
        chat_history += (
            f"User ({entry['user_timestamp']}): {entry['user_question']}\n"
            f"Bot: {entry['response']}\n"
        )
    chat_history += f"User (now): {user_message}\n"

    # Step 3: AI-driven follow-up analysis (only if user_id provided and chat history exists)
    final_query = user_message
    is_followup = False
    if user_id and chat_history_entries:
        try:
            followup_resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                temperature=0.1,
                messages=[
                    {"role": "system", "content": "You are the Receptionist for Kalsolavm. Decide if the user's latest message is a follow-up to the prior conversation. Return strict JSON only."},
                    {"role": "user", "content": f"Conversation so far (last 5 turns):\n{chat_history}\n\nTask: Is the latest message a follow-up to the previous topic? \n\nCRITICAL: A follow-up question refers back to or continues the previous conversation topic. Even if the follow-up repeats some information, you MUST include ALL filters and context from the previous messages.\n\nIf YES, rewrite it into a COMPLETE standalone query that includes ALL context from previous messages:\n- ALWAYS include ALL filters from the most recent previous question, even if the follow-up repeats some of them\n- If previous query mentioned a time (e.g., 'after 2 pm', 'after 3 pm'), include it in the standalone query\n- If previous query mentioned a date or day (e.g., 'Saturday', '12-11-2025'), include it in the standalone query\n- If previous query mentioned a category (e.g., 'category 4'), include it in the standalone query\n- Combine ALL filters from the conversation into one complete query\n- If the follow-up starts with 'then', 'what about', 'also', 'and', or similar continuation words, it's ALWAYS a follow-up\n\nExamples:\n- Previous: 'what items after 2 pm on Saturday?' Follow-up: 'only category 4' → Standalone: 'what are the category 4 items after 2 pm on Saturday?'\n- Previous: 'what programs after 3 pm on Saturday?' Follow-up: 'then what is the programs after 3 pm' → Standalone: 'what are the programs after 3 pm on Saturday?' (maintains the Saturday filter from previous)\n- Previous: 'category 4 programs' Follow-up: 'after 3 pm' → Standalone: 'what are the category 4 programs after 3 pm?'\n\nReturn JSON: {{\"is_followup\": true|false, \"standalone_query\": \"...\"}}"}
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
    
    # Store the chat with full details (only if user_id provided)
    if user_id:
        try:
            store_chat(
                user_mobile=user_id,
                timestamp=datetime.utcnow(),
                question=user_message,
                response=final_reply
            )
            print(f"DEBUG: Conversation stored for user_id: {user_id}")
        except Exception as e:
            print(f"DEBUG: Failed to store conversation: {e}")
    
    result = {
        "status": "success",
        "question": user_message,
        "answer": final_reply
    }
    
    # Include followup info in response if applicable
    if is_followup:
        result["is_followup"] = True
        result["expanded_query"] = final_query
    
    return result

