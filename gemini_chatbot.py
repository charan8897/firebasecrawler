#!/usr/bin/env python3
import firebase_admin
from firebase_admin import credentials, firestore
import google.generativeai as genai
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Firebase Setup
cred = credentials.Certificate("/home/charan/Downloads/iim-ems-firebase-adminsdk-fbsvc-3f4ff679a4.json")
firebase_admin.initialize_app(cred, options={'storageBucket': None})
db = firestore.client()

# Gemini Setup
GEMINI_API_KEY = "AIzaSyB_R1IyNoViLuHX313A9vUyXd1M-Vr5uR4"
genai.configure(api_key=GEMINI_API_KEY)

COLLECTIONS = ['agenda', 'events', 'quiz_results', 'quizzes', 'speakers', 'students', 'users']

# LinkedList-style Attempt Node
@dataclass
class AttemptNode:
    """Represents an attempt in the search chain"""
    attempt_num: int
    search_terms: List[str]
    search_results: Dict[str, Any]
    decision: str  # "SUCCESS", "RETRY_NEW_TERMS", "GO_DEEPER", "INSUFFICIENT_INFO"
    next_terms_suggestion: Optional[List[str]] = None
    llm_analysis: Optional[str] = None
    next_node: Optional['AttemptNode'] = None
    
    def to_dict(self):
        return {
            'attempt': self.attempt_num,
            'terms': self.search_terms,
            'found': len(self.search_results),
            'decision': self.decision,
            'next_terms': self.next_terms_suggestion,
            'analysis': self.llm_analysis
        }

def fetch_all_firebase_data():
    """Fetch all data from Firebase collections"""
    data = {}
    for col_name in COLLECTIONS:
        collection = db.collection(col_name)
        docs = list(collection.stream())
        data[col_name] = [{'doc_id': doc.id, **doc.to_dict()} for doc in docs]
    return data

def get_firebase_schema():
    """Get complete Firebase schema with all fields and sample values"""
    schema = {}
    for col_name in COLLECTIONS:
        collection = db.collection(col_name)
        docs = list(collection.stream())
        if docs:
            sample_doc = docs[0].to_dict()
            schema[col_name] = {
                'total_docs': len(docs),
                'fields': list(sample_doc.keys()),
                'sample': {k: str(v)[:80] for k, v in sample_doc.items()}
            }
    return schema

def find_intent(user_query, firebase_data):
    """Use Gemini to find intent and relevant collections"""
    
    prompt = f"""Analyze this user query and determine:
1. The INTENT (what the user wants to find)
2. The RELEVANT COLLECTIONS to search in
3. KEY SEARCH TERMS to filter the data (extract INDIVIDUAL keywords and names, NOT full phrases)

Available collections:
- agenda: Event agenda with dates, timings, speakers, eventName, startDate, endDate
- events: Event details, name, speakers count, status, date
- quiz_results: Quiz performance, scores, user results, userName, quizTitle
- quizzes: Quiz questions, titles, descriptions
- speakers: Speaker profiles, name, email, company, topic, designation
- students: Student profiles, name, email, company, designation, mobile
- users: User accounts, email, name, uid

User Query: "{user_query}"

IMPORTANT: Extract search terms as INDIVIDUAL keywords:
- "Dr. Misra" → extract ["Subhasis", "Misra"] (last name, first name)
- "CEO event" → extract ["CEO"]
- "Flutter quiz" → extract ["Flutter"]
- Names: split into first and last names

Respond ONLY in valid JSON:
{{
    "intent": "what the user wants",
    "collections": ["relevant", "collections"],
    "search_terms": ["individual_keyword1", "individual_keyword2"],
    "confidence": 0.9
}}"""
    
    model = genai.GenerativeModel("gemma-3-27b-it")
    response = model.generate_content(prompt)
    
    try:
        # Parse JSON from response
        json_str = response.text
        # Extract JSON if wrapped in markdown code blocks
        if "```" in json_str:
            json_str = json_str.split("```")[1].replace("json", "").strip()
        intent_data = json.loads(json_str)
    except Exception as e:
        # Fallback: extract search terms from query
        search_terms = user_query.split()
        intent_data = {
            "intent": "search",
            "collections": COLLECTIONS,
            "search_terms": search_terms,
            "confidence": 0.5
        }
    
    return intent_data

def search_firebase(firebase_data, intent_data):
    """Search Firebase data based on intent"""
    results = {}
    
    target_collections = intent_data.get('collections', COLLECTIONS)
    search_terms = intent_data.get('search_terms', [])
    
    for col_name in target_collections:
        if col_name not in firebase_data:
            continue
            
        docs = firebase_data[col_name]
        matches = []
        
        for doc in docs:
            for search_term in search_terms:
                search_str = str(search_term).lower()
                
                for key, value in doc.items():
                    if isinstance(value, str):
                        if search_str in value.lower():
                            matches.append({
                                'doc_id': doc.get('doc_id', 'N/A'),
                                'field': key,
                                'value': value[:100],
                                'match': search_str
                            })
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, str) and search_str in item.lower():
                                matches.append({
                                    'doc_id': doc.get('doc_id', 'N/A'),
                                    'field': key,
                                    'value': item[:100],
                                    'match': search_str
                                })
        
        if matches:
            results[col_name] = list({v['doc_id']: v for v in matches}.values())  # Remove duplicates
    
    return results

def evaluate_and_decide_next(attempt_node: AttemptNode, user_query: str, firebase_schema: Dict, firebase_data: Dict) -> tuple[bool, Optional[List[str]], str]:
    """
    Evaluate if current search was successful or needs retry.
    Returns: (should_continue, next_search_terms, decision_reason)
    """
    
    has_results = sum(len(v) for v in attempt_node.search_results.values()) > 0
    
    prompt = f"""You are helping refine a database search for: "{user_query}"

ATTEMPT #{attempt_node.attempt_num}:
- Search Terms Used: {attempt_node.search_terms}
- Results Found: {has_results} ({sum(len(v) for v in attempt_node.search_results.values())} total matches)
- Collections Searched: {list(attempt_node.search_results.keys())}

AVAILABLE DATABASE SCHEMA:
{json.dumps(firebase_schema, indent=2)}

DECISION RULES:
1. If we found good results → respond with "SUCCESS"
2. If results are wrong/irrelevant → create new search terms → "RETRY_NEW_TERMS"
3. If results are incomplete → go deeper with refined terms → "GO_DEEPER"
4. If we've tried 3+ times → give up → "INSUFFICIENT_INFO"

Current attempt number: {attempt_node.attempt_num}/5

Respond ONLY in JSON:
{{
    "decision": "SUCCESS" | "RETRY_NEW_TERMS" | "GO_DEEPER" | "INSUFFICIENT_INFO",
    "reasoning": "brief explanation",
    "next_search_terms": ["term1", "term2"] or null,
    "strategy": "which fields to target next or what to search for"
}}"""
    
    model = genai.GenerativeModel("gemma-3-27b-it")
    response = model.generate_content(prompt)
    
    try:
        json_str = response.text
        if "```" in json_str:
            json_str = json_str.split("```")[1].replace("json", "").strip()
        result = json.loads(json_str)
        
        attempt_node.llm_analysis = result.get('reasoning', '')
        decision = result.get('decision', 'INSUFFICIENT_INFO')
        next_terms = result.get('next_search_terms')
        strategy = result.get('strategy', '')
        
        should_continue = decision in ["RETRY_NEW_TERMS", "GO_DEEPER"]
        
        return should_continue, next_terms, decision
    except:
        return False, None, "INSUFFICIENT_INFO"

def search_with_retry_chain(user_query: str, firebase_data: Dict, firebase_schema: Dict, max_attempts: int = 5, min_attempts: int = 2) -> AttemptNode:
    """
    LinkedList-style retry mechanism with chaining.
    Each node's response becomes input for the next node.
    Minimum 2 attempts to allow evaluation and refinement.
    """
    
    # Initial intent finding
    initial_intent = find_intent(user_query, firebase_data)
    
    head_node = None
    current_node = None
    attempt_count = 0
    
    search_terms = initial_intent.get('search_terms', user_query.split())
    target_collections = initial_intent.get('collections', COLLECTIONS)
    
    while attempt_count < max_attempts:
        attempt_count += 1
        
        # Search Firebase
        search_results = search_firebase_targeted(firebase_data, search_terms, target_collections)
        
        # Create attempt node
        new_node = AttemptNode(
            attempt_num=attempt_count,
            search_terms=search_terms,
            search_results=search_results,
            decision="PENDING"
        )
        
        # Link to previous node
        if current_node:
            current_node.next_node = new_node
        else:
            head_node = new_node
        
        current_node = new_node
        
        # Evaluate and decide next step
        should_continue, next_terms, decision = evaluate_and_decide_next(
            new_node, user_query, firebase_schema, firebase_data
        )
        
        new_node.decision = decision
        new_node.next_terms_suggestion = next_terms
        
        print(f"  Attempt {attempt_count}: {decision} | Found: {sum(len(v) for v in search_results.values())} matches")
        
        # MINIMUM 2 ATTEMPTS ENFORCED
        # Only break if we have minimum attempts AND (no more refinement needed OR reached max)
        if attempt_count >= min_attempts:
            if not should_continue or decision == "SUCCESS" or decision == "INSUFFICIENT_INFO":
                break
        
        # Use suggested terms for next attempt
        if next_terms:
            search_terms = next_terms
        else:
            # If no next terms suggested but we haven't hit min_attempts, continue anyway
            if attempt_count < min_attempts:
                continue
            else:
                break
    
    return head_node

def search_firebase_targeted(firebase_data, search_terms, target_collections):
    """Search Firebase data with specific terms and collections - keeps ALL matching fields per document"""
    results = {}
    
    for col_name in target_collections:
        if col_name not in firebase_data:
            continue
            
        docs = firebase_data[col_name]
        doc_matches = {}  # {doc_id: [all matches for this doc]}
        
        for doc in docs:
            for search_term in search_terms:
                search_str = str(search_term).lower().strip()
                # Remove periods and special characters for flexible matching
                search_str_clean = search_str.replace('.', '').replace(',', '').replace("'", '')
                
                doc_id = doc.get('doc_id', 'N/A')
                if doc_id not in doc_matches:
                    doc_matches[doc_id] = []
                
                for key, value in doc.items():
                    if isinstance(value, str):
                        value_clean = value.lower().replace('.', '').replace(',', '').replace("'", '')
                        if search_str in value.lower() or search_str_clean in value_clean:
                            match_entry = {
                                'doc_id': doc_id,
                                'field': key,
                                'value': value[:100],
                                'match': search_str
                            }
                            # Avoid duplicate field entries
                            if not any(m['field'] == key for m in doc_matches[doc_id]):
                                doc_matches[doc_id].append(match_entry)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, str):
                                item_clean = item.lower().replace('.', '').replace(',', '').replace("'", '')
                                if search_str in item.lower() or search_str_clean in item_clean:
                                    match_entry = {
                                        'doc_id': doc_id,
                                        'field': key,
                                        'value': item[:100],
                                        'match': search_str
                                    }
                                    # Avoid duplicate field entries
                                    if not any(m['field'] == key for m in doc_matches[doc_id]):
                                        doc_matches[doc_id].append(match_entry)
        
        if doc_matches:
            # Flatten all matches - return all fields for all matching docs
            all_matches = []
            for doc_id, matches_list in doc_matches.items():
                all_matches.extend(matches_list)
            results[col_name] = all_matches
    
    return results

def generate_response(user_query, attempt_chain_head: AttemptNode, firebase_data: Dict):
    """Use Gemini to generate natural language response from the entire chain"""
    
    # Collect all attempt data
    attempts_info = []
    current = attempt_chain_head
    final_results = {}
    
    while current:
        attempts_info.append(current.to_dict())
        # Merge results from all attempts
        for col, matches in current.search_results.items():
            if col not in final_results:
                final_results[col] = matches
        current = current.next_node
    
    results_text = json.dumps(final_results, indent=2, default=str)
    attempts_text = json.dumps(attempts_info, indent=2)
    
    prompt = f"""You are a helpful chatbot for an IIM event management system.

User Query: "{user_query}"

Search Attempts Made:
{attempts_text}

Final Results from Database:
{results_text}

Based on all the search attempts and final results, provide a helpful and natural response to the user's query.
If no results found, suggest how the user could refine their search.
Keep response concise and friendly."""
    
    model = genai.GenerativeModel("gemma-3-27b-it")
    response = model.generate_content(prompt)
    
    return response.text

def chatbot_response(user_query):
    """Main chatbot function: Query -> Intent -> Search with Retry Chain -> Response"""
    
    print(f"\n🤖 Processing: '{user_query}'...\n")
    
    # Fetch Firebase data
    print("📊 Fetching Firebase data...")
    firebase_data = fetch_all_firebase_data()
    print(f"✓ Loaded {sum(len(v) for v in firebase_data.values())} documents")
    
    # Get schema for LLM context
    print("📋 Analyzing Firebase schema...")
    firebase_schema = get_firebase_schema()
    
    # LinkedList-style retry chain
    print("🔄 Starting search with intelligent retry mechanism...\n")
    attempt_chain_head = search_with_retry_chain(user_query, firebase_data, firebase_schema, max_attempts=5)
    
    # Count total matches across chain
    current = attempt_chain_head
    total_matches = 0
    while current:
        total_matches += sum(len(v) for v in current.search_results.values())
        current = current.next_node
    
    print(f"\n✓ Chain complete | Total matches found: {total_matches}\n")
    
    # Generate response using final chain
    print("📝 Generating response from search chain...\n")
    response = generate_response(user_query, attempt_chain_head, firebase_data)
    
    return response

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        result = chatbot_response(query)
        print(f"🤖 Response:\n{result}\n")
    else:
        print("🎯 Gemini Chatbot - IIM Event Management System")
        print("=" * 50)
        while True:
            user_input = input("\n📝 You: ").strip()
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("👋 Goodbye!")
                break
            if user_input:
                result = chatbot_response(user_input)
                print(f"🤖 Chatbot:\n{result}")
