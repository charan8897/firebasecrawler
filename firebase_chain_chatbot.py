#!/usr/bin/env python3
"""
Firebase Chain Chatbot with Gemini
- Dynamic schema discovery from Firebase
- LinkedList-style retry mechanism
- Each attempt node's response informs next search terms
"""

import firebase_admin
from firebase_admin import credentials, firestore
import google.generativeai as genai
import json
import sys
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import config

# Initialize Gemini
genai.configure(api_key=config.GEMINI_API_KEY)
db = config.get_firestore_client()

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class AttemptNode:
    """LinkedList node representing a search attempt"""
    attempt_num: int
    search_terms: List[str]
    target_collections: List[str]
    search_results: Dict[str, List[Dict[str, Any]]]
    decision: str  # "SUCCESS", "RETRY_NEW_TERMS", "GO_DEEPER", "INSUFFICIENT_INFO"
    llm_analysis: Optional[str] = None
    next_terms_suggestion: Optional[List[str]] = None
    next_node: Optional['AttemptNode'] = None
    
    def to_dict(self):
        """Convert node to dictionary for response generation"""
        return {
            'attempt': self.attempt_num,
            'terms_used': self.search_terms,
            'collections_searched': self.target_collections,
            'results_found': sum(len(v) for v in self.search_results.values()),
            'decision': self.decision,
            'analysis': self.llm_analysis,
            'next_terms': self.next_terms_suggestion
        }

# ============================================================================
# FIREBASE OPERATIONS
# ============================================================================

def fetch_all_firebase_data() -> Dict[str, List[Dict[str, Any]]]:
    """Fetch all documents from all collections"""
    data = {}
    collections = config.discover_collections()
    
    for col_name in collections:
        docs = list(db.collection(col_name).stream())
        data[col_name] = [{'doc_id': doc.id, **doc.to_dict()} for doc in docs]
    
    return data

def search_firebase_targeted(
    firebase_data: Dict,
    search_terms: List[str],
    target_collections: List[str]
) -> Dict[str, List[Dict[str, Any]]]:
    """Search Firebase collections with specific terms"""
    results = {}
    
    for col_name in target_collections:
        if col_name not in firebase_data:
            continue
        
        docs = firebase_data[col_name]
        matches = []
        
        for doc in docs:
            for search_term in search_terms:
                search_str = str(search_term).lower().strip()
                search_str_clean = search_str.replace('.', '').replace(',', '').replace("'", '')
                
                for key, value in doc.items():
                    # String field matching
                    if isinstance(value, str):
                        value_clean = value.lower().replace('.', '').replace(',', '').replace("'", '')
                        if search_str in value.lower() or search_str_clean in value_clean:
                            matches.append({
                                'doc_id': doc.get('doc_id', 'N/A'),
                                'collection': col_name,
                                'field': key,
                                'value': value[:150],
                                'match_term': search_str
                            })
                    # List field matching
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, str):
                                item_clean = item.lower().replace('.', '').replace(',', '').replace("'", '')
                                if search_str in item.lower() or search_str_clean in item_clean:
                                    matches.append({
                                        'doc_id': doc.get('doc_id', 'N/A'),
                                        'collection': col_name,
                                        'field': key,
                                        'value': item[:150],
                                        'match_term': search_str
                                    })
        
        if matches:
            # Remove duplicates by doc_id
            seen = {}
            for match in matches:
                key = (match['doc_id'], match['field'])
                if key not in seen:
                    seen[key] = match
            results[col_name] = list(seen.values())
    
    return results

# ============================================================================
# GEMINI LLM OPERATIONS
# ============================================================================

def find_intent(user_query: str, firebase_schema: Dict) -> Dict[str, Any]:
    """Use Gemini to extract intent and search terms from user query"""
    
    print(f"🧠 [LLM] Analyzing intent for query: '{user_query}'\n")
    
    schema_text = "\n".join([
        f"- {col}: {schema['fields']}"
        for col, schema in firebase_schema.items()
    ])
    
    prompt = f"""Analyze this user query and determine:
1. The INTENT (what the user is looking for)
2. RELEVANT COLLECTIONS to search
3. KEY SEARCH TERMS (individual keywords, not phrases)

Available Collections & Fields:
{schema_text}

User Query: "{user_query}"

Rules for search terms:
- Extract individual keywords
- Split names into components (e.g., "Dr. Misra" → ["Misra", "Dr"])
- Remove common words
- Keep technical terms and names

Respond ONLY in JSON format:
{{
    "intent": "brief description of what user wants",
    "collections": ["collection1", "collection2"],
    "search_terms": ["term1", "term2", "term3"],
    "confidence": 0.8
}}"""
    
    try:
        model = genai.GenerativeModel(config.GEMINI_MODEL)
        response = model.generate_content(prompt)
        
        json_str = response.text
        if "```" in json_str:
            json_str = json_str.split("```")[1].replace("json", "").strip()
        
        intent_data = json.loads(json_str)
        
        # Detailed logging
        print(f"✓ [LLM] Intent Analysis Complete:")
        print(f"    Intent: {intent_data.get('intent')}")
        print(f"    Confidence: {intent_data.get('confidence', 'N/A')}")
        print(f"    Collections: {intent_data.get('collections')}")
        print(f"    Generated Terms: {intent_data.get('search_terms')}\n")
        
        return intent_data
    except Exception as e:
        print(f"⚠️  [LLM] Intent parsing failed: {e}\n")
        # Fallback
        fallback_data = {
            "intent": "search",
            "collections": config.discover_collections(),
            "search_terms": user_query.split(),
            "confidence": 0.3
        }
        print(f"⚠️  [FALLBACK] Using default intent:")
        print(f"    Terms: {fallback_data['search_terms']}\n")
        return fallback_data

def evaluate_and_decide_next(
    attempt_node: AttemptNode,
    user_query: str,
    firebase_schema: Dict,
    firebase_data: Dict,
    max_attempts: int
) -> Tuple[bool, Optional[List[str]], str]:
    """
    Evaluate search results and decide next action.
    Returns: (should_continue, next_search_terms, decision)
    """
    
    has_results = sum(len(v) for v in attempt_node.search_results.values()) > 0
    result_count = sum(len(v) for v in attempt_node.search_results.values())
    
    print(f"\n🔍 [EVAL] Evaluating Attempt #{attempt_node.attempt_num}/{max_attempts}")
    print(f"    Terms Used: {attempt_node.search_terms}")
    print(f"    Collections Searched: {attempt_node.target_collections}")
    print(f"    Results Found: {result_count} matches")
    print(f"    Result Collections: {list(attempt_node.search_results.keys())}")
    
    # Show brief schema info
    schema_info = {
        col: {
            'fields': schema.get('fields', []),
            'total_docs': schema.get('total_docs', 0)
        }
        for col, schema in firebase_schema.items()
    }
    
    # Extract sample values from actual data (filtered)
    sample_values = {}
    for col in attempt_node.target_collections:
        if col in firebase_data and firebase_data[col]:
            doc = filter_large_fields(firebase_data[col][0])
            sample_values[col] = {
                k: str(v)[:50] 
                for k, v in doc.items() 
                if isinstance(v, (str, int, float)) and k != 'doc_id' and '[FILTERED]' not in str(v)
            }
    
    prompt = f"""You are a smart database search assistant. Evaluate this search attempt:

USER QUERY: "{user_query}"
ATTEMPT #{attempt_node.attempt_num}/{max_attempts}

Search Terms Used: {attempt_node.search_terms}
Collections Searched: {attempt_node.target_collections}
Results Found: {result_count} matches
⚠️  CRITICAL: Results = {result_count}

SAMPLE DATA FROM ACTUAL COLLECTIONS (what values exist):
{json.dumps(sample_values, indent=2)}

Database Schema:
{json.dumps(schema_info, indent=2)}

STRICT DECISION RULES:
1. "SUCCESS" → ONLY if results_count > 0 AND relevant to query
2. "RETRY_NEW_TERMS" → If results_count = 0 OR results irrelevant (try different keywords)
3. "GO_DEEPER" → If results exist but need refinement with additional context
4. "INSUFFICIENT_INFO" → Only after 3+ failed retry attempts

KEY CONSTRAINTS:
- NEVER return SUCCESS if results_count = 0 (this means search failed)
- Use ACTUAL VALUES from sample data when suggesting new terms
- Extract keywords from sample values (e.g., if you see "CEO IMMERSION PROGRAMME", search for "CEO")
- Attempt #{attempt_node.attempt_num}: Suggest terms based on REAL DATA shown above

Respond ONLY in JSON:
{{
    "decision": "SUCCESS" | "RETRY_NEW_TERMS" | "GO_DEEPER" | "INSUFFICIENT_INFO",
    "reasoning": "brief explanation",
    "next_search_terms": ["new_term1", "new_term2"] or null,
    "alternative_collections": {attempt_node.target_collections} or null,
    "strategy": "what to try next - use actual values from sample data"
}}"""
    
    try:
        print(f"    🧠 [LLM] Requesting evaluation decision...\n")
        model = genai.GenerativeModel(config.GEMINI_MODEL)
        response = model.generate_content(prompt)
        
        json_str = response.text
        if "```" in json_str:
            json_str = json_str.split("```")[1].replace("json", "").strip()
        
        result = json.loads(json_str)
        
        attempt_node.llm_analysis = result.get('reasoning', '')
        decision = result.get('decision', 'INSUFFICIENT_INFO')
        next_terms = result.get('next_search_terms')
        strategy = result.get('strategy', '')
        
        # FORCED LOGIC: If 0 results, must retry (not SUCCESS)
        if result_count == 0 and decision == "SUCCESS":
            print(f"⚠️  [OVERRIDE] LLM returned SUCCESS with 0 results - forcing RETRY_NEW_TERMS")
            decision = "RETRY_NEW_TERMS"
            # Generate fallback terms if LLM didn't suggest any
            if not next_terms:
                words = user_query.split()
                next_terms = [w for w in words if len(w) > 3]  # Take words longer than 3 chars
                if not next_terms:
                    next_terms = words
        
        should_continue = decision in ["RETRY_NEW_TERMS", "GO_DEEPER"]
        
        # Detailed decision logging
        print(f"✓ [LLM DECISION] Received:")
        print(f"    Decision: {decision}")
        print(f"    Reasoning: {attempt_node.llm_analysis}")
        print(f"    Strategy: {strategy}")
        
        if should_continue:
            print(f"    📌 Next Action: CONTINUE CHAIN")
            if next_terms:
                print(f"    Next Terms Generated: {next_terms}")
        else:
            print(f"    📌 Next Action: STOP CHAIN")
        print()
        
        return should_continue, next_terms, decision
    except Exception as e:
        print(f"⚠️  [LLM] Evaluation failed: {e}\n")
        return False, None, "INSUFFICIENT_INFO"

# ============================================================================
# RETRY CHAIN - LinkedList Style
# ============================================================================

def search_with_retry_chain(
    user_query: str,
    firebase_data: Dict,
    firebase_schema: Dict,
    max_attempts: int = 5,
    min_attempts: int = 2
) -> AttemptNode:
    """
    LinkedList-style retry mechanism.
    Each node's response informs next search terms (like a linked list chain).
    """
    
    print(f"\n{'='*70}")
    print(f"⛓️  STARTING LINKEDLIST RETRY CHAIN")
    print(f"{'='*70}")
    print(f"  Min Attempts: {min_attempts} | Max Attempts: {max_attempts}\n")
    
    # Get initial intent
    initial_intent = find_intent(user_query, firebase_schema)
    
    head_node = None
    current_node = None
    attempt_count = 0
    
    search_terms = initial_intent.get('search_terms', user_query.split())
    target_collections = initial_intent.get('collections', config.discover_collections())
    
    print(f"{'='*70}\n")
    
    # LinkedList chain loop
    while attempt_count < max_attempts:
        attempt_count += 1
        
        print(f"\n{'─'*70}")
        print(f"⛓️  NODE #{attempt_count} - LinkedList Attempt")
        print(f"{'─'*70}")
        print(f"\n🔎 [SEARCH] Executing Firebase search:")
        print(f"    Search Terms: {search_terms}")
        print(f"    Target Collections: {target_collections}")
        
        # Search Firebase
        search_results = search_firebase_targeted(firebase_data, search_terms, target_collections)
        result_count = sum(len(v) for v in search_results.values())
        
        print(f"\n✓ [SEARCH RESULTS]:")
        print(f"    Total Matches: {result_count}")
        if search_results:
            for col, matches in search_results.items():
                print(f"    - {col}: {len(matches)} document(s)")
        else:
            print(f"    - No results found")
        
        # Create attempt node
        new_node = AttemptNode(
            attempt_num=attempt_count,
            search_terms=search_terms.copy(),
            target_collections=target_collections.copy(),
            search_results=search_results,
            decision="PENDING"
        )
        
        # Link to previous node (LinkedList chain)
        if current_node:
            current_node.next_node = new_node
            print(f"\n🔗 [CHAIN] Linked to previous node (#{attempt_count-1})")
        else:
            head_node = new_node
            print(f"\n🔗 [CHAIN] Created head node")
        
        current_node = new_node
        
        # Evaluate and decide next step
        should_continue, next_terms, decision = evaluate_and_decide_next(
            new_node, user_query, firebase_schema, firebase_data, max_attempts
        )
        
        new_node.decision = decision
        new_node.next_terms_suggestion = next_terms
        
        # ENFORCE MINIMUM ATTEMPTS BEFORE STOPPING
        print(f"\n📋 [LOGIC] Checking stop conditions:")
        print(f"    Attempt Count: {attempt_count}/{max_attempts}")
        print(f"    Min Attempts Met: {attempt_count >= min_attempts}")
        print(f"    Decision: {decision}")
        print(f"    Should Continue: {should_continue}\n")
        
        if attempt_count >= min_attempts:
            if not should_continue or decision == "SUCCESS":
                print(f"{'='*70}")
                print(f"🛑 [STOP] Terminating chain at attempt {attempt_count}")
                print(f"   Reason: Min attempts met + {decision}")
                print(f"{'='*70}\n")
                break
        else:
            print(f"ℹ️  Continuing: Minimum attempts not yet reached ({attempt_count}/{min_attempts})\n")
        
        # Prepare next iteration
        if should_continue and next_terms:
            print(f"🔄 [NEXT ITERATION] Preparing attempt #{attempt_count+1}")
            print(f"   New terms from LLM: {next_terms}\n")
            search_terms = next_terms
        elif should_continue and not next_terms:
            print(f"🔄 [NEXT ITERATION] Continuing with same terms\n")
        else:
            if attempt_count < min_attempts:
                print(f"🔄 [NEXT ITERATION] Enforcing min attempts, continuing with same terms\n")
            else:
                print(f"🛑 [STOP] No more iterations needed\n")
                break
    
    print(f"{'='*70}")
    print(f"✅ CHAIN COMPLETE - {attempt_count} total attempts")
    print(f"{'='*70}\n")
    
    return head_node

# ============================================================================
# RESPONSE GENERATION
# ============================================================================

def is_base64_data(s: str) -> bool:
    """Check if string is likely base64 encoded data"""
    if not isinstance(s, str) or len(s) < 50:
        return False
    
    # Base64 with data: prefix
    if s.startswith('data:'):
        return True
    
    # Check if it's mostly base64 characters (A-Z, a-z, 0-9, +, /, =)
    # and longer than 100 chars
    base64_pattern = re.compile(r'^[A-Za-z0-9+/]*={0,2}$')
    return len(s) > 100 and base64_pattern.match(s) is not None

def filter_large_fields(data: Dict) -> Dict:
    """Remove large/binary fields (base64 images, etc) to save tokens"""
    if not isinstance(data, dict):
        return data
    
    filtered = {}
    
    for key, value in data.items():
        # Check if value is base64 encoded data
        if isinstance(value, str) and is_base64_data(value):
            filtered[key] = f"[base64_data_{len(value)}_chars]"
        else:
            filtered[key] = value
    
    return filtered

def generate_final_response(
    user_query: str,
    attempt_chain_head: AttemptNode,
    firebase_data: Dict
) -> str:
    """Use Gemini to generate natural language response from entire chain"""
    
    # Collect all attempt data
    attempts_info = []
    current = attempt_chain_head
    all_results = {}
    
    while current:
        attempts_info.append(current.to_dict())
        # Merge results
        for col, matches in current.search_results.items():
            if col not in all_results:
                all_results[col] = []
            all_results[col].extend(matches)
        current = current.next_node
    
    # Deduplicate by doc_id and enrich with full document data
    enriched_results = {}
    for col in all_results:
        seen = {}
        for match in all_results[col]:
            doc_id = match['doc_id']
            if doc_id not in seen:
                # Find full document in firebase_data
                full_doc = None
                if col in firebase_data:
                    full_doc = next((d for d in firebase_data[col] if d['doc_id'] == doc_id), None)
                    # Filter out large fields
                    if full_doc:
                        full_doc = filter_large_fields(full_doc)
                
                seen[doc_id] = {
                    'match': match,
                    'full_document': full_doc or match
                }
        enriched_results[col] = list(seen.values())
    
    # Extract sample data for LLM context on what data exists (filtered)
    sample_data = {}
    for col, docs in firebase_data.items():
        if docs:
            sample_doc = filter_large_fields(docs[0].copy())
            # Show sample values from first doc
            sample_data[col] = {
                'sample_fields': list(sample_doc.keys()),
                'sample_values': {k: str(v)[:60] for k, v in sample_doc.items() if isinstance(v, (str, int, float)) and k != 'doc_id'}
            }
    
    prompt = f"""You are a helpful assistant for IIM event management system.

User Query: "{user_query}"

Search Process (Attempt Chain):
{json.dumps(attempts_info, indent=2)}

AVAILABLE DATA IN DATABASE (Sample):
{json.dumps(sample_data, indent=2)}

SEARCH RESULTS:
{json.dumps(enriched_results, indent=2, default=str)}

Based on the search results and available data, provide a response:
- If results found: Show relevant details (names, emails, dates, locations, etc.)
- If no results: Suggest specific search terms based on what data exists in the database
- Be friendly and helpful
- Keep response concise"""
    
    try:
        model = genai.GenerativeModel(config.GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {e}"

# ============================================================================
# MAIN CHATBOT
# ============================================================================

def chatbot_response(user_query: str) -> str:
    """Main chatbot function: Query → Intent → Chain Search → Response"""
    
    print(f"\n{'='*70}")
    print(f"🤖 NEW CHATBOT SESSION")
    print(f"{'='*70}")
    print(f"📝 User Query: '{user_query}'")
    print(f"{'='*70}\n")
    
    # Load Firebase data
    print("📊 [INIT] Loading Firebase data...")
    firebase_data = fetch_all_firebase_data()
    total_docs = sum(len(v) for v in firebase_data.values())
    print(f"✓ [INIT] Loaded {total_docs} documents across collections:\n")
    
    for col_name, docs in firebase_data.items():
        print(f"    - {col_name}: {len(docs)} docs")
    
    # Get schema
    print(f"\n📋 [INIT] Discovering Firebase schema...")
    firebase_schema = config.discover_schema()
    print(f"✓ [INIT] Schema discovered for {len(firebase_schema)} collections\n")
    
    # Run LinkedList retry chain
    attempt_chain_head = search_with_retry_chain(
        user_query,
        firebase_data,
        firebase_schema,
        max_attempts=config.MAX_RETRY_ATTEMPTS,
        min_attempts=config.MIN_RETRY_ATTEMPTS
    )
    
    # Generate response
    print("📝 [RESPONSE] Generating final response from chain...\n")
    response = generate_final_response(user_query, attempt_chain_head, firebase_data)
    
    print(f"{'='*70}")
    print(f"✅ SESSION COMPLETE")
    print(f"{'='*70}\n")
    
    return response

# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Main CLI interface"""
    print("\n" + "="*60)
    print("🎯 Gemini Firebase Chain Chatbot")
    print("="*60)
    print("Dynamic schema discovery | LinkedList retry chain\n")
    
    if len(sys.argv) > 1:
        # Command line query
        query = " ".join(sys.argv[1:])
        result = chatbot_response(query)
        print(f"💬 Response:\n{result}\n")
    else:
        # Interactive mode
        while True:
            try:
                user_input = input("\n📝 You: ").strip()
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("👋 Goodbye!\n")
                    break
                if user_input:
                    result = chatbot_response(user_input)
                    print(f"\n💬 Chatbot:\n{result}")
            except KeyboardInterrupt:
                print("\n👋 Interrupted. Goodbye!\n")
                break

if __name__ == "__main__":
    main()
