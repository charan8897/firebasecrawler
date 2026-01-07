"""Configuration file - Dynamic schema discovery from Firebase"""

import firebase_admin
from firebase_admin import credentials, firestore
from typing import Dict, List, Any
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Firebase credentials
FIREBASE_CREDS_PATH = os.getenv('FIREBASE_CREDS_PATH')
if not FIREBASE_CREDS_PATH:
    raise ValueError("❌ FIREBASE_CREDS_PATH not set in .env file")

STORAGE_BUCKET = None

# Gemini API - Get from environment
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not set in .env file")

GEMINI_MODEL = "gemma-3-27b-it"

# Retry mechanism
MAX_RETRY_ATTEMPTS = 5
MIN_RETRY_ATTEMPTS = 2
RETRY_TIMEOUT_SECONDS = 30

# Initialize Firebase once
_db = None
_schema_cache = None

def get_firestore_client():
    """Get or initialize Firestore client"""
    global _db
    if _db is None:
        cred = credentials.Certificate(FIREBASE_CREDS_PATH)
        firebase_admin.initialize_app(cred, options={'storageBucket': STORAGE_BUCKET})
        _db = firestore.client()
    return _db

def discover_collections() -> List[str]:
    """Dynamically fetch all collection names from Firebase"""
    db = get_firestore_client()
    # List all collections in database
    collections = [col.id for col in db.collections()]
    return collections

def discover_schema() -> Dict[str, Dict[str, Any]]:
    """Dynamically discover schema by fetching sample documents from each collection"""
    global _schema_cache
    
    if _schema_cache:
        return _schema_cache
    
    db = get_firestore_client()
    schema = {}
    
    # Get all collections
    collections = discover_collections()
    
    for col_name in collections:
        collection = db.collection(col_name)
        docs = list(collection.limit(5).stream())  # Sample 5 docs to understand structure
        
        if docs:
            # Extract fields from all sample docs
            all_fields = set()
            for doc in docs:
                all_fields.update(doc.to_dict().keys())
            
            schema[col_name] = {
                'total_docs': len(list(collection.stream())),
                'fields': sorted(list(all_fields)),
                'sample': docs[0].to_dict() if docs else {}
            }
    
    _schema_cache = schema
    return schema

def get_collections():
    """Get all collections from Firebase"""
    return discover_collections()

def get_schema():
    """Get full schema from Firebase"""
    return discover_schema()

# For backward compatibility - lazy load collections on demand
COLLECTIONS = None

def load_collections():
    """Lazy load collections from Firebase"""
    global COLLECTIONS
    if COLLECTIONS is None:
        COLLECTIONS = discover_collections()
    return COLLECTIONS

if __name__ == "__main__":
    # Test schema discovery
    print("🔍 Discovering Firebase Collections...\n")
    collections = discover_collections()
    print(f"Found Collections: {collections}\n")
    
    print("📋 Discovering Schema...\n")
    schema = discover_schema()
    
    for col_name, col_schema in schema.items():
        print(f"📦 Collection: {col_name}")
        print(f"   Total Docs: {col_schema.get('total_docs', 0)}")
        print(f"   Fields: {col_schema.get('fields', [])}")
        print()
