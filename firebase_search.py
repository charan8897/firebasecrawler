import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("/home/charan/Downloads/iim-ems-firebase-adminsdk-fbsvc-3f4ff679a4.json")
firebase_admin.initialize_app(cred, options={'storageBucket': None})
db = firestore.client()

collections_to_search = ['agenda', 'events', 'quiz_results', 'quizzes', 'speakers', 'students', 'users']
search_term = 'purav'
results = {}

print(f"Searching for '{search_term}' across all collections...\n")

for collection_name in collections_to_search:
    collection = db.collection(collection_name)
    docs = list(collection.stream())
    
    matches = []
    for doc in docs:
        data = doc.to_dict()
        for key, value in data.items():
            if isinstance(value, str) and search_term.lower() in value.lower():
                matches.append({
                    'doc_id': doc.id,
                    'field': key,
                    'value': value
                })
    
    if matches:
        results[collection_name] = matches

if results:
    print(f"Found '{search_term}' in {len(results)} collection(s):\n")
    for col_name, matches in results.items():
        print(f"📌 Collection: {col_name}")
        for match in matches:
            print(f"   - Doc ID: {match['doc_id']}")
            print(f"     Field: {match['field']}")
            print(f"     Value: {match['value']}")
        print()
else:
    print(f"✗ No matches found for '{search_term}' in any collection")
