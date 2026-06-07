from app.vectorstore.chroma_store import ChromaStore

store = ChromaStore()

data = store.get_collection_data()

for meta in data["metadatas"][:10]:
    print(meta)