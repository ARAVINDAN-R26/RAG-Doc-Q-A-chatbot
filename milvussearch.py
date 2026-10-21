from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

def search_similar_docs(query, top_k=3):
    # Connect to Milvus (adjust host/port if different)
    connections.connect(host="127.0.0.1", port="19530")

    # Load the collection
    collection = Collection("rag_docs")
    collection.load()

    # Create embedding for the query
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = embedder.encode([query])

    # Search in Milvus
    results = collection.search(
        data=query_embedding,
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["text"]
    )

    # Return list of matched texts
    return [hit.entity.get("text") for hit in results[0]]
