import os
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"
COLLECTION_NAME = "rag_docs"

# Load embedder once globally
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def create_milvus_collection():
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

    if utility.has_collection("rag_docs"):
        utility.drop_collection("rag_docs")

    if not utility.has_collection(COLLECTION_NAME):
        fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
                ]
        schema = CollectionSchema(fields, description="RAG document chunks")
        collection = Collection(COLLECTION_NAME, schema)
        collection.create_index(
                field_name="embedding",
                index_params={
                    "index_type": "IVF_FLAT",
                    "metric_type": "L2",
                    "params": {"nlist": 128}
                    }
                )
        print(f"✅ Created collection '{COLLECTION_NAME}'")
    else:
        print(f"ℹ️ Using existing collection '{COLLECTION_NAME}'")

    collection = Collection(COLLECTION_NAME)
    collection.load()
    return collection

def insert_chunks_to_milvus(chunks, collection):
    embeddings = embedder.encode(chunks)
    collection.insert([embeddings, chunks])
    collection.flush()
    print(f"✅ Inserted {len(chunks)} chunks")

def extract_text_from_pdf(file_path):
    import pdfplumber
    full_text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    return full_text

