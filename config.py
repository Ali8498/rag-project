# تنظیمات پروژه RAG
import os
from dotenv import load_dotenv

# بارگذاری توکن از فایل .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# مدل Embedding
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# تنظیمات ChromaDB
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "rag_docs"

# تنظیمات Groq
GROQ_MODEL = "llama-3.3-70b-versatile"

# تنظیمات تقسیم متن
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
