import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# مدل Embedding چندزبانه — پشتیبانی از فارسی، انگلیسی، عربی و ۵۰+ زبان
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# تنظیمات ChromaDB
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "rag_docs"

# تنظیمات Groq
GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 1000
TEMPERATURE = 0.2

# تنظیمات تقسیم متن
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# تنظیمات جستجو
TOP_K = 5
FETCH_K = 10
MMR_LAMBDA = 0.85       # بالاتر = relevance بیشتر
SCORE_THRESHOLD = 0.3   # chunk با score کمتر از این حذف میشه
SHOW_SCORES = True      # نمایش score در رابط کاربری


# تشخیص زبان و ترجمه
ENABLE_TRANSLATION = True