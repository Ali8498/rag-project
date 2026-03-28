# فایل ingest.py — بارگذاری و ایندکس اسناد

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from config import (
    EMBEDDING_MODEL,
    CHROMA_PATH,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

def load_document(file_path):
    """
    فایل رو میخونه — PDF یا TXT
    """
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError("فقط فایل PDF یا TXT قبول میشه!")
    
    documents = loader.load()
    print(f"✅ فایل خونده شد — {len(documents)} صفحه پیدا شد")
    return documents


def split_documents(documents):
    """
    متن رو به chunk های کوچیک تقسیم میکنه
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ متن تقسیم شد — {len(chunks)} chunk ساخته شد")
    return chunks


def save_to_chroma(chunks):
    """
    chunk ها رو به Embedding تبدیل میکنه و توی ChromaDB ذخیره میکنه
    اگه قبلاً داده‌ای بوده، اول پاکش میکنه
    """
    import shutil

    # اگه پوشه ChromaDB قبلاً وجود داشت، پاکش میکنه
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("🗑️ داده‌های قبلی پاک شدن")

    print("⏳ در حال ساخت Embedding — کمی صبر کن...")

    # مدل Embedding رو لود میکنه
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    # توی ChromaDB ذخیره میکنه
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH
    )

    print(f"✅ همه chunk ها توی ChromaDB ذخیره شدن!")
    return vectorstore

def ingest(file_path):
    """
    تابع اصلی — همه مراحل رو یکی یکی انجام میده
    """
    print(f"\n📄 شروع پردازش فایل: {file_path}")
    print("=" * 50)
    
    # مرحله ۱ — خوندن فایل
    documents = load_document(file_path)
    
    # مرحله ۲ — تقسیم به chunk
    chunks = split_documents(documents)
    
    # مرحله ۳ — ذخیره توی ChromaDB
    save_to_chroma(chunks)
    
    print("=" * 50)
    print("🎉 فایل با موفقیت ایندکس شد!")


# اجرای مستقیم فایل
if __name__ == "__main__":
    # مسیر فایلت رو اینجا بذار
    file_path = "sample.txt"
    ingest(file_path)