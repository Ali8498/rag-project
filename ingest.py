import os
import hashlib
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import (
    EMBEDDING_MODEL,
    CHROMA_PATH,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def get_vectorstore(embedding_model=None):
    if embedding_model is None:
        embedding_model = get_embedding_model()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=CHROMA_PATH
    )

def get_file_hash(file_path):
    """
    یه کد یکتا (Hash) از محتوای فایل میسازه
    حتی اگه اسم فایل عوض بشه، hash همونه
    حتی اگه یه بایت محتوا عوض بشه، hash فرق میکنه
    """
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def get_existing_hashes():
    """
    لیست hash همه فایل‌هایی که قبلاً ایندکس شدن رو برمیگردونه
    """
    try:
        vs = get_vectorstore()
        results = vs.get()
        hashes = set()
        if results and "metadatas" in results:
            for meta in results["metadatas"]:
                if meta and "file_hash" in meta:
                    hashes.add(meta["file_hash"])
        return hashes
    except:
        return set()

def get_existing_sources():
    """
    لیست اسم فایل‌هایی که قبلاً ایندکس شدن رو برمیگردونه
    """
    try:
        vs = get_vectorstore()
        results = vs.get()
        sources = set()
        if results and "metadatas" in results:
            for meta in results["metadatas"]:
                if meta and "source" in meta:
                    source = meta["source"].replace("\\", "/").split("/")[-1]
                    if source.startswith("temp_"):
                        source = source[5:]
                    sources.add(source)
        return sources
    except:
        return set()

def load_document(file_path):
    """فایل رو میخونه — PDF یا TXT"""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError("فقط PDF یا TXT قبول میشه!")
    documents = loader.load()
    print(f"✅ {len(documents)} صفحه خونده شد")
    return documents

def split_documents(documents, file_hash, file_name):
    """
    متن رو به chunk تقسیم میکنه
    hash فایل رو هم به metadata هر chunk اضافه میکنه
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    
    # اضافه کردن hash و اسم فایل به metadata هر chunk
    for chunk in chunks:
        chunk.metadata["file_hash"] = file_hash
        chunk.metadata["file_name"] = file_name
    
    print(f"✅ {len(chunks)} chunk ساخته شد")
    return chunks

def save_to_chroma(chunks, progress_callback=None):
    """chunk ها رو ذخیره میکنه با progress bar"""
    print("⏳ در حال ساخت Embedding...")
    embedding_model = get_embedding_model()
    vectorstore = get_vectorstore(embedding_model)
    
    total = len(chunks)
    batch_size = 10
    
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        vectorstore.add_documents(batch)
        if progress_callback:
            done = min(i + batch_size, total)
            progress_callback(done / total, done, total)
    
    print(f"✅ {total} chunk ذخیره شد!")
    return vectorstore

def delete_document(file_name):
    """یه سند خاص رو از ChromaDB حذف میکنه"""
    try:
        vs = get_vectorstore()
        results = vs.get()
        ids_to_delete = []
        
        if results and "metadatas" in results:
            for idx, meta in enumerate(results["metadatas"]):
                if meta and "source" in meta:
                    source = meta["source"].replace("\\", "/").split("/")[-1]
                    if source.startswith("temp_"):
                        source = source[5:]
                    if source == file_name:
                        ids_to_delete.append(results["ids"][idx])
        
        if ids_to_delete:
            vs.delete(ids=ids_to_delete)
            print(f"✅ {len(ids_to_delete)} chunk از {file_name} حذف شد")
            return len(ids_to_delete)
        return 0
    except Exception as e:
        print(f"❌ خطا در حذف: {e}")
        return 0

def ingest(file_path, progress_callback=None):
    """تابع اصلی — با بررسی hash برای جلوگیری از تکراری"""
    print(f"\n📄 پردازش: {file_path}")
    print("=" * 50)
    
    # اسم فایل رو تمیز میکنه
    file_name = os.path.basename(file_path)
    if file_name.startswith("temp_"):
        file_name = file_name[5:]
    
    # محاسبه hash فایل
    file_hash = get_file_hash(file_path)
    print(f"🔑 Hash فایل: {file_hash[:8]}...")
    
    # چک کردن hash — اگه همین محتوا قبلاً بوده
    existing_hashes = get_existing_hashes()
    if file_hash in existing_hashes:
        print(f"⚠️ این محتوا قبلاً ایندکس شده!")
        return "duplicate_hash"
    
    # چک کردن اسم — اگه همین اسم قبلاً بوده
    existing_sources = get_existing_sources()
    if file_name in existing_sources:
        print(f"⚠️ این اسم فایل قبلاً ایندکس شده!")
        return "duplicate_name"
    
    # ایندکس کردن
    documents = load_document(file_path)
    chunks = split_documents(documents, file_hash, file_name)
    save_to_chroma(chunks, progress_callback)
    
    print("=" * 50)
    print("🎉 ایندکس شد!")
    return "success"

def vacuum_database():
    """
    بهینه‌سازی و فشرده‌سازی بانک اطلاعاتی
    بعد از حذف‌های زیاد باید اجرا بشه
    مثل defragment کردن هارد دیسک
    """
    import shutil
    import tempfile
    
    print("🔄 شروع Vacuum بانک اطلاعاتی...")
    
    try:
        # قدم ۱ — گرفتن همه داده‌های معتبر
        vs = get_vectorstore()
        results = vs.get()
        
        if not results or not results.get("ids"):
            print("✅ بانک خالیه — نیازی به Vacuum نیست")
            return 0
        
        total_before = len(results["ids"])
        print(f"📊 تعداد chunk قبل از Vacuum: {total_before}")
        
        # قدم ۲ — ذخیره موقت داده‌ها
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        ids = results.get("ids", [])
        
        # قدم ۳ — پاک کردن بانک قدیمی
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
            print("🗑️ بانک قدیمی پاک شد")
        
        # قدم ۴ — ساخت بانک جدید با داده‌های تمیز
        embedding_model = get_embedding_model()
        new_vs = get_vectorstore(embedding_model)
        
        # قدم ۵ — اضافه کردن داده‌ها به بانک جدید
        if documents:
            batch_size = 50
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i+batch_size]
                batch_meta = metadatas[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                new_vs._collection.add(
                    documents=batch_docs,
                    metadatas=batch_meta,
                    ids=batch_ids
                )
        
        total_after = len(documents)
        print(f"✅ Vacuum کامل شد!")
        print(f"📊 تعداد chunk بعد از Vacuum: {total_after}")
        return total_after
        
    except Exception as e:
        print(f"❌ خطا در Vacuum: {e}")
        return -1