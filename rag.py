# فایل rag.py — سیستم پرسش و پاسخ

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from groq import Groq
from config import (
    EMBEDDING_MODEL,
    CHROMA_PATH,
    COLLECTION_NAME,
    GROQ_MODEL,
    GROQ_API_KEY
)

def load_vectorstore():
    """
    پایگاه داده ChromaDB رو لود میکنه
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )
    
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=CHROMA_PATH
    )
    
    print("✅ پایگاه داده لود شد")
    return vectorstore


def search_documents(vectorstore, query, k=3):
    """
    توی ChromaDB دنبال متن مرتبط با سوال میگرده
    k = تعداد chunk هایی که برمیگردونه
    """
    results = vectorstore.similarity_search(query, k=k)
    
    # متن chunk های پیدا شده رو کنار هم میذاره
    context = "\n\n".join([doc.page_content for doc in results])
    
    print(f"✅ {len(results)} بخش مرتبط پیدا شد")
    return context


def ask_groq(context, question):
    """
    سوال + متن مرتبط رو به Groq میفرسته و جواب میگیره
    """
    client = Groq(api_key=GROQ_API_KEY)
    
    # prompt رو میسازه
    prompt = f"""فقط با استفاده از اطلاعات زیر به سوال جواب بده.
اگر جواب توی اطلاعات نبود بگو: "این اطلاعات در اسناد من نیست."

اطلاعات:
{context}

سوال: {question}

جواب:"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content


def ask(question):
    """
    تابع اصلی — همه مراحل رو یکی یکی انجام میده
    """
    print(f"\n❓ سوال: {question}")
    print("=" * 50)
    
    # مرحله ۱ — لود کردن ChromaDB
    vectorstore = load_vectorstore()
    
    # مرحله ۲ — جستجوی متن مرتبط
    context = search_documents(vectorstore, question)
    
    # مرحله ۳ — گرفتن جواب از Groq
    answer = ask_groq(context, question)
    
    print(f"\n💬 جواب: {answer}")
    print("=" * 50)
    
    return answer


# اجرای مستقیم
if __name__ == "__main__":
    ask("آدرس شرکت کجاست؟")