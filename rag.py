import os
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from groq import Groq
from config import (
    EMBEDDING_MODEL, CHROMA_PATH, COLLECTION_NAME,
    GROQ_MODEL, GROQ_API_KEY, MAX_TOKENS, TEMPERATURE,
    TOP_K, FETCH_K, MMR_LAMBDA, ENABLE_TRANSLATION,
    SCORE_THRESHOLD, SHOW_SCORES
)

# ✅ مدل فقط یک بار لود میشه — نه هر بار!
_embedding_model = None
_vectorstore = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print("🔄 لود مدل Embedding...")
        _embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        print("✅ مدل لود شد!")
    return _embedding_model

def load_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embedding_model(),
            persist_directory=CHROMA_PATH
        )
        print("✅ پایگاه داده لود شد")
    return _vectorstore

def calculate_score(distance):
    """
    L2 distance به درصد تطابق
    distance کم = تطابق زیاد
    """
    score = round(100 / (1 + distance), 1)
    return score

def detect_and_translate(question, client):
    """تشخیص زبان و ترجمه"""
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{
                "role": "user",
                "content": f"""Detect language and translate to English.
Question: {question}
Respond ONLY as JSON: {{"original": "...", "english": "...", "language": "..."}}"""
            }],
            max_tokens=150,
            temperature=0
        )
        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)
        print(f"🌐 زبان: {result.get('language', '?')}")
        print(f"🔄 ترجمه: {result.get('english', question)}")
        return result
    except Exception as e:
        print(f"⚠️ خطا در ترجمه: {e}")
        return {"original": question, "english": question, "language": "unknown"}

def search_documents(vectorstore, query, query_english=None, k=TOP_K):
    """جستجوی چندزبانه با MMR + Score"""
    
    all_results = []
    seen_ids = set()

    # جستجوی اصلی با MMR
    results1 = vectorstore.max_marginal_relevance_search(
        query, k=k, fetch_k=FETCH_K, lambda_mult=MMR_LAMBDA
    )
    for doc in results1:
        doc_id = doc.metadata.get("source", "") + str(doc.metadata.get("page", ""))
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            all_results.append(doc)

    # جستجو با ترجمه انگلیسی
    if query_english and query_english.lower() != query.lower():
        results2 = vectorstore.max_marginal_relevance_search(
            query_english, k=k, fetch_k=FETCH_K, lambda_mult=MMR_LAMBDA
        )
        for doc in results2:
            doc_id = doc.metadata.get("source", "") + str(doc.metadata.get("page", ""))
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_results.append(doc)

    # محاسبه score — از همون مدل که قبلاً لود شده
    scores_map = {}
    try:
        emb_model = get_embedding_model()  # ✅ از cache استفاده میکنه
        query_embedding = emb_model.embed_query(query)
        
        chroma_collection = vectorstore._collection
        count = chroma_collection.count()
        
        if count > 0:
            raw = chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=min(FETCH_K, count),
                include=["distances", "metadatas"]
            )
            
            if raw and raw.get("distances"):
                for i, meta in enumerate(raw["metadatas"][0]):
                    if meta and i < len(raw["distances"][0]):
                        doc_id = meta.get("source", "") + str(meta.get("page", ""))
                        d = raw["distances"][0][i]
                        score = calculate_score(d)
                        scores_map[doc_id] = score
                        print(f"📊 Distance: {d:.3f} → Score: {score}%")
                        
    except Exception as e:
        print(f"⚠️ خطا در score: {e}")

    all_results = all_results[:k]
    if not all_results:
        return "", []

    context_parts = []
    sources = []
    seen_sources = set()

    for doc in all_results:
        context_parts.append(doc.page_content)
        source = doc.metadata.get("source", "نامشخص")
        source = source.replace("\\", "/").split("/")[-1]
        if source.startswith("temp_"):
            source = source[5:]
        page = doc.metadata.get("page", "نامشخص")
        key = f"{source}-{page}"
        doc_id = doc.metadata.get("source", "") + str(doc.metadata.get("page", ""))

        if key not in seen_sources:
            seen_sources.add(key)
            sources.append({
                "file": source,
                "page": page,
                "content": doc.page_content,
                "score": scores_map.get(doc_id, None)
            })

    context = "\n\n".join(context_parts)
    print(f"✅ {len(all_results)} بخش پیدا شد")
    return context, sources

def ask_groq(context, question, client):
    prompt = f"""You are a helpful assistant. Answer based ONLY on the context below.
Answer in the SAME language as the question.
Give detailed answer with examples from context.

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE
    )
    return response.choices[0].message.content

def ask(question):
    print(f"\n❓ سوال: {question}")
    print("=" * 50)

    client = Groq(api_key=GROQ_API_KEY)

    if ENABLE_TRANSLATION:
        translation = detect_and_translate(question, client)
        query_english = translation.get("english", question)
    else:
        query_english = question

    vectorstore = load_vectorstore()
    context, sources = search_documents(vectorstore, query=question, query_english=query_english)

    if not context:
        return "⚠️ هیچ اطلاعات مرتبطی پیدا نشد."

    answer = ask_groq(context, question, client)

    source_text = "\n\n---\n📚 **منابع و متن مرتبط:**\n"

    for i, s in enumerate(sources):
        score = s.get("score", None)

        if score is None:
            score_emoji = "⚪"
            score_text = ""
        elif score >= 15:
            score_emoji = "🟢"
            score_text = f"({score}% — تطابق بالا)"
        elif score >= 10:
            score_emoji = "🟡"
            score_text = f"({score}% — تطابق متوسط)"
        else:
            score_emoji = "🔴"
            score_text = f"({score}% — تطابق پایین)"

        if s["page"] != "نامشخص":
            page_num = int(s["page"]) + 1
            source_text += f"\n**{i+1}. 📖 {s['file']} — صفحه {page_num}** {score_emoji} *{score_text}*\n"
        else:
            source_text += f"\n**{i+1}. 📖 {s['file']}** {score_emoji} *{score_text}*\n"

        if s.get("content"):
            content = s["content"].strip()
            if len(content) > 500:
                content = content[:500] + "..."
            source_text += f"> *{content}*\n"

    print("=" * 50)
    return answer + source_text
