import streamlit as st
import os
import time
from ingest import ingest, get_existing_sources, delete_document, vacuum_database
from rag import ask

# ✅ cache کردن مدل برای جلوگیری از لود مجدد
@st.cache_resource
def load_rag_model():
    from rag import load_vectorstore, get_embedding_model
    get_embedding_model()
    load_vectorstore()
    print("✅ مدل‌ها cache شدن!")

load_rag_model()
import streamlit as st
import os
import time
from ingest import ingest, get_existing_sources, delete_document, vacuum_database
from rag import ask

st.set_page_config(
    page_title="RAG System | سیستم هوشمند",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = list(get_existing_sources())

with st.sidebar:
    lang = st.selectbox("🌐 Language / زبان", ["فارسی", "English"])

is_fa = lang == "فارسی"
direction = "rtl" if is_fa else "ltr"
align = "right" if is_fa else "left"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;700&display=swap');

* {{
    font-family: 'Vazirmatn', Tahoma, sans-serif !important;
}}

.stApp {{
    background: linear-gradient(135deg, #0a0a1a 0%, #1a0533 50%, #0a1a2e 100%);
}}

.block-container {{
    direction: {direction};
    text-align: {align};
    padding-bottom: 100px !important;
}}

section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #1a0533 0%, #0a1a2e 100%);
    border-right: 1px solid rgba(123,47,247,0.3);
}}

.main-title {{
    text-align: center;
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00d2ff, #7b2ff7, #ff6b6b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    padding: 20px 0 5px 0;
    direction: {direction};
}}

.subtitle {{
    text-align: center;
    color: rgba(255,255,255,0.5);
    font-size: 1rem;
    margin-bottom: 20px;
    direction: {direction};
}}

.card {{
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(123,47,247,0.2);
    border-radius: 20px;
    padding: 24px;
    margin-bottom: 16px;
    direction: {direction};
    text-align: {align};
}}

.stButton > button {{
    background: linear-gradient(90deg, #7b2ff7, #00d2ff) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    width: 100% !important;
    transition: all 0.3s !important;
}}

.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(123,47,247,0.4) !important;
}}

.doc-item {{
    background: rgba(123,47,247,0.1);
    border: 1px solid rgba(123,47,247,0.25);
    border-radius: 10px;
    padding: 10px 16px;
    margin: 6px 0;
    color: rgba(255,255,255,0.85);
    font-size: 0.9rem;
    direction: {direction};
    text-align: {align};
}}

.step-item {{
    background: rgba(0,210,255,0.05);
    border-right: 3px solid #00d2ff;
    border-radius: 8px;
    padding: 10px 16px;
    margin: 8px 0;
    color: rgba(255,255,255,0.85);
    direction: {direction};
    text-align: {align};
}}

.stat-box {{
    background: linear-gradient(135deg, rgba(0,210,255,0.1), rgba(123,47,247,0.1));
    border: 1px solid rgba(0,210,255,0.3);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    margin: 12px 0;
}}

.stat-number {{
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00d2ff, #7b2ff7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

.stat-label {{
    color: rgba(255,255,255,0.6);
    font-size: 0.9rem;
    margin-top: 4px;
}}

.section-header {{
    font-size: 1.2rem;
    font-weight: 700;
    color: white;
    margin-bottom: 16px;
    direction: {direction};
    text-align: {align};
}}

.divider {{
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(123,47,247,0.5), transparent);
    margin: 20px 0;
}}

.stChatMessage {{
    background: rgba(255,255,255,0.04) !important;
    border-radius: 16px !important;
    margin: 8px 0 !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    direction: {direction} !important;
}}

.stChatMessage p {{
    direction: {direction} !important;
    text-align: {align} !important;
}}

label {{
    direction: {direction} !important;
    text-align: {align} !important;
    color: rgba(255,255,255,0.7) !important;
}}

h1, h2, h3 {{
    color: white !important;
    direction: {direction} !important;
}}

.stSuccess {{
    background: rgba(0,255,100,0.08) !important;
    border: 1px solid rgba(0,255,100,0.25) !important;
    border-radius: 10px !important;
}}

.stError {{
    background: rgba(255,50,50,0.08) !important;
    border: 1px solid rgba(255,50,50,0.25) !important;
    border-radius: 10px !important;
}}

.stWarning {{
    background: rgba(255,165,0,0.08) !important;
    border: 1px solid rgba(255,165,0,0.25) !important;
    border-radius: 10px !important;
}}

div[data-testid="stChatInput"] textarea {{
    direction: {direction} !important;
    text-align: {align} !important;
}}

.stChatFloatingInputContainer {{
    position: fixed !important;
    bottom: 0 !important;
    left: 0 !important;
    right: 0 !important;
    z-index: 1000 !important;
    background: rgba(10, 10, 26, 0.97) !important;
    padding: 12px 24px 16px 24px !important;
    border-top: 1px solid rgba(123,47,247,0.3) !important;
    backdrop-filter: blur(20px) !important;
}}

section.main > div {{
    padding-bottom: 120px !important;
}}
</style>
""", unsafe_allow_html=True)

if is_fa:
    t = {
        "title": "🤖 سیستم پرسش و پاسخ هوشمند",
        "subtitle": "اسناد خود را آپلود کنید و سوال بپرسید",
        "upload_header": "📄 بارگذاری اسناد",
        "upload_label": "فایل PDF یا TXT انتخاب کنید",
        "index_btn": "⚡ ایندکس کردن اسناد",
        "processing": "در حال پردازش",
        "success": "با موفقیت ایندکس شد!",
        "duplicate_hash": "محتوای این فایل قبلاً بارگذاری شده!",
        "duplicate_name": "این فایل قبلاً بارگذاری شده!",
        "guide_header": "📊 راهنمای استفاده",
        "steps": [
            "📤 فایل PDF یا TXT آپلود کن",
            "⚡ روی ایندکس کلیک کن",
            "💬 سوال خود را بنویس",
            "✅ جواب دقیق + منبع ببین"
        ],
        "chat_header": "💬 پرسش و پاسخ",
        "chat_input": "سوال خود را اینجا بنویسید...",
        "thinking": "در حال پردازش سوال...",
        "docs_header": "📚 اسناد بارگذاری شده",
        "total_docs": "تعداد اسناد",
        "clear_chat": "🗑️ پاک کردن چت",
        "delete_doc": "حذف سند",
        "delete_confirm": "سند حذف شد!",
        "about_title": "درباره سیستم",
        "about_text": "این سیستم با RAG پاسخ دقیق از اسناد شما میده.",
        "tech": "تکنولوژی‌ها",
        "chunks": "chunk پردازش شد",
        "settings": "تنظیمات",
        "stop_app": "🔴 بستن برنامه",
        "stopping": "برنامه در حال بسته شدن...",
        "no_docs_warning": "⚠️ ابتدا یک سند ایندکس کنید!",
        "db_management": "مدیریت بانک",
        "vacuum_btn": "🔧 بهینه‌سازی بانک",
        "vacuuming": "در حال بهینه‌سازی...",
        "vacuum_success": "بانک بهینه شد",
        "vacuum_error": "خطا در بهینه‌سازی",
        "db_stats": "آمار بانک اطلاعاتی",
        "total_chunks": "تعداد کل chunk",
    }
else:
    t = {
        "title": "🤖 Intelligent Q&A System",
        "subtitle": "Upload your documents and ask questions",
        "upload_header": "📄 Upload Documents",
        "upload_label": "Choose PDF or TXT files",
        "index_btn": "⚡ Index Documents",
        "processing": "Processing",
        "success": "Indexed successfully!",
        "duplicate_hash": "This content was already indexed!",
        "duplicate_name": "This file was already indexed!",
        "guide_header": "📊 How to Use",
        "steps": [
            "📤 Upload your PDF or TXT",
            "⚡ Click Index",
            "💬 Type your question",
            "✅ See answer + source"
        ],
        "chat_header": "💬 Chat",
        "chat_input": "Type your question here...",
        "thinking": "Processing your question...",
        "docs_header": "📚 Loaded Documents",
        "total_docs": "Total Documents",
        "clear_chat": "🗑️ Clear Chat",
        "delete_doc": "Delete",
        "delete_confirm": "Document deleted!",
        "about_title": "About",
        "about_text": "This system uses RAG to give accurate answers from your documents.",
        "tech": "Technologies",
        "chunks": "chunks processed",
        "settings": "Settings",
        "stop_app": "🔴 Stop Application",
        "stopping": "Shutting down...",
        "no_docs_warning": "⚠️ Please index a document first!",
        "db_management": "Database",
        "vacuum_btn": "🔧 Optimize Database",
        "vacuuming": "Optimizing...",
        "vacuum_success": "Database optimized",
        "vacuum_error": "Optimization failed",
        "db_stats": "Database Statistics",
        "total_chunks": "Total chunks",
    }

with st.sidebar:
    st.markdown("---")
    st.markdown(f'<div class="section-header">ℹ️ {t["about_title"]}</div>', unsafe_allow_html=True)
    st.caption(t["about_text"])
    st.markdown("---")
    st.markdown(f'<div class="section-header">🛠️ {t["tech"]}</div>', unsafe_allow_html=True)
    for tech in ["🧠 Groq LLaMA 3.3", "📦 ChromaDB", "🔗 LangChain", "🤗 Sentence Transformers"]:
        st.markdown(f'<div class="doc-item">{tech}</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f'<div class="section-header">🗄️ {t["db_management"]}</div>', unsafe_allow_html=True)

    if st.button(t["vacuum_btn"]):
        with st.spinner(t["vacuuming"]):
            result = vacuum_database()
            if result >= 0:
                st.success(f"✅ {t['vacuum_success']} — {result} chunk")
                st.session_state.indexed_docs = list(get_existing_sources())
                st.rerun()
            else:
                st.error("❌ " + t["vacuum_error"])

    st.markdown("---")
    st.markdown(f'<div class="section-header">⚙️ {t["settings"]}</div>', unsafe_allow_html=True)
    if st.button(t["stop_app"]):
        st.success("✅ " + t["stopping"])
        time.sleep(1)
        os.kill(os.getpid(), 9)

st.markdown(f'<div class="main-title">{t["title"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="subtitle">{t["subtitle"]}</div>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-header">{t["upload_header"]}</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        t["upload_label"],
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button(t["index_btn"], type="primary"):
            for uploaded_file in uploaded_files:
                temp_path = "temp_" + uploaded_file.name
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                try:
                    st.markdown(f"**📄 {uploaded_file.name}**")
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def update_progress(progress, done, total):
                        progress_bar.progress(progress)
                        status_text.markdown(
                            f"⚡ {t['processing']}: **{done}/{total}** {t['chunks']}"
                        )

                    result = ingest(temp_path, update_progress)

                    if result == "duplicate_hash":
                        progress_bar.empty()
                        status_text.empty()
                        st.warning("⚠️ " + uploaded_file.name + " — " + t["duplicate_hash"])
                    elif result == "duplicate_name":
                        progress_bar.empty()
                        status_text.empty()
                        st.warning("⚠️ " + uploaded_file.name + " — " + t["duplicate_name"])
                    else:
                        progress_bar.progress(1.0)
                        status_text.empty()
                        st.success("✅ " + uploaded_file.name + " — " + t["success"])
                        if uploaded_file.name not in st.session_state.indexed_docs:
                            st.session_state.indexed_docs.append(uploaded_file.name)
                except Exception as e:
                    st.error("❌ " + str(e))
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-header">{t["guide_header"]}</div>', unsafe_allow_html=True)
    for step in t["steps"]:
        st.markdown(f'<div class="step-item">{step}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.indexed_docs:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="section-header">{t["docs_header"]}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="stat-box">'
            f'<div class="stat-number">{len(st.session_state.indexed_docs)}</div>'
            f'<div class="stat-label">{t["total_docs"]}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        for doc in list(st.session_state.indexed_docs):
            col_doc, col_del = st.columns([4, 1])
            with col_doc:
                st.markdown(f'<div class="doc-item">📄 {doc}</div>', unsafe_allow_html=True)
            with col_del:
                if st.button("🗑️", key="del_" + doc, help=t["delete_doc"]):
                    deleted = delete_document(doc)
                    if deleted > 0:
                        st.session_state.indexed_docs.remove(doc)
                        st.success(t["delete_confirm"])
                        st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-header">{t["chat_header"]}</div>', unsafe_allow_html=True)

    if st.button(t["clear_chat"]):
        st.session_state.messages = []
        st.rerun()

    chat_container = st.container(height=500)
    with chat_container:
        if not st.session_state.messages:
            if st.session_state.indexed_docs:
                st.markdown(
                    f'<div style="text-align:center; color:rgba(255,255,255,0.3); '
                    f'padding:40px 0; direction:{direction};">'
                    f'{"💬 سوال خود را بنویسید..." if is_fa else "💬 Ask your first question..."}'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div style="text-align:center; color:rgba(255,165,0,0.7); '
                    f'padding:40px 0; direction:{direction};">'
                    f'{t["no_docs_warning"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    question = st.chat_input(t["chat_input"])
    if question:
        if not st.session_state.indexed_docs:
            st.warning(t["no_docs_warning"])
        else:
            st.session_state.messages.append({
                "role": "user",
                "content": question
            })
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(question)
                with st.chat_message("assistant"):
                    with st.spinner(t["thinking"]):
                        try:
                            answer = ask(question)
                            st.markdown(answer)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer
                            })
                        except Exception as e:
                            st.error("❌ " + str(e))

    st.markdown('</div>', unsafe_allow_html=True)