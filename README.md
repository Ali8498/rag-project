<div align="center">

# 🤖 RAG System — Intelligent Question Answering
# سیستم پرسش و پاسخ هوشمند

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-latest-green)](https://langchain.com)
[![Groq](https://img.shields.io/badge/Groq-LLaMA3.3-orange)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

</div>

---

## 🇬🇧 English

### 📌 Description
A powerful **RAG (Retrieval Augmented Generation)** system that lets you 
chat intelligently with your own documents using LangChain, ChromaDB, 
and Groq LLM.

### ✨ Features
- 📄 Load PDF and TXT files
- 🔍 Semantic Search across documents
- 🤖 Intelligent answers powered by Groq (LLaMA 3.3)
- 📚 Support for 100+ documents simultaneously
- 🎯 Source citation — book name and page number
- 💾 Persistent vector database with ChromaDB

### 🛠️ Tech Stack
| Tool | Purpose |
|------|---------|
| Python 3.13 | Core language |
| LangChain | RAG framework |
| ChromaDB | Vector database |
| Sentence Transformers | Embedding model |
| Groq (LLaMA 3.3) | Language model |
| Streamlit | Web interface |

### 🚀 Getting Started

#### 1. Clone the repository
```bash
git clone https://github.com/Ali8498/rag-project.git
cd rag-project
```

#### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

#### 3. Install dependencies
```bash
pip install -r requirements.txt
```

#### 4. Set up environment variables
Create a `.env` file:
```
GROQ_API_KEY=your_groq_token_here
```

#### 5. Index your documents
```bash
python ingest.py
```

#### 6. Start asking questions
```bash
python rag.py
```

### 🔮 Roadmap
- [x] Basic RAG system
- [x] PDF and TXT support
- [x] GitHub repository
- [ ] Streamlit web interface
- [ ] Support for 100+ PDF files
- [ ] Source citation (book + page number)
- [ ] Docker containerization
- [ ] CI/CD pipeline

---

## 🇮🇷 فارسی

### 📌 توضیحات
یک سیستم قدرتمند **RAG** که به شما امکان می‌دهد با اسناد خودتان 
به صورت هوشمند گفتگو کنید. با استفاده از LangChain، ChromaDB و Groq 
ساخته شده است.

### ✨ قابلیت‌ها
- 📄 بارگذاری فایل‌های PDF و TXT
- 🔍 جستجوی معنایی در اسناد
- 🤖 پاسخ هوشمند با Groq (LLaMA 3.3)
- 📚 پشتیبانی از ۱۰۰+ سند همزمان
- 🎯 ذکر منبع — نام کتاب و شماره صفحه
- 💾 پایگاه داده برداری با ChromaDB

### 🚀 نصب و راه‌اندازی

#### ۱. کلون کردن پروژه
```bash
git clone https://github.com/Ali8498/rag-project.git
cd rag-project
```

#### ۲. ساخت محیط مجازی
```bash
python -m venv venv
venv\Scripts\activate
```

#### ۳. نصب کتابخانه‌ها
```bash
pip install -r requirements.txt
```

#### ۴. تنظیم توکن Groq
فایل `.env` بساز:
```
GROQ_API_KEY=your_groq_token_here
```

#### ۵. ایندکس کردن اسناد
```bash
python ingest.py
```

#### ۶. پرسش و پاسخ
```bash
python rag.py
```

---

## 📁 Project Structure / ساختار پروژه
```
rag-project/
├── config.py          # تنظیمات / Configuration
├── ingest.py          # بارگذاری اسناد / Document indexing
├── rag.py             # پرسش و پاسخ / Question answering
├── requirements.txt   # کتابخانه‌ها / Dependencies
├── sample.txt         # نمونه سند / Sample document
└── .gitignore         # فایل‌های نادیده / Git ignore
```

---

<div align="center">

## 👨‍💻 Developer / توسعه‌دهنده

**Ali Pornouri** — [@Ali8498](https://github.com/Ali8498)

---

🇮🇷 Made with ❤️ in Italy 🇮🇹

⭐ If you find this useful, please give it a star!
⭐ اگه مفید بود، ستاره بده!

</div>
```

---

