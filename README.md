# 🎓 UniVerse-PK: One-Stop AI Chatbot for Pakistani University Admissions
> **Instant, accurate university admissions guidance for Pakistani students — powered by RAG + LLaMA 3.3 70B**


**UniVerse-PK** is a Retrieval-Augmented Generation (RAG) based AI chatbot that centralizes verified information from Pakistan's top universities and delivers instant, accurate answers in a conversational interface. Students can compare universities, check eligibility, and explore scholarship options — all in one place.

### Currently Covered Universities
| University | Type | Location |
|---|---|---|
| COMSATS University Islamabad (CUI) | Public Federal | Islamabad + 6 campuses |
| NUST | Public Federal | Islamabad |
| UET Lahore | Public Provincial | Lahore |
| QAU (Quaid-i-Azam University) | Public Federal | Islamabad |

---

## 🏗️ Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                        │
│                    Gradio ChatInterface                       │
│              (Deployed on HuggingFace Spaces)                │
└─────────────────────┬───────────────────────────────────────┘
                      │ User Query
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                     RAG PIPELINE                             │
│                                                              │
│  1. RETRIEVAL                                                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Query → HuggingFace Embeddings                     │    │
│  │  (sentence-transformers/all-MiniLM-L6-v2)           │    │
│  │         ↓                                           │    │
│  │  FAISS Vector Store ← similarity_search(k=5)        │    │
│  │  (Persisted locally as faiss_index/)                │    │
│  └─────────────────────────────────────────────────────┘    │
│                      │ Top-5 relevant chunks                 │
│                      ▼                                       │
│  2. AUGMENTATION                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Context (retrieved chunks) + User Query            │    │
│  │  → Structured Prompt Engineering                    │    │
│  └─────────────────────────────────────────────────────┘    │
│                      │ Augmented prompt                      │
│                      ▼                                       │
│  3. GENERATION                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Groq API → LLaMA 3.3 70B Versatile                 │    │
│  │  (temperature=0.3, max_tokens=1024)                  │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────┬───────────────────────────────────────┘
                      │ Generated Answer
                      ▼
              Response to User
```

---

## ⚙️ RAG Workflow (Step-by-Step)

```
Step 1: KNOWLEDGE BASE PREPARATION
        Curated Documents (COMSATS, NUST, UET, QAU)
              ↓
        RecursiveCharacterTextSplitter
        (chunk_size=600, chunk_overlap=80)
              ↓
        HuggingFace Embeddings → Dense Vectors
              ↓
        FAISS Index (saved to disk)

Step 2: QUERY PROCESSING (at runtime)
        User Query → Embed with same model
              ↓
        FAISS similarity_search → Top 5 chunks

Step 3: RESPONSE GENERATION
        [System Prompt + Context + User Query]
              ↓
        Groq API (LLaMA 3.3 70B)
              ↓
        Structured Answer → Gradio UI
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **LLM** | LLaMA 3.3 70B Versatile via Groq API |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector Store** | FAISS (Facebook AI Similarity Search) |
| **RAG Framework** | LangChain |
| **UI** | Gradio `ChatInterface` |
| **Deployment** | HuggingFace Spaces |
| **Language** | Python 3.10+ |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/UniVerse-PK.git
cd UniVerse-PK
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables
```bash
export GROQ_API_KEY="your_groq_api_key_here"
export GROQ_MODEL="llama-3.3-70b-versatile"   # optional, this is the default
```

### 4. Run Locally
```bash
python app.py
```

---

## ☁️ Deployment on HuggingFace Spaces

1. Create a new **Gradio** Space on [huggingface.co/spaces](https://huggingface.co/spaces)
2. Upload `app.py` and `requirements.txt`
3. Add your `GROQ_API_KEY` under **Settings → Repository Secrets**
4. The Space will build and deploy automatically

> ⚠️ Never hardcode your API key. Always use HuggingFace Secrets for deployment.

---

## 📊 Knowledge Base Coverage

| Topic | Covered |
|---|---|
| BS/BE Undergraduate Admissions | ✅ |
| MS/MPhil Graduate Admissions | ✅ |
| PhD Admissions & Eligibility | ✅ |
| Fee Structures (per semester) | ✅ |
| Entry Test Requirements | ✅ |
| Scholarship Information | ✅ |
| University Comparisons | ✅ |

---

## ⚠️ Disclaimer

Information in this chatbot is curated for educational guidance purposes. Always verify the latest admission details, deadlines, and fee structures on the **official university websites** before applying.

---

## 🤝 Contributing

Contributions are welcome! To add more universities or update information:
1. Fork the repository
2. Add new `Document` entries to the `KNOWLEDGE_BASE` list in `app.py`
3. Submit a Pull Request

---


---

<p align="center">Built with ❤️ for Pakistani students · Powered by RAG + LLaMA 3.3</p>
