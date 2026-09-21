# 🧠 Feedback-to-Feature Studio

> An AI-powered product feedback analysis system that transforms raw customer feedback into structured, actionable product feature proposals.

---

## Project Purpose

Product teams collect feedback from many sources — surveys, app reviews, support tickets, and emails. Manually reading hundreds of feedback entries to identify trends is time-consuming and error-prone.

**Feedback-to-Feature Studio** automates this process using:
- **Semantic embeddings** to understand the meaning of feedback, not just keywords.
- **Clustering** to automatically group similar feedback into themes.
- **Retrieval-Augmented Generation (RAG)** to ground AI output in real product knowledge.
- **LLM (Large Language Model)** to generate structured feature proposals from each theme.

---

## Planned Technologies

| Layer            | Technology                              |
|------------------|-----------------------------------------|
| UI / App         | Python · Streamlit                      |
| Data             | pandas                                  |
| Embeddings       | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector Store     | FAISS or ChromaDB                       |
| Clustering       | scikit-learn (KMeans)                   |
| LLM              | Google Gemini API                       |
| Environment      | python-dotenv                           |

---

## Current Status — Phase 1: Project Foundation ✅

- [x] Project folder structure created
- [x] Demo feedback dataset (`data/feedback.csv`) — 38 records, 7 themes
- [x] Product knowledge base (`data/knowledge_base.txt`) — fictional Nexora Analytics
- [x] Module scaffolds with documented placeholders:
  - `modules/preprocessing.py`
  - `modules/embeddings.py`
  - `modules/clustering.py`
  - `modules/rag.py`
  - `modules/llm.py`
  - `modules/feature_generator.py`
- [x] Shared utilities (`utils/helpers.py`)
- [x] Minimal Streamlit app shell (`app.py`)
- [x] `requirements.txt`, `.env.example`, `.gitignore`, `README.md`

---

## Planned Pipeline

```
Customer Feedback (CSV)
        ↓
  Preprocessing        ← Clean & normalize text
        ↓
  Embeddings           ← Convert feedback to semantic vectors
        ↓
  Clustering           ← Group similar feedback into themes
        ↓
  RAG                  ← Retrieve relevant product knowledge
        ↓
  LLM                  ← Generate AI-powered insights
        ↓
  Feature Proposal     ← Pain points · User stories · Acceptance criteria · Priority
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd Feedback-to-Feature-Studio
```

### 2. Create a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

```bash
# Copy the example file and fill in your API keys
cp .env.example .env
```

Open `.env` and replace `your_api_key_here` with your actual Gemini API key.

### 5. Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Project Structure

```
Feedback-to-Feature-Studio/
│
├── app.py                      ← Streamlit application entry point
│
├── data/
│   ├── feedback.csv            ← Demo feedback dataset (38 records)
│   └── knowledge_base.txt      ← Fictional product knowledge base
│
├── modules/
│   ├── __init__.py
│   ├── preprocessing.py        ← Text cleaning & normalization
│   ├── embeddings.py           ← Semantic embedding generation
│   ├── clustering.py           ← KMeans-based feedback clustering
│   ├── rag.py                  ← Knowledge retrieval (RAG)
│   ├── llm.py                  ← LLM API communication
│   └── feature_generator.py   ← Feature proposal orchestration
│
├── utils/
│   ├── __init__.py
│   └── helpers.py              ← Shared utility functions
│
├── .env.example                ← Environment variable template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## License

This project is for educational and demonstration purposes.
