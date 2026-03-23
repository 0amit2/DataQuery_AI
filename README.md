#  DataQuery AI  
### RAG-based Data Science Query Assistant using Gemini

DataQuery AI is an end-to-end Retrieval-Augmented Generation (RAG) application that allows users to ask questions from a PDF document and get accurate, context-aware answers using Google Gemini LLM.

This project demonstrates practical implementation of LLMs + Vector Databases + NLP pipelines in a real-world scenario.

---

##  Project Overview

DataQuery AI enables users to:
- Ask questions in natural language  
- Retrieve relevant information from a PDF dataset  
- Generate precise answers using Gemini  

Unlike traditional chatbots, this system:
- Uses retrieval-based grounding  
- Avoids hallucination  
- Ensures context-based responses only  

---

##  Key Features

- Built using Retrieval-Augmented Generation (RAG)  
- Uses Google Gemini (gemini-2.5-flash)  
- PDF-based knowledge extraction  
- Context-aware answer generation  
- Custom prompt engineering for controlled responses  
- Interactive Streamlit UI  
- No hallucination (answers strictly from context)  

---

##  System Architecture

```
User Query
   ↓
Streamlit UI
   ↓
Text Embedding (MiniLM)
   ↓
Vector Database (Chroma)
   ↓
Retriever (Top-K relevant chunks)
   ↓
Prompt Template
   ↓
Gemini LLM
   ↓
Final Answer
```

---

##  Tech Stack

- Language: Python  
- Frontend: Streamlit  
- LLM: Google Gemini (gemini-2.5-flash)  
- Embeddings: all-MiniLM-L6-v2  
- Vector DB: Chroma  
- Framework: LangChain  

---

##  Project Structure

```
project/
│── app.py               # Main Streamlit application
│── Data.pdf             # Input knowledge base
│── requirements.txt     # Dependencies
│── README.md            # Documentation
```

---

##  How It Works

### 1. Document Loading
- PDF is loaded using PyPDFLoader  

### 2. Text Splitting
- Chunk size: 800  
- Overlap: 200  

### 3. Embedding Generation
- Using all-MiniLM-L6-v2  

### 4. Vector Storage
- Stored in Chroma DB  

### 5. Retrieval
- Top 10 relevant chunks retrieved (k=10)  

### 6. LLM Response
- Gemini generates answers using retrieved context and prompt  

---

