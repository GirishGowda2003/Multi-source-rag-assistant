# 🧠 Multi-Source RAG Chatbot | ChromaDB + Groq + Streamlit
An end-to-end multi-source RAG application that enables intelligent question answering over web URLs, PDFs, CSV files, and research papers using ChromaDB, Groq LLMs, and Streamlit.

## 📌 Project Overview

The Multi-Source RAG Chatbot is an AI-powered application that enables users to upload, ingest, and query information from multiple data sources using Retrieval-Augmented Generation (RAG).

Unlike a basic chatbot that relies only on an LLM's pretrained knowledge, this system retrieves relevant information from user-provided documents and uses that context to generate answers.

The application supports multiple data sources, including PDF documents, web URLs, CSV files, and research papers.

It combines vector-based semantic search with large language models to provide context-aware responses grounded in retrieved content.

## 🎯 Objectives

- Build an end-to-end Retrieval-Augmented Generation system.
- Support multiple data ingestion methods.
- Extract and process information from different document formats.
- Generate embeddings for semantic search.
- Store and retrieve document chunks using ChromaDB.
- Integrate Groq-hosted LLMs for response generation.
- Develop an interactive interface using Streamlit.
- Improve response relevance by grounding answers in retrieved context.

## ✨ Key Features

- 📄 **PDF Ingestion:** Extract text from PDF documents and make it searchable.
- 🌐 **Web Ingestion:** Load content from web URLs for question answering.
- 📊 **CSV Ingestion:** Process tabular datasets and retrieve relevant information.
- 📚 **Research Paper Analysis:** Query research papers and retrieve relevant passages.
- 🔍 **Semantic Search:** Find relevant content using vector embeddings.
- 🧠 **Context-Aware Answers:** Generate answers using retrieved document context.
- 💬 **Interactive Chat Interface:** Ask questions through a Streamlit application.
- ⚡ **LLM Integration:** Use Groq-hosted language models for response generation.
- 📚 **Source References:** Display available document or URL references for retrieved information.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core application development |
| LangChain | Document processing and RAG orchestration |
| ChromaDB | Vector storage and similarity search |
| Groq API | LLM-powered answer generation |
| Streamlit | Interactive web interface |
| Sentence Transformers | Local text embedding generation |
| PyPDF | PDF text extraction |
| Pandas | CSV processing and tabular data handling |
| BeautifulSoup | Web content extraction |
| NumPy | Numerical operations |
| python-dotenv | Environment variable management |

*The technologies listed above describe the proposed implementation. Adjust the list to match your actual dependencies.*

## 🏗️ System Architecture

```text
                 ┌──────────────────────────┐
                 │       Streamlit UI       │
                 │ Upload / URL / CSV / PDF │
                 └─────────────┬────────────┘
                               │
                               ▼
                 ┌──────────────────────────┐
                 │      Data Ingestion      │
                 │                          │
                 │  PDF Documents           │
                 │  Web Pages               │
                 │  CSV Files               │
                 │  Research Papers         │
                 └─────────────┬────────────┘
                               │
                               ▼
                 ┌──────────────────────────┐
                 │  Extraction & Processing │
                 │                          │
                 │ Text Extraction          │
                 │ Cleaning                 │
                 │ Chunking                 │
                 │ Metadata Enrichment      │
                 └─────────────┬────────────┘
                               │
                               ▼
                 ┌──────────────────────────┐
                 │    Embedding Model       │
                 │  Text → Vector Embedding │
                 └─────────────┬────────────┘
                               │
                               ▼
                 ┌──────────────────────────┐
                 │        ChromaDB          │
                 │ Vector Storage & Search  │
                 └─────────────┬────────────┘
                               │
                      User Question
                               │
                               ▼
                 ┌──────────────────────────┐
                 │   Similarity Retrieval   │
                 │  Relevant Document Chunks│
                 └─────────────┬────────────┘
                               │
                               ▼
                 ┌──────────────────────────┐
                 │       Groq LLM           │
                 │ Context + User Question  │
                 └─────────────┬────────────┘
                               │
                               ▼
                 ┌──────────────────────────┐
                 │   Contextual Answer      │
                 │   + Source References    │
                 └──────────────────────────┘
```

## ⚙️ How It Works

### 1. Data Ingestion

Users provide information through supported sources:

- PDF files
- Web URLs
- CSV files
- Research papers in supported document formats

Each source is processed using an appropriate extraction method.

### 2. Text Extraction and Preprocessing

The system extracts content from the selected source and prepares it for retrieval.

Typical preprocessing operations include:

- Removing unnecessary whitespace.
- Handling empty or invalid content.
- Normalizing extracted text where appropriate.
- Preserving useful metadata.
- Separating long documents into manageable chunks.

### 3. Document Chunking

Large documents are divided into smaller text segments.

Chunking helps the retrieval system identify relevant passages without passing entire documents to the LLM.

Chunk size and overlap should be configured according to the document type and embedding model.

### 4. Embedding Generation

An embedding model converts text chunks into numerical vectors.

These vectors represent semantic information about the text and enable similarity-based retrieval.

### 5. Vector Storage Using ChromaDB

ChromaDB stores the generated embeddings alongside their corresponding text chunks and metadata.

Metadata can include:

- Source type
- Document name
- Web URL
- Page number
- Chunk identifier

This information can be used to filter search results and display source references.

### 6. Semantic Retrieval

When a user asks a question, the system converts the query into an embedding and searches ChromaDB for relevant chunks.

The retrieved passages provide context for generating an answer.

### 7. Answer Generation Using Groq

The retrieved context and user question are passed to a Groq-hosted LLM.

The model generates an answer grounded in the available context.

The application can instruct the model to acknowledge when the retrieved information is insufficient rather than inventing unsupported facts.

### 8. Response Display

Streamlit displays the generated response and, where available, the sources used to answer the question.

## 🔄 Supported Data Sources

| Data Source | Processing Approach | Example Use Case |
|---|---|---|
| PDF | Extract text and split into chunks | Ask questions about reports |
| Web URLs | Extract readable page content | Research information from websites |
| CSV | Process rows and columns | Explore structured datasets |
| Research Papers | Extract paper text and metadata | Understand methodology and findings |

### Important CSV Consideration

CSV files contain structured data, so their processing may differ from ordinary documents.

For simple question answering, rows can be converted into text representations and indexed.

For numerical analysis, aggregations, filtering, and exact calculations, a structured query or Pandas-based processing layer is often more reliable than semantic retrieval alone.

## 📂 Project Structure

```text
multi-source-rag-chatbot/
│
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
│
├── ingestion/
│   ├── pdf_loader.py
│   ├── web_loader.py
│   ├── csv_loader.py
│   └── research_paper_loader.py
│
├── rag/
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   └── chain.py
│
├── utils/
│   ├── text_processing.py
│   └── metadata.py
│
├── data/
│   └── .gitkeep
│
└── chroma_db/
    └── .gitkeep
```

*This is a suggested structure. Adapt it to your actual implementation.*

## 🚀 Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/multi-source-rag-chatbot.git
```

### Step 2: Navigate to the Project

```bash
cd multi-source-rag-chatbot
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Configure Environment Variables

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_groq_api_key
```

Obtain an API key through the official Groq developer platform.

Never commit your actual API key to GitHub.

### Step 6: Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

## 📦 Requirements

An example `requirements.txt` for an implementation using these libraries:

```text
streamlit
langchain
langchain-community
langchain-groq
langchain-chroma
chromadb
sentence-transformers
pypdf
pandas
beautifulsoup4
python-dotenv
```

Depending on the APIs and classes used, additional or different packages may be necessary. Pin tested versions before deploying the application.

## 🔐 Security Considerations

- Store API keys in environment variables.
- Do not upload private documents or confidential datasets to a public repository.
- Validate uploaded file types and enforce file-size limits.
- Treat retrieved documents and web content as untrusted input.
- Do not follow instructions found inside retrieved documents as system instructions.
- Restrict web ingestion to safe, permitted URLs and protect against server-side request forgery.
- Avoid exposing sensitive document content through logs or source previews.

## 📈 Evaluation

The RAG pipeline can be evaluated across several dimensions.

| Evaluation Area | What to Measure |
|---|---|
| Retrieval quality | Whether relevant chunks are retrieved |
| Answer correctness | Whether the generated answer is factually correct |
| Faithfulness | Whether claims are supported by retrieved context |
| Context relevance | Whether retrieved passages help answer the question |
| Latency | Time required to retrieve and generate an answer |
| Source accuracy | Whether displayed references support the answer |

Evaluation should use a representative set of questions with known answers and relevant source documents.

## 🔮 Future Enhancements

- Add hybrid search combining keyword and vector retrieval.
- Implement reranking to improve retrieval relevance.
- Support multiple document collections.
- Add document-level access controls.
- Implement conversation memory.
- Add streaming responses.
- Introduce automated RAG evaluation.
- Support OCR for scanned PDFs.
- Improve structured CSV querying.
- Add citation verification.
- Deploy the application to a cloud platform.

## 🌍 Applications

- Research paper analysis
- Document question answering
- Business report analysis
- Knowledge-base assistants
- Web content research
- Dataset exploration
- Academic research support

## 👨‍💻 Author

**Your Name**

Aspiring AI / Machine Learning Engineer

GitHub: https://github.com/YOUR_USERNAME

## 📄 License

Choose an appropriate open-source license for your code and verify that any datasets, documents, and model components can be used and redistributed under their applicable terms.

---

⭐ If you find this project useful, consider giving the repository a star!
