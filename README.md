# Local Document RAG System

A powerful local document search system implementing Retrieval-Augmented Generation (RAG) using Streamlit, ChromaDB, and advanced OCR capabilities. This system enables efficient semantic and exact-match searching across PDF documents and images with optimized text extraction and vector storage.

## 🌟 Key Features

- **Dual Search Modes**: Supports both semantic search and exact matching
- **Multi-Format Support**: Handles PDF documents and images (JPG, PNG, TIFF)
- **Advanced OCR**: Implements optimized OCR with image enhancement
- **GPU Acceleration**: Automatic GPU detection and utilization when available
- **Vector Storage**: Efficient document embedding using ChromaDB
- **Batch Processing**: Optimized batch processing of documents
- **Interactive UI**: Clean, user-friendly interface built with Streamlit
- **Real-time Processing**: Progress tracking and status updates during indexing
- **Confidence Scoring**: OCR confidence metrics for extracted text

## 🛠️ Technical Architecture

### Core Components:
1. **Frontend**: Streamlit-based web interface (`app.py`)
2. **Vector Database**: ChromaDB implementation (`vector_db.py`)
3. **Text Processing**: Enhanced OCR using Tesseract and PyMuPDF
4. **Embedding Model**: Sentence-BERT (paraphrase-MiniLM-L3-v2)

### Dependencies:
```bash
streamlit
torch
sentence-transformers
chromadb
pytesseract
opencv-python
PyMuPDF
Pillow
numpy
tqdm
```

## 📋 Prerequisites

1. Python 3.7+
2. Tesseract OCR installed (Windows: `C:\Program Files\Tesseract-OCR\tesseract.exe`)
3. CUDA-capable GPU (optional, for acceleration)

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/local-document-rag.git
cd local-document-rag
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Install Tesseract OCR:
- Windows: Download and install from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
- Linux: `sudo apt-get install tesseract-ocr`
- Mac: `brew install tesseract`

## 💻 Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Access the web interface at `http://localhost:8501`

3. Upload documents:
   - Use the sidebar to upload PDF documents or images
   - Click "Index Documents" to process uploads
   - Monitor progress in the progress bar

4. Search documents:
   - Enter your search query in the search box
   - Choose search type (Semantic or Exact Match)
   - View results with relevance scores and OCR confidence

## 🔧 Configuration

Key configuration options in `vector_db.py`:
```python
chunk_size = 512        # Text chunk size for indexing
batch_size = 32         # Processing batch size
persist_directory = "vector_db"  # Vector database storage location
```

## 🎯 Performance Optimization

The system includes several optimizations:
- Parallel processing using ThreadPoolExecutor
- GPU acceleration for embedding generation
- CLAHE and adaptive thresholding for OCR
- Batch processing of documents
- Efficient text chunking with sentence awareness

## 📊 Results Format

Search results include:
- Retrieved text segments
- Source file name and page number
- Relevance scores
- OCR confidence metrics
- Match type (Semantic/Exact)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Tesseract OCR for text extraction
- Sentence-Transformers for embeddings
- ChromaDB for vector storage
- Streamlit for the web interface
