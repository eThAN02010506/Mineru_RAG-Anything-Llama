RAG-Anything for macOS M3

This is an enhanced version of RAG-Anything optimized for macOS with M3 chip. The system provides fully offline Retrieval-Augmented Generation (RAG) capabilities for PDF documents.


Features
• **Fully Offline Operation**: All components run locally without requiring internet connectivity
• **M3 Chip Optimization**: Enhanced for Apple Silicon M3 processors
• **Robust PDF Processing**: Multiple fallback methods for PDF text extraction
• **Error Recovery**: Automatic detection and fixing of common issues
• **Interactive Mode**: Ask multiple questions about your documents


Setup Instructions

1. Environment Setup

Create and activate a virtual environment:


python -m venv .venv
source .venv/bin/activate


2. Install Dependencies

Install the required packages:


pip install -r requirements.txt


Key dependencies include:
• sentence-transformers
• faiss-cpu
• llama-cpp-python
• PyPDF2 (fallback PDF parser)


3. Install Poppler (for PDF processing)

For better PDF text extraction, install Poppler:


brew install poppler


This provides the `pdftotext` utility which gives better results than Python-only solutions.


Usage

Basic Usage

Process documents and ask a question:


python run_rag.py ./data "What is the main topic of the document?"


Interactive Mode

For multiple questions about the same documents:


python run_rag.py ./data


Fixing Issues

If you encounter problems with chunks.jsonl:


python run_rag.py ./data --fix


This will attempt to repair any issues with the document processing.


Force Reprocessing

To reprocess all documents:


python run_rag.py ./data --reparse


Troubleshooting

Issue: "No valid text chunks" Error

If you see an error about no valid text chunks:

1. Try the fix mode:
```bash
python run_rag.py ./data --fix
```

2. If that doesn't work, try reprocessing:
```bash
python run_rag.py ./data --reparse
```

3. If issues persist, run the simple parser directly:
```bash
python simple_pdf_parser.py ./data ./data
python run_rag.py ./data
```


Issue: MinerU Import Errors

If you see errors related to MinerU:

1. The system will automatically fall back to simpler PDF processing methods
2. Check that your PDF is text-based and not just scanned images
3. Try using the simple parser directly:
```bash
python simple_pdf_parser.py ./data ./data
```


Issue: Memory Errors

If you encounter memory issues:

1. Process fewer documents at a time
2. Close other memory-intensive applications
3. Adjust batch size in embed.py if needed


System Components
1. **PDF Processing**: 
- Primary: MinerU (if available)
- Fallback 1: pdftotext (from Poppler)
- Fallback 2: PyPDF2
- Fallback 3: textract

2. **Embedding Generation**:
- Model: all-MiniLM-L6-v2
- Index: FAISS for similarity search

3. **Response Generation**:
- Model: Mistral-7B (GGUF format)
- Runtime: llama-cpp-python


File Structure
• `run_rag.py`: Main entry point
• `parse_with_mineru.py`: Enhanced PDF parser with fallbacks
• `simple_pdf_parser.py`: Simplified PDF parser for troubleshooting
• `embed.py`: Creates vector embeddings for text chunks
• `query.py`: Processes questions and generates answers
• `fix_chunks_issue.py`: Utility to fix common issues


Advanced Usage

Custom PDF Processing

If you have specific PDF processing needs:


python parse_with_mineru.py ./input_dir ./output_dir


Direct Embedding

To create embeddings directly:


python embed.py ./data/chunks.jsonl


Direct Querying

To query an existing index:


python query.py ./data "Your question here"


Notes for M3 Mac Users
• The system is optimized for Apple Silicon M3 processors
• Uses CPU-based inference for compatibility
• Multiple fallback methods ensure PDF processing works even without MinerU
• Automatic installation of required utilities when possible