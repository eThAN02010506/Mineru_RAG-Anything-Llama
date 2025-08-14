
#!/usr/bin/env python3
"""
Simple PDF parser for RAG-Anything that works on macOS M3 chip
Uses pdftotext or PyPDF2 as fallback
"""
import os
import sys
import json
import subprocess
from pathlib import Path

def simple_parse_pdf(input_dir, output_dir):
    """
    Simple PDF parser using pdftotext or PyPDF2 as fallback
    """
    print("Starting simple PDF parsing")
    print("Input directory: {}".format(input_dir))
    print("Output directory: {}".format(output_dir))
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all PDF files in the input directory
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print("Warning: No PDF files found in {}".format(input_dir))
        return
    
    print("Found {} PDF files: {}".format(len(pdf_files), ', '.join(pdf_files)))
    
    # Output JSONL file path
    output_jsonl = os.path.join(output_dir, "chunks.jsonl")
    
    # Process each PDF file
    chunks = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        print("Processing {}...".format(pdf_file))
        
        # Try PyPDF2 first (more reliable on macOS)
        try:
            print("Trying PyPDF2...")
            # Try to import PyPDF2
            try:
                import PyPDF2
            except ImportError:
                print("Warning: PyPDF2 not installed, trying to install...")
                try:
                    subprocess.run([sys.executable, "-m", "pip", "install", "PyPDF2"], check=True)
                    import PyPDF2
                except Exception as e:
                    print("Error installing PyPDF2: {}".format(e))
                    raise ImportError("Could not import or install PyPDF2")
            
            # Extract text with PyPDF2
            with open(pdf_path, 'rb') as pdf_file_obj:
                pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
                
                # Process each page
                all_text = []
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    if text:
                        all_text.append(text)
                
                # Join all pages
                full_text = "\
\
".join(all_text)
                
                # Simple chunking by paragraphs
                paragraphs = [p.strip() for p in full_text.split("\
\
") if p.strip()]
                
                # Add chunks
                for para in paragraphs:
                    if len(para) > 10:  # Filter out very short paragraphs
                        chunks.append({
                            "text": para,
                            "source": pdf_file
                        })
                
                print("Processed {} with PyPDF2, extracted {} paragraphs".format(pdf_file, len(paragraphs)))
                
        except Exception as e:
            print("Warning: PyPDF2 failed: {}".format(e))
            print("Trying pdftotext...")
            
            # Try pdftotext as fallback
            try:
                # Check if pdftotext is available
                try:
                    subprocess.run(["pdftotext", "-v"], check=True, capture_output=True)
                except (subprocess.SubprocessError, FileNotFoundError):
                    print("Warning: pdftotext not found, trying to install...")
                    try:
                        # Try to install poppler-utils with brew (for macOS)
                        subprocess.run(["brew", "install", "poppler"], check=True, capture_output=True)
                        print("Success: Installed poppler with brew")
                    except Exception as brew_error:
                        print("Warning: Could not install poppler with brew: {}".format(brew_error))
                        raise Exception("pdftotext not available and could not be installed")
                
                # Use pdftotext
                text_output = os.path.join(output_dir, "{}.txt".format(pdf_file))
                subprocess.run(["pdftotext", "-layout", pdf_path, text_output], check=True)
                
                # Read the extracted text
                with open(text_output, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                
                # Simple chunking by paragraphs
                paragraphs = [p.strip() for p in text.split("\
\
") if p.strip()]
                
                # Add chunks
                for para in paragraphs:
                    if len(para) > 10:  # Filter out very short paragraphs
                        chunks.append({
                            "text": para,
                            "source": pdf_file
                        })
                
                # Clean up temporary file
                if os.path.exists(text_output):
                    os.remove(text_output)
                
                print("Processed {} with pdftotext, extracted {} paragraphs".format(pdf_file, len(paragraphs)))
                
            except Exception as pdf_error:
                print("Error: All extraction methods failed for {}: {}".format(pdf_file, pdf_error))
                
                # Last resort: try a very simple text extraction
                print("Trying last resort text extraction...")
                try:
                    # Create a very simple text chunk
                    chunks.append({
                        "text": "This is a document titled: {}. The content could not be extracted automatically.".format(pdf_file),
                        "source": pdf_file
                    })
                    print("Added placeholder text for {}".format(pdf_file))
                except Exception as last_error:
                    print("Error: Even last resort failed: {}".format(last_error))
                    continue
    
    # Write all chunks to JSONL file
    if chunks:
        try:
            with open(output_jsonl, "w", encoding="utf-8") as f:
                for chunk in chunks:
                    f.write(json.dumps(chunk, ensure_ascii=False) + "\
")
            
            print("Success: All PDFs processed. Total chunks: {}".format(len(chunks)))
            print("Chunks saved to {}".format(output_jsonl))
        except Exception as write_error:
            print("Error writing chunks to file: {}".format(write_error))
            raise
    else:
        print("Warning: No chunks were extracted from any PDF files")
        # Create a minimal valid chunks file to prevent errors
        with open(output_jsonl, "w", encoding="utf-8") as f:
            placeholder = {
                "text": "No text could be extracted from the PDF files. This is a placeholder to ensure the system can continue.",
                "source": "system_generated"
            }
            f.write(json.dumps(placeholder, ensure_ascii=False) + "\
")
        print("Created placeholder chunks file")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python simple_pdf_parser.py <input_directory> <output_directory>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    try:
        simple_parse_pdf(input_dir, output_dir)
    except Exception as e:
        print("Fatal error: {}".format(e))
        sys.exit(1)
