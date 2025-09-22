import json
import os
import subprocess
import sys
from pathlib import Path

from embed import embed_chunks
from query import interactive_query, query


def run_pipeline(pdf_dir, question=None, force_reparse=False):
    """
    \u5b8c\u5168\u79bb\u7ebf\u7684RAG\u6d41\u6c34\u7ebf
    """
    # \u8bbe\u7f6e\u73af\u5883\u53d8\u91cf\uff0c\u5f3a\u5236\u79bb\u7ebf\u6a21\u5f0f
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    print("\u542f\u52a8\u79bb\u7ebfRAG\u7cfb\u7edf")
    print("\u76ee\u6807\u76ee\u5f55: {}".format(pdf_dir))

    # \u68c0\u67e5\u76ee\u5f55
    if not os.path.exists(pdf_dir):
        raise FileNotFoundError(
            "\u9519\u8bef: \u76ee\u5f55\u4e0d\u5b58\u5728: {}".format(pdf_dir)
        )

    # \u6587\u4ef6\u8def\u5f84
    chunk_jsonl_path = os.path.join(pdf_dir, "chunks.jsonl")
    faiss_index_path = os.path.join(pdf_dir, "faiss_index.index")
    mapping_path = os.path.join(pdf_dir, "mapping.json")

    # 1. \u68c0\u67e5PDF\u6587\u4ef6
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(
            "\u8b66\u544a: \u5728\u76ee\u5f55 {} \u4e2d\u672a\u627e\u5230PDF\u6587\u4ef6".format(
                pdf_dir
            )
        )
        # \u68c0\u67e5\u662f\u5426\u5df2\u6709\u5904\u7406\u597d\u7684chunks.jsonl
        if not os.path.exists(chunk_jsonl_path):
            raise ValueError(
                "\u9519\u8bef: \u65e2\u6ca1\u6709PDF\u6587\u4ef6\u4e5f\u6ca1\u6709chunks.jsonl\u6587\u4ef6"
            )
        else:
            print(
                "\u6210\u529f: \u627e\u5230\u73b0\u6709\u7684chunks.jsonl\u6587\u4ef6"
            )
    else:
        print(
            "\u627e\u5230 {} \u4e2aPDF\u6587\u4ef6: {}".format(
                len(pdf_files), ", ".join(pdf_files)
            )
        )

    # 2. \u89e3\u6790PDF (\u5982\u679c\u9700\u8981)
    if force_reparse or (pdf_files and not os.path.exists(chunk_jsonl_path)):
        print("\u5f00\u59cb\u89e3\u6790PDF\u6587\u4ef6...")
        try:
            # \u9996\u5148\u5c1d\u8bd5\u4f7f\u7528parse_with_mineru.py
            print("\u5c1d\u8bd5\u4f7f\u7528parse_with_mineru.py...")
            env = os.environ.copy()
            env["HF_DATASETS_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"

            result = subprocess.run(
                [sys.executable, "parse_with_mineru.py", pdf_dir, pdf_dir],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            print("\u6210\u529f: PDF\u89e3\u6790\u5b8c\u6210")
            if result.stdout:
                print(result.stdout)

            # \u68c0\u67e5\u662f\u5426\u6210\u529f\u521b\u5efa\u4e86chunks.jsonl\u6587\u4ef6
            if (
                not os.path.exists(chunk_jsonl_path)
                or os.path.getsize(chunk_jsonl_path) == 0
            ):
                print(
                    "\u8b66\u544a: parse_with_mineru.py\u672a\u80fd\u521b\u5efa\u6709\u6548\u7684chunks.jsonl\u6587\u4ef6"
                )
                raise Exception(
                    "\u672a\u80fd\u521b\u5efa\u6709\u6548\u7684chunks.jsonl\u6587\u4ef6"
                )

        except Exception as e:
            print(
                "\u8b66\u544a: parse_with_mineru.py\u89e3\u6790\u5931\u8d25: {}".format(
                    e
                )
            )
            if hasattr(e, "stderr") and e.stderr:
                print("\u9519\u8bef\u4fe1\u606f: {}".format(e.stderr))

            # \u5c1d\u8bd5\u4f7f\u7528\u5907\u7528\u89e3\u6790\u5668
            print(
                "\u5c1d\u8bd5\u4f7f\u7528\u5907\u7528\u89e3\u6790\u5668simple_pdf_parser.py..."
            )
            try:
                # \u68c0\u67e5\u662f\u5426\u6709simple_pdf_parser.py
                if not os.path.exists("simple_pdf_parser.py"):
                    # \u521b\u5efa\u7b80\u5355\u89e3\u6790\u5668
                    create_simple_parser()

                env = os.environ.copy()
                env["HF_DATASETS_OFFLINE"] = "1"
                env["TRANSFORMERS_OFFLINE"] = "1"

                result = subprocess.run(
                    [sys.executable, "simple_pdf_parser.py", pdf_dir, pdf_dir],
                    check=True,
                    capture_output=True,
                    text=True,
                    env=env,
                )
                print("\u6210\u529f: \u5907\u7528\u89e3\u6790\u5668\u6210\u529f")
                if result.stdout:
                    print(result.stdout)
            except Exception as backup_e:
                print(
                    "\u9519\u8bef: \u5907\u7528\u89e3\u6790\u5668\u4e5f\u5931\u8d25\u4e86: {}".format(
                        backup_e
                    )
                )

                # \u521b\u5efa\u4e00\u4e2a\u57fa\u672c\u7684chunks.jsonl\u6587\u4ef6
                print("\u521b\u5efa\u57fa\u672c\u7684chunks.jsonl\u6587\u4ef6...")
                with open(chunk_jsonl_path, "w", encoding="utf-8") as f:
                    for pdf_file in pdf_files:
                        chunk = {
                            "text": "\u8fd9\u662f\u6587\u6863 {} \u7684\u5185\u5bb9\u3002\u7531\u4e8e\u89e3\u6790\u5668\u5931\u8d25\uff0c\u8fd9\u662f\u4e00\u4e2a\u5360\u4f4d\u7b26\u3002".format(
                                pdf_file
                            ),
                            "source": pdf_file,
                        }
                        f.write(
                            json.dumps(chunk, ensure_ascii=False)
                            + "\
"
                        )
                print("\u521b\u5efa\u4e86\u57fa\u672c\u7684chunks.jsonl\u6587\u4ef6")
    else:
        print(
            "\u6210\u529f: \u627e\u5230\u73b0\u6709\u7684\u6587\u672c\u7247\u6bb5\u6587\u4ef6\uff0c\u8df3\u8fc7PDF\u89e3\u6790"
        )

    # 2.5 \u9a8c\u8bc1chunks.jsonl\u6587\u4ef6
    if os.path.exists(chunk_jsonl_path):
        print("\u9a8c\u8bc1chunks.jsonl\u6587\u4ef6...")
        valid_chunks = 0
        try:
            with open(chunk_jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            text = data.get("text", "").strip()
                            if (
                                text and len(text) > 10
                            ):  # \u8fc7\u6ee4\u592a\u77ed\u7684\u6587\u672c
                                valid_chunks += 1
                        except json.JSONDecodeError:
                            continue

            print(
                "\u627e\u5230 {} \u4e2a\u6709\u6548\u6587\u672c\u7247\u6bb5".format(
                    valid_chunks
                )
            )

            if valid_chunks == 0:
                print(
                    "\u8b66\u544a: chunks.jsonl\u6587\u4ef6\u4e2d\u6ca1\u6709\u6709\u6548\u7684\u6587\u672c\u7247\u6bb5\uff0c\u5c06\u91cd\u65b0\u89e3\u6790PDF"
                )
                force_reparse = True
                # \u91cd\u65b0\u89e3\u6790PDF
                if pdf_files:
                    try:
                        # \u5220\u9664\u65e0\u6548\u7684chunks\u6587\u4ef6
                        os.remove(chunk_jsonl_path)
                        print(
                            "\u5df2\u5220\u9664\u65e0\u6548\u7684chunks\u6587\u4ef6: {}".format(
                                chunk_jsonl_path
                            )
                        )

                        # \u9996\u5148\u5c1d\u8bd5\u4f7f\u7528parse_with_mineru.py
                        print("\u5c1d\u8bd5\u4f7f\u7528parse_with_mineru.py...")
                        try:
                            env = os.environ.copy()
                            env["HF_DATASETS_OFFLINE"] = "1"
                            env["TRANSFORMERS_OFFLINE"] = "1"

                            result = subprocess.run(
                                [
                                    sys.executable,
                                    "parse_with_mineru.py",
                                    pdf_dir,
                                    pdf_dir,
                                ],
                                check=True,
                                capture_output=True,
                                text=True,
                                env=env,
                            )
                            print("\u6210\u529f: PDF\u89e3\u6790\u5b8c\u6210")

                            # \u68c0\u67e5\u662f\u5426\u6210\u529f\u521b\u5efa\u4e86chunks.jsonl\u6587\u4ef6
                            if (
                                not os.path.exists(chunk_jsonl_path)
                                or os.path.getsize(chunk_jsonl_path) == 0
                            ):
                                print(
                                    "\u8b66\u544a: parse_with_mineru.py\u672a\u80fd\u521b\u5efa\u6709\u6548\u7684chunks.jsonl\u6587\u4ef6"
                                )
                                raise Exception(
                                    "\u672a\u80fd\u521b\u5efa\u6709\u6548\u7684chunks.jsonl\u6587\u4ef6"
                                )

                        except Exception as mineru_e:
                            print(
                                "\u8b66\u544a: parse_with_mineru.py\u89e3\u6790\u5931\u8d25: {}".format(
                                    mineru_e
                                )
                            )

                            # \u5c1d\u8bd5\u4f7f\u7528\u5907\u7528\u89e3\u6790\u5668
                            if not os.path.exists("simple_pdf_parser.py"):
                                create_simple_parser()

                            env = os.environ.copy()
                            env["HF_DATASETS_OFFLINE"] = "1"
                            env["TRANSFORMERS_OFFLINE"] = "1"

                            result = subprocess.run(
                                [
                                    sys.executable,
                                    "simple_pdf_parser.py",
                                    pdf_dir,
                                    pdf_dir,
                                ],
                                check=True,
                                capture_output=True,
                                text=True,
                                env=env,
                            )
                            print(
                                "\u6210\u529f: \u5907\u7528\u89e3\u6790\u5668\u6210\u529f"
                            )
                    except Exception as e:
                        print(
                            "\u9519\u8bef: \u91cd\u65b0\u89e3\u6790\u5931\u8d25: {}".format(
                                e
                            )
                        )

                        # \u521b\u5efa\u4e00\u4e2a\u57fa\u672c\u7684chunks.jsonl\u6587\u4ef6
                        print(
                            "\u521b\u5efa\u57fa\u672c\u7684chunks.jsonl\u6587\u4ef6..."
                        )
                        with open(chunk_jsonl_path, "w", encoding="utf-8") as f:
                            for pdf_file in pdf_files:
                                chunk = {
                                    "text": "\u8fd9\u662f\u6587\u6863 {} \u7684\u5185\u5bb9\u3002\u7531\u4e8e\u89e3\u6790\u5668\u5931\u8d25\uff0c\u8fd9\u662f\u4e00\u4e2a\u5360\u4f4d\u7b26\u3002".format(
                                        pdf_file
                                    ),
                                    "source": pdf_file,
                                }
                                f.write(
                                    json.dumps(chunk, ensure_ascii=False)
                                    + "\
"
                                )
                        print(
                            "\u521b\u5efa\u4e86\u57fa\u672c\u7684chunks.jsonl\u6587\u4ef6"
                        )
                else:
                    print(
                        "\u9519\u8bef: \u6ca1\u6709PDF\u6587\u4ef6\u53ef\u4f9b\u89e3\u6790"
                    )
                    sys.exit(1)
        except Exception as e:
            print("\u8b66\u544a: \u8bfb\u53d6chunks.jsonl\u51fa\u9519: {}".format(e))
            if pdf_files:
                force_reparse = True
                print("\u5c06\u91cd\u65b0\u89e3\u6790PDF\u6587\u4ef6")
            else:
                print(
                    "\u9519\u8bef: chunks.jsonl\u65e0\u6548\u4e14\u6ca1\u6709PDF\u6587\u4ef6\u53ef\u4f9b\u89e3\u6790"
                )
                sys.exit(1)

    # 3. \u751f\u6210\u5411\u91cf\u7d22\u5f15 (\u5982\u679c\u9700\u8981)
    if (
        force_reparse
        or not os.path.exists(faiss_index_path)
        or not os.path.exists(mapping_path)
        or os.path.getsize(faiss_index_path) == 0
        or os.path.getsize(mapping_path) == 0
    ):
        print("\u751f\u6210\u5411\u91cf\u7d22\u5f15...")
        try:
            embed_chunks(
                jsonl_path=chunk_jsonl_path,
                faiss_index_path=faiss_index_path,
                mapping_path=mapping_path,
            )
        except Exception as e:
            print(
                "\u9519\u8bef: \u751f\u6210\u5411\u91cf\u7d22\u5f15\u5931\u8d25: {}".format(
                    e
                )
            )
            sys.exit(1)
    else:
        print(
            "\u6210\u529f: \u627e\u5230\u73b0\u6709\u7684\u5411\u91cf\u7d22\u5f15\uff0c\u8df3\u8fc7\u751f\u6210\u6b65\u9aa4"
        )
        print(
            "  faiss_index.index \u5927\u5c0f: {} \u5b57\u8282".format(
                os.path.getsize(faiss_index_path)
            )
        )
        print(
            "  mapping.json \u5927\u5c0f: {} \u5b57\u8282".format(
                os.path.getsize(mapping_path)
            )
        )

    # 4. \u95ee\u7b54\u67e5\u8be2
    if question:
        # \u5355\u6b21\u67e5\u8be2
        print("\u95ee\u9898: {}".format(question))
        try:
            answer = query(question, index_dir=pdf_dir)
            print(
                "\
\u56de\u7b54:\
{}".format(
                    answer
                )
            )
        except Exception as e:
            print("\u9519\u8bef: \u67e5\u8be2\u5931\u8d25: {}".format(e))
            sys.exit(1)
    else:
        # \u4ea4\u4e92\u6a21\u5f0f
        interactive_query(pdf_dir)


def create_simple_parser():
    """\u521b\u5efa\u7b80\u5355\u7684PDF\u89e3\u6790\u5668\u4f5c\u4e3a\u5907\u7528"""
    parser_code = """#!/usr/bin/env python3
import os
import sys
import json
import subprocess
from pathlib import Path

def simple_parse_pdf(input_dir, output_dir):
    \"\"\"
    Simple PDF parser using pdftotext or PyPDF2 as fallback
    \"\"\"
    # \u8bbe\u7f6e\u73af\u5883\u53d8\u91cf\uff0c\u5f3a\u5236\u79bb\u7ebf\u6a21\u5f0f
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
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
                full_text = "\\
\\
".join(all_text)
                
                # Simple chunking by paragraphs
                paragraphs = [p.strip() for p in full_text.split("\\
\\
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
                paragraphs = [p.strip() for p in text.split("\\
\\
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
                    f.write(json.dumps(chunk, ensure_ascii=False) + "\\
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
            f.write(json.dumps(placeholder, ensure_ascii=False) + "\\
")
        print("Created placeholder chunks file")

if __name__ == "__main__":
    # \u8bbe\u7f6e\u73af\u5883\u53d8\u91cf\uff0c\u5f3a\u5236\u79bb\u7ebf\u6a21\u5f0f
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
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
"""

    with open("simple_pdf_parser.py", "w", encoding="utf-8") as f:
        f.write(parser_code)

    # Make it executable
    os.chmod("simple_pdf_parser.py", 0o755)

    print("\u521b\u5efa\u4e86simple_pdf_parser.py\u4f5c\u4e3a\u5907\u7528")


def main():
    # \u8bbe\u7f6e\u73af\u5883\u53d8\u91cf\uff0c\u5f3a\u5236\u79bb\u7ebf\u6a21\u5f0f
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    if len(sys.argv) < 2:
        print(
            """
\u79bb\u7ebfRAG\u7cfb\u7edf\u4f7f\u7528\u8bf4\u660e:

\u57fa\u672c\u7528\u6cd5:
  python run_rag.py <PDF\u76ee\u5f55> [\u95ee\u9898]

\u53c2\u6570\u8bf4\u660e:
  PDF\u76ee\u5f55    - \u5305\u542bPDF\u6587\u4ef6\u7684\u76ee\u5f55\u8def\u5f84
  \u95ee\u9898       - \u53ef\u9009\uff0c\u8981\u8be2\u95ee\u7684\u95ee\u9898\u3002\u5982\u679c\u4e0d\u63d0\u4f9b\u5219\u8fdb\u5165\u4ea4\u4e92\u6a21\u5f0f

\u9009\u9879:
  --reparse  - \u5f3a\u5236\u91cd\u65b0\u89e3\u6790PDF\u548c\u751f\u6210\u7d22\u5f15
  --fix      - \u5c1d\u8bd5\u4fee\u590dchunks.jsonl\u6587\u4ef6

\u793a\u4f8b:
  python run_rag.py ./data "\u8fd9\u4e2a\u6587\u6863\u8bb2\u4e86\u4ec0\u4e48\uff1f"
  python run_rag.py ./data  # \u4ea4\u4e92\u6a21\u5f0f
  python run_rag.py ./data --reparse "\u91cd\u65b0\u5904\u7406\u540e\u95ee\u95ee\u9898"
  python run_rag.py ./data --fix  # \u5c1d\u8bd5\u4fee\u590dchunks.jsonl

\u53ef\u7528\u7684\u6570\u636e\u76ee\u5f55:
  ./data - \u5305\u542b\u5df2\u5904\u7406\u7684PDF\u6570\u636e
        """
        )
        sys.exit(1)

    # \u89e3\u6790\u53c2\u6570
    args = sys.argv[1:]
    force_reparse = "--reparse" in args
    fix_mode = "--fix" in args

    if force_reparse:
        args.remove("--reparse")

    if fix_mode:
        args.remove("--fix")

    pdf_dir = args[0]
    question = " ".join(args[1:]) if len(args) > 1 else None

    # \u5982\u679c\u662f\u4fee\u590d\u6a21\u5f0f
    if fix_mode:
        print(
            "\u4fee\u590d\u6a21\u5f0f: \u5c1d\u8bd5\u4fee\u590d {} \u4e2d\u7684\u6570\u636e".format(
                pdf_dir
            )
        )
        try:
            # \u8fd0\u884c\u4fee\u590d\u811a\u672c
            env = os.environ.copy()
            env["HF_DATASETS_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"

            subprocess.run(
                [sys.executable, "fix_chunks_issue.py", pdf_dir], check=True, env=env
            )
            print(
                "\u6210\u529f: \u4fee\u590d\u5b8c\u6210\uff0c\u73b0\u5728\u8fd0\u884cRAG\u7cfb\u7edf"
            )
        except Exception as e:
            print("\u9519\u8bef: \u4fee\u590d\u5931\u8d25: {}".format(e))
            sys.exit(1)

    try:
        run_pipeline(pdf_dir, question, force_reparse)
    except Exception as e:
        print("\u7cfb\u7edf\u9519\u8bef: {}".format(e))
        sys.exit(1)


if __name__ == "__main__":
    # \u8bbe\u7f6e\u73af\u5883\u53d8\u91cf\uff0c\u5f3a\u5236\u79bb\u7ebf\u6a21\u5f0f
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    main()
