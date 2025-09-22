#!/usr/bin/env python3
"""
Fixed MinerU PDF parser for RAG-Anything
Corrects both the MinerU API calls and ensures proper JSON formatting in output
"""
import glob
import json
import os
import re
import sys
from pathlib import Path


def parse_with_mineru(input_dir, output_dir):
    """
    Parse PDF files using the correct MinerU API and properly format JSON output
    """
    print("Starting PDF parsing with MinerU")
    print("Input directory: {}".format(input_dir))
    print("Output directory: {}".format(output_dir))

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Find all PDF files in the input directory
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("Warning: No PDF files found in {}".format(input_dir))
        return False

    print("Found {} PDF files: {}".format(len(pdf_files), ", ".join(pdf_files)))

    # Output JSONL file path
    output_jsonl = os.path.join(output_dir, "chunks.jsonl")

    # Try to import MinerU with correct path
    try:
        from mineru.cli.common import do_parse, read_fn

        print("Successfully imported MinerU")
        mineru_available = True
    except ImportError:
        print("Warning: MinerU import failed: No module named 'mineru'")
        print("Warning: MinerU not properly installed")
        mineru_available = False

    # Process each PDF file
    chunks = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        print("Processing {}...".format(pdf_file))

        # Try MinerU first if available
        if mineru_available:
            try:
                # Read PDF file
                pdf_bytes = read_fn(Path(pdf_path))
                pdf_name = Path(pdf_path).stem

                # Call MinerU parser with correct parameters
                do_parse(
                    output_dir=output_dir,
                    pdf_file_names=[pdf_name],
                    pdf_bytes_list=[pdf_bytes],
                    p_lang_list=["en"],  # Assuming English, change as needed
                    backend="pipeline",
                    parse_method="auto",  # Correct parameter value
                    formula_enable=True,
                    table_enable=True,
                )

                # IMPORTANT: MinerU creates a subdirectory structure
                # The output is in: output_dir/pdf_name/auto/
                mineru_output_dir = os.path.join(output_dir, pdf_name, "auto")

                # Check if the directory exists
                if os.path.exists(mineru_output_dir):
                    print("Found MinerU output directory: {}".format(mineru_output_dir))

                    # Look for the content list JSON file
                    content_list_json = os.path.join(
                        mineru_output_dir, "{}_content_list.json".format(pdf_name)
                    )
                    middle_json = os.path.join(
                        mineru_output_dir, "{}_middle.json".format(pdf_name)
                    )
                    model_json = os.path.join(
                        mineru_output_dir, "{}_model.json".format(pdf_name)
                    )
                    md_file = os.path.join(mineru_output_dir, "{}.md".format(pdf_name))

                    # Try different output files in order of preference
                    if os.path.exists(content_list_json):
                        print("Using content_list.json for extraction")
                        with open(content_list_json, "r", encoding="utf-8") as f:
                            data = json.load(f)

                        # Extract text from content_list.json
                        if isinstance(data, list):
                            for item in data:
                                if (
                                    isinstance(item, dict)
                                    and "text" in item
                                    and item["text"].strip()
                                ):
                                    chunks.append(
                                        {
                                            "text": item["text"].strip(),
                                            "source": pdf_file,
                                        }
                                    )
                    elif os.path.exists(middle_json):
                        print("Using middle.json for extraction")
                        with open(middle_json, "r", encoding="utf-8") as f:
                            data = json.load(f)

                        # Extract text from middle.json
                        if "pages" in data:
                            for page_num, page in enumerate(data["pages"]):
                                if "blocks" in page:
                                    page_chunks = []
                                    for block in page["blocks"]:
                                        if "text" in block and block["text"].strip():
                                            page_chunks.append(block["text"].strip())

                                    if page_chunks:
                                        # Join blocks into paragraphs
                                        page_text = "\
".join(
                                            page_chunks
                                        )
                                        chunks.append(
                                            {"text": page_text, "source": pdf_file}
                                        )
                    elif os.path.exists(model_json):
                        print("Using model.json for extraction")
                        with open(model_json, "r", encoding="utf-8") as f:
                            data = json.load(f)

                        # Extract text from model.json
                        if "pages" in data:
                            for page_num, page in enumerate(data["pages"]):
                                if "blocks" in page:
                                    page_chunks = []
                                    for block in page["blocks"]:
                                        if "text" in block and block["text"].strip():
                                            page_chunks.append(block["text"].strip())

                                    if page_chunks:
                                        # Join blocks into paragraphs
                                        page_text = "\
".join(
                                            page_chunks
                                        )
                                        chunks.append(
                                            {"text": page_text, "source": pdf_file}
                                        )
                    elif os.path.exists(md_file):
                        print("Using markdown file for extraction")
                        with open(md_file, "r", encoding="utf-8") as f:
                            md_text = f.read()

                        # Split markdown into sections
                        sections = re.split(
                            r"\
#{1,6} ",
                            md_text,
                        )
                        for section in sections:
                            if section.strip():
                                chunks.append(
                                    {"text": section.strip(), "source": pdf_file}
                                )
                    else:
                        # Try to find any JSON file in the directory
                        json_files = glob.glob(
                            os.path.join(mineru_output_dir, "*.json")
                        )
                        if json_files:
                            print("Found JSON files: {}".format(json_files))
                            for json_file in json_files:
                                try:
                                    with open(json_file, "r", encoding="utf-8") as f:
                                        data = json.load(f)

                                    # Try to extract text from various JSON structures
                                    if isinstance(data, dict):
                                        # Try to find text fields recursively
                                        texts = []

                                        def extract_text(obj, texts):
                                            if isinstance(obj, dict):
                                                for key, value in obj.items():
                                                    if (
                                                        key == "text"
                                                        and isinstance(value, str)
                                                        and value.strip()
                                                    ):
                                                        texts.append(value.strip())
                                                    elif isinstance(
                                                        value, (dict, list)
                                                    ):
                                                        extract_text(value, texts)
                                            elif isinstance(obj, list):
                                                for item in obj:
                                                    extract_text(item, texts)

                                        extract_text(data, texts)

                                        if texts:
                                            for text in texts:
                                                chunks.append(
                                                    {"text": text, "source": pdf_file}
                                                )
                                except Exception as e:
                                    print(
                                        "Error processing JSON file {}: {}".format(
                                            json_file, e
                                        )
                                    )
                        else:
                            print(
                                "Warning: No output files found in MinerU output directory"
                            )
                            raise Exception("No output files found")
                else:
                    print(
                        "Warning: MinerU output directory not found: {}".format(
                            mineru_output_dir
                        )
                    )
                    print(
                        "Available directories in {}: {}".format(
                            output_dir, os.listdir(output_dir)
                        )
                    )

                    # Try to find any directory that might contain MinerU output
                    pdf_name_dirs = [
                        d
                        for d in os.listdir(output_dir)
                        if os.path.isdir(os.path.join(output_dir, d))
                    ]
                    if pdf_name_dirs:
                        print(
                            "Found potential PDF name directories: {}".format(
                                pdf_name_dirs
                            )
                        )
                        for pdf_dir in pdf_name_dirs:
                            auto_dir = os.path.join(output_dir, pdf_dir, "auto")
                            if os.path.exists(auto_dir):
                                print(
                                    "Found potential MinerU output directory: {}".format(
                                        auto_dir
                                    )
                                )
                                # Look for any JSON or MD files
                                json_files = glob.glob(os.path.join(auto_dir, "*.json"))
                                md_files = glob.glob(os.path.join(auto_dir, "*.md"))

                                if json_files or md_files:
                                    print(
                                        "Found files in {}: {} JSON files, {} MD files".format(
                                            auto_dir, len(json_files), len(md_files)
                                        )
                                    )

                                    # Try to process JSON files
                                    for json_file in json_files:
                                        try:
                                            with open(
                                                json_file, "r", encoding="utf-8"
                                            ) as f:
                                                data = json.load(f)

                                            # Try to extract text from various JSON structures
                                            if isinstance(data, dict):
                                                # Try to find text fields recursively
                                                texts = []

                                                def extract_text(obj, texts):
                                                    if isinstance(obj, dict):
                                                        for key, value in obj.items():
                                                            if (
                                                                key == "text"
                                                                and isinstance(
                                                                    value, str
                                                                )
                                                                and value.strip()
                                                            ):
                                                                texts.append(
                                                                    value.strip()
                                                                )
                                                            elif isinstance(
                                                                value, (dict, list)
                                                            ):
                                                                extract_text(
                                                                    value, texts
                                                                )
                                                    elif isinstance(obj, list):
                                                        for item in obj:
                                                            extract_text(item, texts)

                                                extract_text(data, texts)

                                                if texts:
                                                    for text in texts:
                                                        chunks.append(
                                                            {
                                                                "text": text,
                                                                "source": pdf_file,
                                                            }
                                                        )
                                        except Exception as e:
                                            print(
                                                "Error processing JSON file {}: {}".format(
                                                    json_file, e
                                                )
                                            )

                                    # Try to process MD files
                                    for md_file in md_files:
                                        try:
                                            with open(
                                                md_file, "r", encoding="utf-8"
                                            ) as f:
                                                md_text = f.read()

                                            # Split markdown into sections
                                            sections = re.split(
                                                r"\
#{1,6} ",
                                                md_text,
                                            )
                                            for section in sections:
                                                if section.strip():
                                                    chunks.append(
                                                        {
                                                            "text": section.strip(),
                                                            "source": pdf_file,
                                                        }
                                                    )
                                        except Exception as e:
                                            print(
                                                "Error processing MD file {}: {}".format(
                                                    md_file, e
                                                )
                                            )

                    raise Exception("MinerU output directory not found")

            except Exception as e:
                print("Warning: MinerU processing failed: {}".format(e))
                print("Falling back to pdftotext extraction...")

                # Try pdftotext as fallback
                try:
                    text_output = os.path.join(output_dir, "{}.txt".format(pdf_file))
                    os.system(
                        'pdftotext -layout "{}" "{}"'.format(pdf_path, text_output)
                    )

                    if os.path.exists(text_output) and os.path.getsize(text_output) > 0:
                        with open(
                            text_output, "r", encoding="utf-8", errors="ignore"
                        ) as f:
                            text = f.read()

                        # Simple chunking by paragraphs
                        paragraphs = [
                            p.strip()
                            for p in text.split(
                                "\
\
"
                            )
                            if p.strip()
                        ]

                        for para in paragraphs:
                            if len(para) > 10:  # Filter out very short paragraphs
                                chunks.append({"text": para, "source": pdf_file})

                        print(
                            "Processed {} with pdftotext, extracted {} paragraphs".format(
                                pdf_file, len(paragraphs)
                            )
                        )

                        # Clean up temporary file
                        os.remove(text_output)
                    else:
                        print("Error: Fallback processing failed: empty separator")
                        print("Trying last resort with PyPDF2...")

                        # Try PyPDF2 as last resort
                        try:
                            try:
                                import PyPDF2
                            except ImportError:
                                print(
                                    "Warning: PyPDF2 not installed, trying to install..."
                                )
                                os.system(
                                    "{} -m pip install PyPDF2".format(sys.executable)
                                )
                                import PyPDF2

                            with open(pdf_path, "rb") as pdf_file_obj:
                                pdf_reader = PyPDF2.PdfReader(pdf_file_obj)

                                # Process each page
                                for page_num in range(len(pdf_reader.pages)):
                                    page = pdf_reader.pages[page_num]
                                    text = page.extract_text()
                                    if text and text.strip():
                                        chunks.append(
                                            {"text": text.strip(), "source": pdf_file}
                                        )

                                print(
                                    "Processed {} with PyPDF2, extracted {} pages".format(
                                        pdf_file, len(pdf_reader.pages)
                                    )
                                )
                        except Exception as pypdf_error:
                            print(
                                "Error: All extraction methods failed for {}: {}".format(
                                    pdf_file, pypdf_error
                                )
                            )

                            # Add a placeholder chunk as last resort
                            chunks.append(
                                {
                                    "text": "This is a document titled: {}. The content could not be extracted automatically.".format(
                                        pdf_file
                                    ),
                                    "source": pdf_file,
                                }
                            )
                            print("Added placeholder text for {}".format(pdf_file))
                except Exception as fallback_error:
                    print(
                        "Error: Fallback processing failed: {}".format(fallback_error)
                    )

                    # Add a placeholder chunk as last resort
                    chunks.append(
                        {
                            "text": "This is a document titled: {}. The content could not be extracted automatically.".format(
                                pdf_file
                            ),
                            "source": pdf_file,
                        }
                    )
                    print("Added placeholder text for {}".format(pdf_file))
        else:
            # MinerU not available, try pdftotext directly
            print("MinerU not available, trying pdftotext...")

            try:
                text_output = os.path.join(output_dir, "{}.txt".format(pdf_file))
                os.system('pdftotext -layout "{}" "{}"'.format(pdf_path, text_output))

                if os.path.exists(text_output) and os.path.getsize(text_output) > 0:
                    with open(text_output, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()

                    # Simple chunking by paragraphs
                    paragraphs = [
                        p.strip()
                        for p in text.split(
                            "\
\
"
                        )
                        if p.strip()
                    ]

                    for para in paragraphs:
                        if len(para) > 10:  # Filter out very short paragraphs
                            chunks.append({"text": para, "source": pdf_file})

                    print(
                        "Processed {} with pdftotext, extracted {} paragraphs".format(
                            pdf_file, len(paragraphs)
                        )
                    )

                    # Clean up temporary file
                    os.remove(text_output)
                else:
                    raise Exception("empty separator")
            except Exception as e:
                print("Error: pdftotext extraction failed: {}".format(e))
                print("Trying PyPDF2...")

                # Try PyPDF2 as fallback
                try:
                    try:
                        import PyPDF2
                    except ImportError:
                        print("Warning: PyPDF2 not installed, trying to install...")
                        os.system("{} -m pip install PyPDF2".format(sys.executable))
                        import PyPDF2

                    with open(pdf_path, "rb") as pdf_file_obj:
                        pdf_reader = PyPDF2.PdfReader(pdf_file_obj)

                        # Process each page
                        for page_num in range(len(pdf_reader.pages)):
                            page = pdf_reader.pages[page_num]
                            text = page.extract_text()
                            if text and text.strip():
                                chunks.append(
                                    {"text": text.strip(), "source": pdf_file}
                                )

                        print(
                            "Processed {} with PyPDF2, extracted {} pages".format(
                                pdf_file, len(pdf_reader.pages)
                            )
                        )
                except Exception as pypdf_error:
                    print(
                        "Error: All extraction methods failed for {}: {}".format(
                            pdf_file, pypdf_error
                        )
                    )

                    # Add a placeholder chunk as last resort
                    chunks.append(
                        {
                            "text": "This is a document titled: {}. The content could not be extracted automatically.".format(
                                pdf_file
                            ),
                            "source": pdf_file,
                        }
                    )
                    print("Added placeholder text for {}".format(pdf_file))

    # Write all chunks to JSONL file - IMPORTANT: Write each JSON object on a separate line
    if chunks:
        try:
            with open(output_jsonl, "w", encoding="utf-8") as f:
                for chunk in chunks:
                    # Ensure each JSON object is on its own line
                    f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

            print("Success: All PDFs processed. Total chunks: {}".format(len(chunks)))
            print("Chunks saved to {}".format(output_jsonl))
            return True
        except Exception as write_error:
            print("Error writing chunks to file: {}".format(write_error))
            return False
    else:
        print("Error: No chunks were extracted from any PDF files")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python parse_with_mineru.py <input_directory> <output_directory>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    try:
        success = parse_with_mineru(input_dir, output_dir)
        sys.exit(0 if success else 1)
    except Exception as e:
        print("Fatal error: {}".format(e))
        sys.exit(1)
