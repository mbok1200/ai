import json
from pathlib import Path
import re

from docx import Document
import pandas as pd
from pypdf import PdfReader
# from googleapiclient.discovery import build
def sanitize_filename(filename):
    return re.sub(r'[\\/:"*?<>|]+', '_', filename)
# Load file map from JSON
def load_file_map():
    try:
        with open('/home/mikola/projects/ai/config/gdrive_file_map.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return {}

def get_all_files(root_dir):
    return [str(p) for p in Path(root_dir).rglob("*") if p.is_file()]

def normalize_text(text: str) -> str:
    if not text:
        return ""

    # Remove control characters and excessive whitespace
    text = re.sub(r"[\x00-\x1F\x7F]", "", text)  # Remove non-printable characters
    text = re.sub(r"\s+", " ", text)             # Collapse all whitespace to single spaces
    text = re.sub(r"^\s+|\s+$", "", text)        # Strip leading/trailing whitespace

    # Optional: Remove likely metadata lines (e.g., headers/footers from PDFs)
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        # Skip if line is too short or looks like metadata
        if len(line) < 3:
            continue
        if re.search(r"page\s*\d+", line, re.IGNORECASE):
            continue
        if re.match(r"^\W+$", line):  # line is only punctuation or symbols
            continue
        cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)

    return cleaned_text
def extract_text_from_local_file(file_path):
    ext = Path(file_path).suffix.lower()

    try:
        if ext == ".pdf":
            pdf_reader = PdfReader(file_path)
            text = "\n".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
            return text

        elif ext in [".docx", ".doc"]:
            doc = Document(file_path)
            return "\n".join(para.text for para in doc.paragraphs)

        elif ext in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

        elif ext == ".csv":
            df = pd.read_csv(file_path)
            return df.to_csv(index=False)

        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(file_path)
            return df.to_csv(index=False)

        else:
            return ""

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return ""
