from pathlib import Path
from docx import Document
import PyPDF2
import json
import html2text
import pandas as pd
import re

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
def extract_text_from_file(file_path: Path) -> str:
    ext = file_path.suffix.lower()
    try:
        if ext == ".txt" or ext == ".md":
            return file_path.read_text(encoding='utf-8', errors='ignore')

        elif ext == ".docx":
            doc = Document(file_path)
            return normalize_text("\n".join([para.text for para in doc.paragraphs]))

        elif ext == ".pdf":
            text = ""
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ""
            return normalize_text(text)

        elif ext == ".html" or ext == ".htm":
            html = file_path.read_text(encoding='utf-8', errors='ignore')
            return html2text.html2text(html)

        elif ext == ".json":
            data = json.loads(file_path.read_text(encoding='utf-8', errors='ignore'))
            return normalize_text(json.dumps(data, indent=2))
        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(file_path, sheet_name=None)  # Load all sheets
            text = ""
            for sheet, content in df.items():
                text += f"--- Sheet: {sheet} ---\n"
                text += content.fillna("").astype(str).to_string(index=False)
                text += "\n\n"
            return normalize_text(text)
        else:
            return f""  # Skip unsupported file types

    except Exception as e:
        print(f"[ERROR] Failed to extract from {file_path}: {e}")
        return ""

def extract_texts_from_folder(folder_path: str) -> dict:
    folder = Path(folder_path)
    all_texts = {}
    print(f"[INFO] Found folder: {folder.rglob("*")}")
    for file_path in folder.rglob("*"):
        if file_path.is_file():
            print(f"[DEBUG] Reading: {file_path}") 
            text = extract_text_from_file(file_path)
            if text.strip():
                all_texts[str(file_path)] = text
        elif file_path.is_dir():
            print(f"[INFO] Found folder: {file_path}")
            
    return all_texts

# === USAGE ===
def get_all_files(root_dir):
    return [str(p) for p in Path(root_dir).rglob("*") if p.is_file()]

source_folder = Path("data/gdrive").resolve()  # Replace with your folder path
print("Resolved folder path:", Path("data/gdrive").resolve().exists())
texts = extract_texts_from_folder(Path(source_folder).resolve())

# Optional: Save to JSONL for training
with open("all_texts.jsonl", "w", encoding="utf-8") as f:
    for file, content in texts.items():
        f.write(json.dumps({"source": file, "text": content}, ensure_ascii=False) + "\n")
