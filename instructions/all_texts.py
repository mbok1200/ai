import os
import json
from ebooklib import epub
from ebooklib import ITEM_DOCUMENT  # Додайте цей імпорт
import docx
import pandas as pd
import pdfplumber

from project.helpers.helpers_fn import get_all_files
from PIL import Image
import pytesseract


folder_path = "data/gdrive"
output_file = "all_texts.jsonl"

def extract_epub_text(filepath):
    book = epub.read_epub(filepath)
    text = []
    for item in book.get_items():
        if item.get_type() == ITEM_DOCUMENT:
            text.append(item.get_content().decode("utf-8"))
    return "\n".join(text)

def extract_docx_text(filepath):
    doc = docx.Document(filepath)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_xlsx_text(filepath):
    df = pd.read_excel(filepath, header=None)
    values = df.astype(str).replace("nan", "").values.flatten()
    return "\n".join([v for v in values if v.strip()])
def extract_pdf_text(filepath):
    text = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text.append(page_text)
            else:
                # OCR fallback
                img = page.to_image(resolution=300).original
                ocr_text = pytesseract.image_to_string(img, lang="rus+eng")
                if ocr_text.strip():
                    text.append(ocr_text)
    return "\n".join(text)
added = 0
skipped = 0

with open(output_file, "w", encoding="utf-8") as out_f:
    for filename in get_all_files(folder_path):
        filepath = filename
        text = ""
        if filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
        elif filename.endswith(".epub"):
            text = extract_epub_text(filepath)
        elif filename.endswith(".docx"):
            text = extract_docx_text(filepath)
        elif filename.endswith(".xlsx"):
            text = extract_xlsx_text(filepath)
        elif filename.endswith(".pdf"):
            text = extract_pdf_text(filepath)
        if text.strip():
            out_f.write(json.dumps({"text": text.strip()}, ensure_ascii=False) + "\n")
            print(f"Added: {filename}")
            added += 1
        else:
            print(f"Skipped (empty): {filename}")
            skipped += 1

print(f"\nTotal added: {added}")
print(f"Total skipped: {skipped}")