import os
import openai
import json
import time
from dotenv import load_dotenv

from helpers.helpers_fn import extract_paragraphs, get_all_files
load_dotenv(".env")
# ============ КОНФІГУРАЦІЯ ================
openai.api_key = os.getenv("OPENAI_API_KEY")
PDF_PATH = "/home/mikola/projects/ai/data/gdrive/"
OUTPUT_PATH = "data/dataset"
MODEL = "gpt-4"
SYSTEM_PROMPT = "You are a strict corporate assistant. You answer clearly, officially, based only on the text provided."
# =========================================


def generate_qa(paragraph):
    prompt = (
        "Text:\n"
        f"\"\"\"{paragraph}\"\"\"\n\n"
        "Based on this text, generate:\n"
        "1. Short, clear user question\n"
        "2. The assistant's response in a strict style\n\n"
        "Response format:\n"
        "### User:\n<question>\n\n### Assistant:\n<answer>"
    )

    try:
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        print("💬 Згенеровано QA:", response["choices"][0]["message"]["content"].strip())
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("Error:", e)
        return None

def format_to_jsonl(qa_text):
    print("DEBUG QA TEXT:", qa_text)
    # Підтримка обох варіантів написання
    if "### User:" in qa_text and ("### Assistant:" in qa_text or "### Assistant:" in qa_text):
        try:
            if "### Assistant:" in qa_text:
                user = qa_text.split("### User:")[1].split("### Assistant:")[0].strip()
                assistant = qa_text.split("### Assistant:")[1].strip()
            else:
                user = qa_text.split("### User:")[1].split("### Assistant:")[0].strip()
                assistant = qa_text.split("### Assistant:")[1].strip()
            text_block = f"### User:\n{user}\n\n### Assistant:\n{assistant}"
            print("📦 Форматування в JSONL:", text_block)
            return json.dumps({"text": text_block}, ensure_ascii=False)
        except Exception as e:
            print("Parsing error:", e)
            return None
    print("❌ Не знайдено потрібних роздільників у відповіді!")
    return None

def generated_jsonl_by_path(path=PDF_PATH):
    if not os.path.exists(path):
        print(f"❌ Файл не знайдено: {path}")
        return
    filename = f"{os.path.splitext(os.path.basename(path))[0]}.jsonl"
    paragraphs = extract_paragraphs(path)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    with open(f"{OUTPUT_PATH}/{filename}", "w", encoding="utf-8") as f:
        for i, para in enumerate(paragraphs):
            print(f"🔄 Обробка {i+1}/{len(paragraphs)}...")
            qa = generate_qa(para)
            if qa:
                jsonl = format_to_jsonl(qa)
                if jsonl:
                    f.write(jsonl + "\n")
            time.sleep(1.5)
    print(f"✅ Готово! Датасет збережено у: {OUTPUT_PATH}")
if __name__ == "__main__":
    files = get_all_files(PDF_PATH)
    for file_path in files:
        filename = os.path.relpath(file_path, PDF_PATH) 
        generated_jsonl_by_path(file_path)
