import re
import json

INPUT_MD = "data/structured/100 Most Useful Productivity Hacks.md"
OUTPUT_JSONL = "data/productivity_hacks.jsonl"
SOURCE_URL = "https://drive.google.com/file/d/1IAqxLYLJAwvNIY3XNJQ5NxjOMQOQ74oo/view?usp=drive_link"

def parse_md(md_path):
    with open(md_path, encoding="utf-8") as f:
        lines = f.readlines()

    hacks = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Знаходимо початок хаку
        if line.startswith("## "):
            title = line[3:].strip()
            # Збираємо опис
            desc_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("## ") and not re.match(r"^[A-Za-z ]+$", lines[i].strip()):
                if lines[i].strip() and not lines[i].strip().startswith("Next") and not lines[i].strip().startswith("previous") and not lines[i].strip().startswith("index"):
                    desc_lines.append(lines[i].strip())
                i += 1
            description = " ".join(desc_lines).strip()
            # Категорія (наступний не-порожній рядок, який не починається з ##)
            category = ""
            subcategory = ""
            while i < len(lines):
                cat_line = lines[i].strip()
                if cat_line and not cat_line.startswith("## ") and not cat_line.startswith("Next") and not cat_line.startswith("previous") and not cat_line.startswith("index"):
                    category = cat_line.capitalize()
                    break
                i += 1
            # Формуємо структуру
            hacks.append({
                "instruction": "Проаналізуй цей запит",
                "input": f"{title}, {category.lower()}, productivity",
                "output": description,
                "metadata": {
                    "language": "ua",
                    "source": SOURCE_URL,
                    "category": category,
                    "subcategory": title
                }
            })
        else:
            i += 1
    return hacks

def save_jsonl(hacks, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for hack in hacks:
            f.write(json.dumps(hack, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    hacks = parse_md(INPUT_MD)
    save_jsonl(hacks, OUTPUT_JSONL)
    print(f"Збережено {len(hacks)} записів у {OUTPUT_JSONL}")