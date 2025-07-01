import re
import json

# Шлях до файлу
filename = "data/structured/Agile._._2020.md"

# Джерело
source_url = "https://drive.google.com/file/d/1cVNOKcuVMUQT9vbVI8qor_zDavluFFdV/view?usp=sharing"

# Мова
language = "ru"

# Категорія (можна визначати автоматично або вручну)
category = "Agile"
subcategory = ""

# Зчитування файлу
with open(filename, encoding="utf-8") as f:
    text = f.read()

# Розбиваємо на глави (Глава N. Назва)
chapter_pattern = re.compile(r"(Глава\s+\d+\..+?)(?=Глава\s+\d+\.|$)", re.DOTALL)
chapters = chapter_pattern.findall(text)

dataset = []

for chapter in chapters:
    # Витягуємо назву глави
    chapter_title_match = re.match(r"Глава\s+(\d+)\.\s*(.+)", chapter)
    if chapter_title_match:
        chapter_num = chapter_title_match.group(1)
        chapter_title = chapter_title_match.group(2).strip()
    else:
        chapter_num = ""
        chapter_title = ""

    # Розбиваємо на підглави (по заголовках, наприклад, "ЖИЗНЕННЫЙ ЦИКЛ", "ПРОФЕССИОНАЛИЗМ" тощо)
    subchapter_pattern = re.compile(r"([А-ЯA-Z][А-ЯA-Z\s\-]+)\n(.+?)(?=\n[А-ЯA-Z][А-ЯA-Z\s\-]+\n|$)", re.DOTALL)
    subchapters = subchapter_pattern.findall(chapter)

    if not subchapters:
        # Якщо підглав немає, беремо всю главу як одну інструкцію
        dataset.append({
            "instruction": f"Стисло перекажи зміст глави '{chapter_title}'",
            "input": chapter_title,
            "output": chapter.strip(),
            "metadata": {
                "language": language,
                "source": source_url,
                "category": category,
                "subcategory": chapter_title
            }
        })
    else:
        for subchapter_title, subchapter_text in subchapters:
            # Обрізаємо зайві пробіли
            subchapter_title = subchapter_title.strip()
            subchapter_text = subchapter_text.strip()
            if len(subchapter_text) < 100:  # Пропускаємо дуже короткі підглави
                continue
            dataset.append({
                "instruction": f"Стисло перекажи зміст підглави '{subchapter_title}'",
                "input": subchapter_title,
                "output": subchapter_text,
                "metadata": {
                    "language": language,
                    "source": source_url,
                    "category": category,
                    "subcategory": chapter_title
                }
            })

# Зберігаємо у файл
with open("agile_dataset.jsonl", "w", encoding="utf-8") as f:
    for item in dataset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Збережено {len(dataset)} записів у agile_dataset.jsonl")