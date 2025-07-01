import json

INPUT_MD = "data/structured/Conflict management -Level 2  Team work.md"
OUTPUT_JSONL = "data/Conflict management -Level 2  Team work.jsonl"
SOURCE_URL = "https://docs.google.com/document/d/1BWKP9OineazCcZS9-3C1aH8cIbF2zDFLKOfz063pNEw/edit?usp=drive_link"

def parse_md(md_path):
    with open(md_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    dataset = []

    # Основні блоки
    skill_for = lines[0].replace("Для кого данный навык:", "").strip() if lines[0].startswith("Для кого") else ""
    skill_purpose = lines[1].replace("Для чего данный навык:", "").strip() if lines[1].startswith("Для чего") else ""
    skill_name = lines[2]
    skill_desc = lines[3]
    # Принципи та кейси
    principles = []
    cases = []
    for i, line in enumerate(lines):
        if line.startswith("Принцип"):
            principles.append(line)
        if line.startswith("Рассмотрим кейс:"):
            case = line
            # Додаємо наступний рядок як продовження кейсу, якщо він є
            if i+1 < len(lines) and not lines[i+1].startswith("Принцип") and not lines[i+1].startswith("Рассмотрим кейс:"):
                case += " " + lines[i+1]
            cases.append(case)

    # 1. Загальний опис навику
    dataset.append({
        "instruction": "Проаналізуй цей запит",
        "input": f"{skill_name}. Для кого: {skill_for}. Для чого: {skill_purpose}",
        "output": f"{skill_desc}",
        "metadata": {
            "language": "ua",
            "source": SOURCE_URL,
            "category": "Effective Communication",
            "subcategory": "Active Listening"
        }
    })

    # 2. Принципи
    for p in principles:
        dataset.append({
            "instruction": "Проаналізуй цей запит",
            "input": f"{skill_name}. {p}",
            "output": "Це один із принципів активного слухання, який допомагає краще зрозуміти співрозмовника та уникнути помилок у роботі.",
            "metadata": {
                "language": "ua",
                "source": SOURCE_URL,
                "category": "Effective Communication",
                "subcategory": "Active Listening"
            }
        })

    # 3. Кейси
    for c in cases:
        dataset.append({
            "instruction": "Проаналізуй цей запит",
            "input": f"{skill_name}. {c}",
            "output": "Цей кейс ілюструє важливість уточнення деталей та розуміння контексту завдання.",
            "metadata": {
                "language": "ua",
                "source": SOURCE_URL,
                "category": "Effective Communication",
                "subcategory": "Active Listening"
            }
        })

    return dataset

def save_jsonl(dataset, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    dataset = parse_md(INPUT_MD)
    save_jsonl(dataset, OUTPUT_JSONL)
    print(f"Збережено {len(dataset)} записів у {OUTPUT_JSONL}")