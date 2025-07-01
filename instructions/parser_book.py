import json

INPUT_MD = "data/structured/Adizes_Upravlyaya-izmeneniyami.9riHkA.370675.md"
OUTPUT_JSONL = "data/adizes_book.jsonl"
SOURCE_URL = "https://www.litres.ru/ichak-adizes/upravlyaya-izmeneniyami-mastering-change/"

def parse_md(md_path):
    with open(md_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    dataset = []

    # 1. Annotation (перший великий блок)
    annotation = []
    for line in lines:
        if line.lower().startswith("annotation"):
            annotation.append(line)
        elif annotation and not line.startswith("Annotation") and not line.startswith("Из этой книги"):
            annotation.append(line)
        elif annotation and line.startswith("Из этой книги"):
            break
    annotation_text = " ".join(annotation).replace("Annotation", "").strip()
    if annotation_text:
        dataset.append({
            "instruction": "Проаналізуй цей запит",
            "input": "Анотація до книги Адизеса 'Управляя изменениями'",
            "output": annotation_text,
            "metadata": {
                "language": "ru",
                "source": SOURCE_URL,
                "category": "Менеджмент",
                "subcategory": "Анотація"
            }
        })

    # 2. Основні тези (після "Из этой книги вы узнаете")
    main_points = []
    in_points = False
    for line in lines:
        if line.startswith("Из этой книги вы узнаете"):
            in_points = True
            continue
        if in_points:
            if line and not line.startswith("языки."):
                main_points.append(line)
            else:
                break
    main_points_text = " ".join(main_points).strip()
    if main_points_text:
        dataset.append({
            "instruction": "Проаналізуй цей запит",
            "input": "Основні тези книги Адизеса 'Управляя изменениями'",
            "output": main_points_text,
            "metadata": {
                "language": "ru",
                "source": SOURCE_URL,
                "category": "Менеджмент",
                "subcategory": "Тези"
            }
        })

    # 3. Зміст (все, що виглядає як перелік розділів)
    toc = []
    in_toc = False
    for line in lines:
        if "Первая беседа" in line or "Вторая беседа" in line or "Третья беседа" in line:
            in_toc = True
        if in_toc:
            if line and not line.startswith("©") and not line.startswith("Published") and not line.startswith("First English edition"):
                toc.append(line)
            if "Жизненный цикл" in line:
                break
    toc_text = " ".join(toc).strip()
    if toc_text:
        dataset.append({
            "instruction": "Проаналізуй цей запит",
            "input": "Зміст книги Адизеса 'Управляя изменениями'",
            "output": toc_text,
            "metadata": {
                "language": "ru",
                "source": SOURCE_URL,
                "category": "Менеджмент",
                "subcategory": "Зміст"
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