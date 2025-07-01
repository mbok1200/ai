import json

input_path = "merged.jsonl"
output_path = "merged_min.jsonl"

with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    buffer = ""
    for line in infile:
        if not line.strip() or line.strip().startswith("//"):
            continue
        buffer += line
        # Якщо кількість відкриваючих і закриваючих дужок співпадає — об'єкт завершено
        if buffer.count("{") == buffer.count("}"):
            try:
                obj = json.loads(buffer)
                outfile.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
            except Exception as e:
                print("Помилка парсингу:", e)
                print(buffer)
            buffer = ""
print("Готово! Мінімізований JSONL у", output_path)