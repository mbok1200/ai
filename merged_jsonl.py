import os

input_folder = "data/jsonl"  # Вкажіть вашу папку з .jsonl файлами
output_file = "merged.jsonl"

with open(output_file, "w", encoding="utf-8") as outfile:
    for filename in os.listdir(input_folder):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(input_folder, filename)
            with open(file_path, "r", encoding="utf-8") as infile:
                for line in infile:
                    if line.strip():
                        outfile.write(line.rstrip() + "\n")
print(f"Об'єднано всі .jsonl у {output_file}")