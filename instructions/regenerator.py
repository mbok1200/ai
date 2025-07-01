import json
from config.helpers.helpers_fn import split_dialog, get_all_files
class Regenerator:
    def __init__(self):
        self.name = "Regenerator"
        self.description = "Regenerates the text and answer from the raw data format to a clean format."
        self.version = "1.0.0"
        self.save_path = "cleaned_data.jsonl"
        self.files = get_all_files("data/dataset")
        print(f"Found {len(self.files)} files in the dataset directory.")
    def run(self):
        print(f"Running {self.name}...")
        if not self.files:
            print("No files found in the dataset directory.")
            return
        for file in self.files:
            print(f"Processing file: {file}")
            try:
                self.save(file)
                print(f"File {file} processed successfully.")
            except Exception as e:
                print(f"Error processing file {file}: {e}")
        print(f"{self.name} completed. Cleaned data saved to {self.save_path}.")
    def save(self, data):
        with open(data, "r", encoding="utf-8") as fin, open(self.save_path, "a+", encoding="utf-8") as fout:
            for line in fin:                
                raw_entry = json.loads(line)
                new_entry = split_dialog(raw_entry)
                fout.write(json.dumps(new_entry, ensure_ascii=False) + "\n")
def main ():
    regenerator = Regenerator()
    regenerator.run()
if __name__ == "__main__":
    main()