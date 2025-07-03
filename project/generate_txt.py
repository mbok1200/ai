from helpers.helpers_fn import extract_paragraphs, get_all_files
import os
OUTPUT_SAVE_PATH = "data/text"
class GenerateTxt:
    def __init__(self, path):
        self.path = path
        self.files = get_all_files(path)

    def run(self):
        for file in self.files:
            filename = os.path.relpath(file, self.path)
            print(f"Processing file: {file} {filename}")
            paragraphs = extract_paragraphs(file, False)
            txt_path = os.path.splitext(filename)[0] + ".txt"
            full_txt_path = os.path.join(OUTPUT_SAVE_PATH, txt_path)
            os.makedirs(os.path.dirname(full_txt_path), exist_ok=True)  # Ensure folder exists
            with open(full_txt_path, "w", encoding="utf-8") as f:
                for paragraph in paragraphs:
                    f.write(paragraph + "\n\n")
if __name__ == "__main__":
    pdf_path = "data/gdrive"  # Replace with your actual path
    generator = GenerateTxt(pdf_path)
    generator.run()
    print("Text files generated successfully.")