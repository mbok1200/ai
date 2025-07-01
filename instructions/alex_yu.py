import re
import json

def parse_sections(text):
    """
    Парсить розділи з markdown-файлу, повертає список словників з інструкціями, інпутами, аутпутами та метаданими.
    """
    # Розділи визначаються як CHAPTER N: TITLE
    chapter_pattern = re.compile(r'^CHAPTER\s+\d+:\s+(.+)$', re.MULTILINE)
    chapters = [(m.start(), m.group(1).strip()) for m in chapter_pattern.finditer(text)]
    sections = []
    for idx, (start, title) in enumerate(chapters):
        end = chapters[idx + 1][0] if idx + 1 < len(chapters) else len(text)
        section_text = text[start:end].strip()
        sections.append({'title': title, 'text': section_text})
    return sections

def extract_instruction(section):
    """
    Витягує інструкцію, інпут, аутпут з тексту розділу.
    """
    # Інструкція — це перший абзац після заголовка
    lines = section['text'].split('\n')
    instruction = lines[1].strip() if len(lines) > 1 else ""
    # Інпут — питання або приклад, якщо є
    input_ = ""
    output = ""
    # Витягуємо приклад, якщо є (наприклад, Example: Estimate Twitter QPS and storage requirements)
    example_match = re.search(r'Example:(.*?)(Tips|Reference materials|$)', section['text'], re.DOTALL)
    if example_match:
        example_text = example_match.group(1).strip()
        # Витягуємо перший рядок як інпут, решту як аутпут
        example_lines = [l.strip() for l in example_text.split('\n') if l.strip()]
        if example_lines:
            input_ = example_lines[0]
            output = '\n'.join(example_lines[1:]).strip()
    # Якщо немає прикладу, беремо перший абзац як інпут, другий як аутпут
    if not input_:
        paragraphs = [p.strip() for p in section['text'].split('\n\n') if p.strip()]
        if len(paragraphs) > 1:
            input_ = paragraphs[0]
            output = paragraphs[1]
    return instruction, input_, output

def get_category(title):
    """
    Витягує категорію та підкатегорію з назви розділу.
    """
    # Категорія — перше слово, підкатегорія — решта
    parts = title.split()
    category = parts[0] if parts else ""
    subcategory = ' '.join(parts[1:]) if len(parts) > 1 else ""
    return category, subcategory

def main():
    # Зчитати файл
    with open('data/structured/Alex_Yu_System_Design_Interview_An_Insider_s_Guide_Independently.md', encoding='utf-8') as f:
        text = f.read()

    sections = parse_sections(text)
    dataset = []
    for section in sections:
        instruction, input_, output = extract_instruction(section)
        category, subcategory = get_category(section['title'])
        if instruction and input_ and output:
            item = {
                "instruction": instruction,
                "input": input_,
                "output": output,
                "metadata": {
                    "language": "ua",
                    "source": "https://drive.google.com/file/d/1imSknPMQf7uEORLWPCgs2KE-rVWh1DKS/view?usp=sharing",
                    "category": category,
                    "subcategory": subcategory
                }
            }
            dataset.append(item)

    # Записати у файл
    with open('data/alex_yu_system_design_dataset.jsonl', 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    main()