import json

def split_text_and_questions(text):
    question_markers = [
        "Вопросы:", "Вопроси:", "Питання:", "Questions:", "Вопрос:", "Питання", "Вопрос", "Q:", "Q.", "Q-", "Вопроси", "Питання"
    ]
    idx = -1
    marker = ""
    for m in question_markers:
        idx = text.find(m)
        if idx != -1:
            marker = m
            break
    if idx == -1:
        # Якщо маркер не знайдено — ділимо текст навпіл по найближчій крапці
        mid = len(text) // 2
        left = text.rfind('.', 0, mid)
        right = text.find('.', mid)
        # Вибираємо найближчу до середини крапку
        if left == -1 and right == -1:
            split_idx = mid
        elif left == -1:
            split_idx = right + 1
        elif right == -1:
            split_idx = left + 1
        else:
            split_idx = left + 1 if (mid - left) <= (right - mid) else right + 1
        main_text = text[:split_idx].strip()
        questions = text[split_idx:].strip()
        return main_text, questions
    else:
        main_text = text[:idx].strip()
        questions = text[idx+len(marker):].strip()
        return main_text, questions

output = []
with open("all_texts.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip() or line.strip().startswith("//"):
            continue
        obj = json.loads(line)
        text = obj["text"]
        main_text, questions = split_text_and_questions(text)
        output.append({"text": main_text, "answer": questions})

with open("all_texts_qa.jsonl", "w", encoding="utf-8") as f:
    for item in output:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")