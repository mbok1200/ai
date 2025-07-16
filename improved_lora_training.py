from datasets import Dataset
import torch, json
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
import random

# 1. Пристрій та налаштування
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Використовується пристрій: {device}")

# 2. Модель та токенізатор
model_name = "google/gemma-2-2b"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
    low_cpu_mem_usage=True
).to(device)

# Оптимізація для навчання
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.config.use_cache = False

# 3. LoRA конфігурація
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 4. Завантаження датасету з обробкою помилок JSON
data = []
with open("merged_with_links_is.jsonl", 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            # Виправляємо можливі помилки JSON
            if '"value_2": 8 "value_3"' in line:
                line = line.replace('"value_2": 8 "value_3"', '"value_2": 8, "value_3"')
            
            # Виправляємо подвійні function_call
            if '"function_call": { "name": "check_have_access"}, "function_call": {' in line:
                line = line.replace('"function_call": { "name": "check_have_access"}, "function_call": {', '"function_call": {')
            
            parsed_data = json.loads(line)
            data.append(parsed_data)
        except json.JSONDecodeError as e:
            print(f"Помилка JSON у рядку {line_num}: {e}")
            print(f"Проблемний рядок: {line[:100]}...")
            continue

print(f"Завантажено {len(data)} прикладів")

# 5. Функція для генерації природних відповідей з функціональністю
def enhance_response_with_function_call(text, function_call_info):
    """
    Генерує більш природну відповідь з інтеграцією викликів функцій
    """
    if not function_call_info or not isinstance(function_call_info, dict):
        return text
    
    function_name = function_call_info.get('name', '')
    arguments = function_call_info.get('arguments', {})
    
    # Шаблони для різних типів функцій
    function_templates = {
        'get_issue_by_date': [
            f"Перевіряю завдання в Redmine на дату {arguments.get('value_1', '')}...",
            f"Шукаю інформацію про завдання на {arguments.get('value_1', '')} у системі...",
            f"Дивлюся, які завдання у вас запланованні на {arguments.get('value_1', '')}..."
        ],
        'get_issue_by_id': [
            f"Знаходжу завдання #{arguments.get('value_1', '')} у Redmine...",
            f"Переглядаю деталі завдання {arguments.get('value_1', '')}...",
            f"Отримую інформацію про завдання {arguments.get('value_1', '')}..."
        ],
        'get_issue_by_name': [
            f"Шукаю завдання '{arguments.get('value_1', '')}' у системі...",
            f"Знаходжу інформацію про {arguments.get('value_1', '')}...",
            f"Переглядаю завдання {arguments.get('value_1', '')}..."
        ],
        'fill_issue_hours': [
            f"Заповнюю {arguments.get('value_2', '')} годин для завдання {arguments.get('value_1', '')}...",
            f"Додаю відпрацьований час до завдання {arguments.get('value_1', '')}...",
            f"Записую {arguments.get('value_2', '')} годин роботи..."
        ],
        'get_issue_status': [
            f"Перевіряю поточний статус завдання {arguments.get('value_1', '')}...",
            f"Дивлюся статус {arguments.get('value_1', '')} у Redmine...",
            f"Отримую інформацію про стан завдання..."
        ],
        'create_issue': [
            f"Створюю нове завдання '{arguments.get('value_1', '')}'...",
            f"Додаю завдання до системи з описом: {arguments.get('value_2', '')}...",
            f"Формую нове завдання для проекту..."
        ]
    }
    
    # Варіанти закінчення
    endings = [
        "Якщо потрібна додаткова інформація, звертайтеся!",
        "Чи можу ще чимось допомогти?",
        "Потрібно щось ще з цього приводу?",
        "Якщо є питання, я готовий допомогти!",
        "Звертайтеся, якщо знадобиться підтримка!"
    ]
    
    if function_name in function_templates:
        action_text = random.choice(function_templates[function_name])
        ending = random.choice(endings)
        
        # Комбінуємо оригінальний текст з контекстом функції
        enhanced_text = f"{action_text} {text} {ending}"
        return enhanced_text
    
    return text

# 6. Очищення та покращення датасету
def clean_and_enhance_data(data_list):
    """Очищає check_have_access і покращує відповіді"""
    enhanced_data = []
    
    for item in data_list:
        if 'output' in item and isinstance(item['output'], dict):
            metadata = item['output'].get('metadata', {})
            original_text = item['output'].get('text', '')
            
            # Видаляємо check_have_access але зберігаємо інші function_call
            function_call = None
            if 'function_call' in metadata:
                if isinstance(metadata['function_call'], dict):
                    if metadata['function_call'].get('name') != 'check_have_access':
                        function_call = metadata['function_call']
                        # Нормалізуємо arguments
                        if 'arguments' in function_call:
                            args = function_call['arguments']
                            for key, value in args.items():
                                args[key] = str(value)
                elif isinstance(metadata['function_call'], str) and metadata['function_call'] != 'check_have_access':
                    function_call = {'name': metadata['function_call']}
            
            # Покращуємо текст відповіді
            enhanced_text = enhance_response_with_function_call(original_text, function_call)
            
            # Створюємо новий item
            new_item = {
                'instruction': item.get('instruction', ''),
                'input': item.get('input', ''),
                'output': {
                    'text': enhanced_text,
                    'metadata': {
                        'language': metadata.get('language', 'ua')
                    }
                }
            }
            
            # Додаємо function_call якщо є
            if function_call:
                new_item['output']['metadata']['function_call'] = function_call
            
            enhanced_data.append(new_item)
    
    return enhanced_data

# Покращуємо датасет
data = clean_and_enhance_data(data)
dataset = Dataset.from_list(data)

print(f"Приклад покращеної відповіді:")
print(f"Input: {data[0]['input']}")
print(f"Output: {data[0]['output']['text']}")
print(f"Function: {data[0]['output']['metadata'].get('function_call', 'None')}")

# 7. Покращений промпт-шаблон для HR Assistant
SYSTEM_PROMPT = """Ти досвідчений HR-асистент для роботи з системою управління проектами Redmine. 
Твоя роль - допомагати користувачам ефективно працювати з завданнями, проектами та часом.

Ти можеш:
- Знаходити та аналізувати завдання за різними критеріями
- Заповнювати відпрацьований час
- Створювати нові завдання
- Призначати завдання користувачам
- Отримувати інформацію з Wiki
- Відстежувати статуси завдань та користувачів

Відповідай природно, професійно та доброзичливо. Завжди готовий допомогти!"""

PROMPT_TEMPLATE = """<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{input_text}<|im_end|>
<|im_start|>assistant
{output_text}<|im_end|>"""

def format_example(example):
    # Отримуємо текст відповіді
    output_text = example['output'].get('text', '')
    
    # Формуємо повний текст для навчання
    full_text = PROMPT_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT,
        input_text=example.get('input', ''),
        output_text=output_text
    )
    
    # Токенізація
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=1024,  # Збільшуємо довжину для більш детальних відповідей
        padding="max_length"
    )
    
    # Створюємо labels (маскуємо все крім відповіді асистента)
    input_ids = tokenized["input_ids"]
    labels = input_ids.copy()
    
    # Знаходимо початок відповіді асистента
    assistant_start = "<|im_start|>assistant"
    assistant_tokens = tokenizer.encode(assistant_start, add_special_tokens=False)
    
    # Маскуємо все до відповіді асистента
    for i in range(len(input_ids) - len(assistant_tokens)):
        if input_ids[i:i+len(assistant_tokens)] == assistant_tokens:
            labels[:i+len(assistant_tokens)] = [-100] * (i + len(assistant_tokens))
            break
    
    return {
        "input_ids": input_ids,
        "attention_mask": tokenized["attention_mask"],
        "labels": labels
    }

# Препроцесинг датасету
print("Препроцесинг датасету...")
tokenized_dataset = dataset.map(format_example, remove_columns=dataset.column_names)

# Розділяємо на train/eval
train_size = int(0.9 * len(tokenized_dataset))
train_dataset = tokenized_dataset.select(range(train_size))
eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))

print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

# 8. Покращені навчальні параметри
training_args = TrainingArguments(
    output_dir="./lora_enhanced_hr_assistant",
    per_device_train_batch_size=1,  # Зменшуємо через збільшену довжину
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,  # Збільшуємо для компенсації меншого batch_size
    learning_rate=5e-5,  # Трохи менша learning rate для стабільності
    num_train_epochs=15,
    warmup_steps=200,
    logging_dir="./logs",
    logging_steps=10,
    eval_steps=50,
    save_steps=100,
    save_total_limit=3,
    eval_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    bf16=device.type == 'cuda',
    fp16=False,
    optim="adamw_torch",
    dataloader_drop_last=True,
    remove_unused_columns=False,
    report_to="none",
    seed=42,
    data_seed=42,
    lr_scheduler_type="cosine"
)

# 9. Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False,
    pad_to_multiple_of=8 if device.type == 'cuda' else None
)

# 10. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# 11. Функція для тестування моделі під час навчання
def test_model_generation():
    """Тестує генерацію моделі"""
    test_input = "які в мене завдання на сьогодні"
    test_prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{test_input}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("<|im_start|>assistant\n")[-1]
    print(f"\nТест генерації:")
    print(f"Запит: {test_input}")
    print(f"Відповідь: {response}")

# 12. Запуск навчання з періодичним тестуванням
print("Початок навчання...")

# Тестуємо модель до навчання
print("Модель ДО навчання:")
test_model_generation()

# Навчання
trainer.train()

print("Навчання завершено!")
