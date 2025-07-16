import re
from datasets import Dataset
import torch, json
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, EarlyStoppingCallback
from peft import get_peft_model, LoraConfig, TaskType

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "google/gemma-2-2b"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
    low_cpu_mem_usage=True
).to(device)

model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.config.use_cache = False

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="all",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.config.pad_token_id = tokenizer.pad_token_id
data = []
error_lines = []

with open("/content/drive/MyDrive/train/merged.jsonl", 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue        
        try:
            parsed_data = json.loads(line)
            data.append(parsed_data)
        except json.JSONDecodeError as e:
            error_lines.append((line_num, line, str(e)))

if error_lines:
    print(f"Кількість рядків з помилками JSON: {len(error_lines)}")
    for ln, content, err in error_lines[:10]:
        print(f"Рядок {ln}: {err}")
        print(f"Текст: {content[:100]}...")

def restructure_function_calls(item):
    metadata = item.get("output", {}).get("metadata", {})
    if not isinstance(metadata, dict):
        return item

    fc_list = []
    for k in list(metadata.keys()):
        if k.startswith("function_call"):
            fc = metadata.pop(k)
            if isinstance(fc, dict):
                fc_list.append(fc)

    if fc_list:
        metadata["function_calls"] = fc_list

    return item

data = [restructure_function_calls(item) for item in data]
dataset = Dataset.from_list(data)

PROMPT_TEMPLATE = """
Інструкція: Ти HR-асистент для Redmine. Аналізуй запит користувача і генеруй відповідь у форматі JSON з викликом функції і текстовим повідомленням.

Запит:
{input_text}

Відповідь:
{output_text}
"""
def format_example(example):
    output_dict = {
        "text": "",
        "function_calls": []
    }

    if isinstance(example.get('output'), dict):
        output_dict["text"] = example['output'].get('text', '')
        metadata = example['output'].get('metadata', {})

        for key in metadata:
            if key.startswith('function_call'):
                function_info = metadata[key]
                if isinstance(function_info, dict):
                    output_dict["function_calls"].append(function_info)

    output_text = json.dumps(output_dict, ensure_ascii=False)

    full_text = PROMPT_TEMPLATE.format(
        input_text=example.get('input', ''),
        output_text=output_text
    )

    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=1024,
        padding="max_length"
    )

    input_ids = tokenized["input_ids"]
    labels = input_ids.copy()

    response_start = "Відповідь:"
    response_tokens = tokenizer.encode(response_start, add_special_tokens=False)

    index = -1
    for i in range(len(input_ids) - len(response_tokens)):
        if input_ids[i:i+len(response_tokens)] == response_tokens:
            index = i
            break

    if index != -1:
        labels[:index + len(response_tokens)] = [-100] * (index + len(response_tokens))

    return {
        "input_ids": input_ids,
        "attention_mask": tokenized["attention_mask"],
        "labels": labels
    }

tokenized_dataset = dataset.map(format_example, remove_columns=dataset.column_names)
train_size = int(0.9 * len(tokenized_dataset))
train_dataset = tokenized_dataset.select(range(train_size))
eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
training_args = TrainingArguments(
    output_dir="./lora_redmine_assistant_clean",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    num_train_epochs=20,
    warmup_steps=100,
    logging_dir="./logs",
    logging_steps=10,
    eval_steps=100,
    save_steps=200,
    save_total_limit=3,
    eval_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    bf16=device.type == 'cuda',
    fp16=False,
    optim="adamw_torch",
    dataloader_drop_last=True,
    remove_unused_columns=False,
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False,
    pad_to_multiple_of=8 if device.type == 'cuda' else None
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()

model.save_pretrained("./lora_only", save_adapter=True)

