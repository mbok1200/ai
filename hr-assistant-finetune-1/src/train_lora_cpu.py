import json
import torch
from datasets import load_dataset
from transformers import EarlyStoppingCallback, AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import PeftModel, get_peft_model, LoraConfig, TaskType

OUTPUT_DIR= "lora_gpt_neo_1_3b_adapter"
MERGED_DIR = "lora_gpt_neo_1_3b_merged"
MODEL_NAME = "google/gemma-2-2b"
DATA_FILE = "data/merged_with_links_is.jsonl"
PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input_text}

### Output:
"""

BASE_INSTRUCTION = (
    "You are an AI assistant. Given an input query, analyze it and generate a structured JSON response.\n"
    "The response must contain an object with the following keys:\n"
    " - 'text': a helpful answer to the query (in the same language as the input).\n"
    " - 'metadata': an object with:\n"
    "    - 'source': a valid URL starting with 'https://'\n"
    "    - 'language': the detected language of the input (ISO 639-1, like 'en' or 'uk')\n"
    " - 'function_method' (optional): if applicable, specify the internal function to call for this query.\n"
)
# 1. Вибір моделі і пристрою
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Токенайзер
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# 3. Модель
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, attn_implementation="eager", torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.config.use_cache = False

# 4. LoRA конфіг
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
   target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.to(device)

# 5. Датасет (приклад)
dataset = load_dataset("json", data_files=DATA_FILE, split="train")
def format_example(example):
    input_text = example.get("input", "").strip()
    output_json = example.get("output", {})

    prompt = PROMPT_TEMPLATE.format(
        instruction=BASE_INSTRUCTION,
        input_text=input_text
    )
    output_text = json.dumps(output_json, ensure_ascii=False)
    full_text = prompt + output_text

    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=512,
        padding="max_length"
    )

    labels = tokenized["input_ids"].copy()
    prompt_token_count = len(tokenizer(prompt, truncation=True, max_length=512)["input_ids"])
    labels[:prompt_token_count] = [-100] * prompt_token_count

    tokenized["labels"] = labels
    return tokenized
tokenized_dataset = dataset.map(format_example, remove_columns=dataset.column_names)

# 6. Параметри тренування
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=2,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",  # або "eval_loss"
    save_total_limit=2,
    logging_dir=f"{OUTPUT_DIR}/logs",
    bf16=device.type == 'cuda',
    fp16=device.type == 'cuda',
    optim="adamw_torch",
    report_to="none"
)

# 8. Тренер
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset.select(range(200)),
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# 9. Запуск тренування
trainer.train()

# Зберігаємо LoRA-адаптер (як є, це завжди FP32)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# === 10. Об’єднання LoRA + base (для інференсу) ===

# Зберегти злиту модель у FP16 (для GPU)
base_model_fp16 = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
lora_model_fp16 = PeftModel.from_pretrained(base_model_fp16, OUTPUT_DIR)
merged_fp16 = lora_model_fp16.merge_and_unload()
merged_fp16.save_pretrained(f"{MERGED_DIR}_fp16")
tokenizer.save_pretrained(f"{MERGED_DIR}_fp16")

# Зберегти злиту модель у FP32 (для CPU)
base_model_fp32 = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
lora_model_fp32 = PeftModel.from_pretrained(base_model_fp32, OUTPUT_DIR)
merged_fp32 = lora_model_fp32.merge_and_unload()
merged_fp32.save_pretrained(f"{MERGED_DIR}_fp32")
tokenizer.save_pretrained(f"{MERGED_DIR}_fp32")