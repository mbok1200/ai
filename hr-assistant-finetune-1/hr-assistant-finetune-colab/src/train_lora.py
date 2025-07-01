from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType

# 1. Базова модель
model_name = "mistralai/Mistral-7B-v0.1" 
device = torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=False).to(device)

# 2. LoRA конфіг
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, peft_config)

# 3. Завантаження даних
dataset = load_dataset("json", data_files="data/merged_min.jsonl", split="train")

# 4. Препроцесинг
def format_example(example):
    source = ""
    if "metadata" in example and example["metadata"] and "source" in example["metadata"]:
        source = example["metadata"]["source"]
    output = example["output"]
    if source and output:
        output = f"{output}\n\nДетальніше: {source}"
    text = f"### Інструкція:\n{example['instruction']}\n\n### Вхід:\n{example['input']}\n\n### Вихід:\n{output}"
    return {"input_ids": tokenizer(text, truncation=True, max_length=512, padding="max_length")["input_ids"]}

tokenized_dataset = dataset.map(format_example)

# 5. Навчання
training_args = TrainingArguments(
    output_dir="./lora_model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    bf16=True,
    optim="adamw_torch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

trainer.train()