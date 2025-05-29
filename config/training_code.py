from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import json
from datasets import Dataset

model_id = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

lora_config = LoraConfig(
    r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Now prepare dataset...
# Make a Dataset object (from your cleaned file)

training_args = TrainingArguments(
    output_dir="./custom-multilang-model",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    save_steps=500,
    logging_dir="./logs",
)
lines = []
with open("all_texts.jsonl", "r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f]
dataset = Dataset.from_list(lines)
trainer = Trainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
