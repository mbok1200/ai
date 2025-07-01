import json
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments, MT5Tokenizer, MT5ForConditionalGeneration
import random
from custom_LR_scheduler_callback import CustomLRSchedulerCallback
model_name = 'google/mt5-small' #'google/flan-t5-small'

import gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
class HRGenDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512, augment=False):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def augment_text(self, text):
        words = text.split()
        if len(words) > 4:
            random.shuffle(words)
            return ' '.join(words)
        return text

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item['prompt']
        target_text = item['response']
        if self.augment and random.random() < 0.5:
            input_text = self.augment_text(input_text)
        inputs = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        targets = self.tokenizer(
            target_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        labels = targets['input_ids'].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100 
        labels = labels.to(torch.long)  # Ensure labels are torch.int64
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        item['labels'] = labels
        return item

def main():
    file_path = 'data/cleaned_data.jsonl'
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = HRGenDataset(file_path, tokenizer, augment=True)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    device = torch.device("cpu")
    
    model = MT5ForConditionalGeneration.from_pretrained(model_name, low_cpu_mem_usage=False).to(device)
    #model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=10,
        gradient_accumulation_steps=4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        warmup_steps=100,
        weight_decay=0.01,
        save_steps=500,
        logging_dir='./logs',
        learning_rate=1e-6,
        logging_steps=10,
        save_total_limit=2,
        eval_strategy="epoch",
        dataloader_pin_memory=False,
        max_grad_norm=0.1,
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[CustomLRSchedulerCallback()]
    )

    trainer.train()  
    trainer.save_model('./results/checkpoint-final')
    tokenizer.save_pretrained('./results/checkpoint-final')

if __name__ == "__main__":
    main()