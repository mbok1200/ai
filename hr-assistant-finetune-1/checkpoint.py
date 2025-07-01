from transformers import AutoModelForSeq2SeqLM

# Завантажте оригінальний чекпойнт (НЕ -cpu, НЕ -realsave)
model = AutoModelForSeq2SeqLM.from_pretrained('./results/checkpoint-500', device_map=None)
model.save_pretrained('./results/checkpoint-500-cpu')