from huggingface_hub import snapshot_download

# Завантажує модель повністю у кеш і повертає шлях до неї
model_path = snapshot_download(repo_id="google/gemma-2-2b")
# lora_path = snapshot_download(repo_id="bradi12/lora_model_perfs")