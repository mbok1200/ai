import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import subprocess
import tempfile

def download_ollama_model_weights(model_name="gemma2:2b"):
    """
    Витягує ваги моделі з Ollama в HuggingFace формат
    """
    print(f"📥 Завантажуємо модель {model_name} з Ollama...")
    
    # Мапінг Ollama моделей на HuggingFace
    model_mapping = {
        "gemma2:2b": "google/gemma-2-2b",
        "gemma2:9b": "google/gemma-2-9b", 
        "llama3.1:8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "qwen2:7b": "Qwen/Qwen2-7B-Instruct"
    }
    
    hf_model_name = model_mapping.get(model_name)
    if not hf_model_name:
        print(f"❌ Невідома модель: {model_name}")
        return None
    
    print(f"🔗 Використовуємо HuggingFace модель: {hf_model_name}")
    return hf_model_name

def merge_lora_with_ollama_model(ollama_model_name, lora_path, output_path):
    """
    Об'єднує LoRA адаптер з моделлю з Ollama
    """
    print(f"🚀 Початок об'єднання {ollama_model_name} з LoRA...")
    
    # Отримуємо HuggingFace назву моделі
    hf_model_name = download_ollama_model_weights(ollama_model_name)
    if not hf_model_name:
        return False
    
    try:
        print(f"📥 Завантажуємо базову модель {hf_model_name}...")
        
        # Завантажуємо базову модель
        base_model = AutoModelForCausalLM.from_pretrained(
            hf_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Завантажуємо токенізатор
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"🔗 Завантажуємо LoRA адаптер з {lora_path}...")
        
        # Перевіряємо чи існує LoRA
        if not os.path.exists(os.path.join(lora_path, "adapter_config.json")):
            print(f"❌ LoRA адаптер не знайдено в {lora_path}")
            return False
        
        # Завантажуємо PEFT модель
        peft_model = PeftModel.from_pretrained(base_model, lora_path)
        
        print("🔀 Об'єднуємо LoRA з базовою моделлю...")
        # Об'єднуємо адаптер з базовою моделлю
        merged_model = peft_model.merge_and_unload()
        
        print(f"💾 Зберігаємо об'єднану модель в {output_path}...")
        os.makedirs(output_path, exist_ok=True)
        
        # Зберігаємо як один файл (не sharded)
        merged_model.save_pretrained(
            output_path,
            max_shard_size="10GB",  # Один файл
            safe_serialization=True
        )
        tokenizer.save_pretrained(output_path)
        
        print(f"✅ Модель успішно збережена в: {output_path}")
        
        # Показуємо що створилося
        files = os.listdir(output_path)
        print(f"📁 Створені файли: {files}")
        
        return True
        
    except Exception as e:
        print(f"❌ Помилка при об'єднанні: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_ollama_modelfile(model_path, model_name):
    """
    Створює Modelfile для Ollama
    """
    print(f"📝 Створюємо Modelfile для {model_name}...")
    
    # Використовуємо абсолютний шлях
    abs_model_path = os.path.abspath(model_path)
    
    modelfile_content = f"""FROM {abs_model_path}

TEMPLATE \"\"\"{{{{ if .System }}}}<start_of_turn>system
{{{{ .System }}}}<end_of_turn>
{{{{ end }}}}{{{{ if .Prompt }}}}<start_of_turn>user
{{{{ .Prompt }}}}<end_of_turn>
<start_of_turn>model
{{{{ end }}}}{{{{ .Response }}}}<end_of_turn>\"\"\"

PARAMETER stop "<start_of_turn>"
PARAMETER stop "<end_of_turn>"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096

SYSTEM \"\"\"Ти професійний HR асистент з великим досвідом в управлінні персоналом.

Твої завдання на сьогодні можуть включати:
- Перегляд та оцінка резюме кандидатів
- Підготовка до співбесід та їх проведення
- Аналіз продуктивності команди
- Робота з HR метриками та звітами
- Планування заходів з розвитку персоналу
- Вирішення конфліктних ситуацій
- Оновлення HR політик та процедур

Відповідай українською мовою, структуровано та професійно. Завжди пропонуй конкретні дії та практичні поради.\"\"\"
"""
    
    modelfile_path = f"Modelfile.{model_name}"
    with open(modelfile_path, 'w', encoding='utf-8') as f:
        f.write(modelfile_content)
    
    print(f"✅ Modelfile створено: {modelfile_path}")
    return modelfile_path

def create_ollama_model(modelfile_path, model_name):
    """
    Створює модель в Ollama
    """
    print(f"📤 Створюємо модель {model_name} в Ollama...")
    
    try:
        cmd = ["ollama", "create", model_name, "-f", modelfile_path]
        print(f"🔄 Виконуємо: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Модель {model_name} створена успішно!")
            print(f"🚀 Для запуску: ollama run {model_name}")
            return True
        else:
            print(f"❌ Помилка створення моделі:")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Помилка: {e}")
        return False

def main():
    """
    Головна функція
    """
    print("🤖 HR Assistant LoRA + Ollama Model Merger")
    print("=" * 50)
    
    # Налаштування
    ollama_model = "gemma2:2b"                    # Модель з Ollama
    lora_path = "./lora_adapter"                  # Ваш LoRA адаптер
    merged_model_path = "./merged_hr_model"       # Папка для об'єднаної моделі
    final_model_name = "hr_assistant_local"      # Назва в Ollama
    
    try:
        # 1. Об'єднуємо LoRA з базовою моделлю
        print(f"🔄 Крок 1: Об'єднання {ollama_model} з LoRA...")
        success = merge_lora_with_ollama_model(ollama_model, lora_path, merged_model_path)
        
        if not success:
            print("❌ Не вдалося об'єднати модель з LoRA")
            return
        
        # 2. Створюємо Modelfile
        print(f"🔄 Крок 2: Створення Modelfile...")
        modelfile_path = create_ollama_modelfile(merged_model_path, final_model_name)
        
        # 3. Створюємо модель в Ollama
        print(f"🔄 Крок 3: Імпорт в Ollama...")
        success = create_ollama_model(modelfile_path, final_model_name)
        
        if success:
            print(f"\n🎉 Успіх! Ваш HR Assistant готовий!")
            print(f"📋 Команди для використання:")
            print(f"   ollama run {final_model_name}")
            print(f"   >>> які в мене завдання на сьогодні")
            
            # Тестуємо модель
            print(f"\n🧪 Хочете протестувати модель зараз? (y/n)")
            if input().lower() == 'y':
                os.system(f"ollama run {final_model_name}")
        
    except Exception as e:
        print(f"❌ Критична помилка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()