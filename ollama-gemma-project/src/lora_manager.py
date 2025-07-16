"""
Менеджер для роботи з LoRA адаптерами
"""

import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

logger = logging.getLogger(__name__)


class LoRAManager:
    def __init__(self):
        self.adapter_path: Optional[Path] = None
        self.config: Optional[Dict] = None
        self.is_adapter_loaded = False

    def is_loaded(self) -> bool:
        """Перевірка чи завантажений адаптер"""
        return self.is_adapter_loaded

    async def load_adapter(self, adapter_path: str) -> bool:
        """Завантаження LoRA адаптера (спрощена версія для Ollama)"""
        try:
            logger.info(f"Перевірка LoRA адаптера в {adapter_path}")
            
            self.adapter_path = Path(adapter_path)
            
            # Просто перевіряємо наявність файлів
            if self._validate_adapter_files():
                self.config = self._load_adapter_config()
                self.is_adapter_loaded = True
                logger.info("LoRA конфігурація завантажена")
                return True
            else:
                logger.warning("LoRA файли не знайдені, використовуємо базову модель")
                return True  # Не блокуємо запуск
                
        except Exception as e:
            logger.error(f"Помилка завантаження LoRA адаптера: {e}")
            return True

    def _validate_adapter_files(self) -> bool:
        """Перевірка наявності необхідних файлів"""
        required_files = ["adapter_config.json", "adapter_model.safetensors"]

        for file_name in required_files:
            file_path = self.adapter_path / file_name
            if not file_path.exists():
                logger.error(f"Відсутній файл: {file_path}")
                return False

        return True

    def _load_adapter_config(self) -> Dict:
        """Завантаження конфігурації адаптера"""
        config_path = self.adapter_path / "adapter_config.json"

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        return config

    async def reload_adapter(self, adapter_path: str) -> bool:
        """Перезавантаження адаптера"""
        logger.info("Перезавантаження LoRA адаптера...")

        # Скидаємо поточний стан
        self.is_adapter_loaded = False
        self.config = None
        self.adapter_path = "/app/lora_adapter/checkpoint-202"

        # Завантажуємо знову
        return await self.load_adapter(adapter_path)

    def get_adapter_info(self) -> Dict[str, Any]:
        """Інформація про завантажений адаптер"""
        if not self.is_loaded():
            return {"status": "not_loaded"}

        return {
            "status": "loaded",
            "adapter_path": str(self.adapter_path),
            "base_model": self.config.get("base_model_name_or_path"),
            "peft_type": self.config.get("peft_type"),
            "rank": self.config.get("r"),
            "alpha": self.config.get("lora_alpha"),
            "dropout": self.config.get("lora_dropout"),
            "target_modules": self.config.get("target_modules"),
            "task_type": self.config.get("task_type"),
        }

    def merge_adapter_with_base(self, output_path: str) -> bool:
        """Злиття адаптера з базовою моделлю (для експорту)"""
        try:
            if not self.is_loaded():
                raise Exception("Адаптер не завантажений")

            logger.info("Початок злиття адаптера з базовою моделлю...")

            # Завантажуємо базову модель
            base_model_name = self.config["base_model_name_or_path"]
            logger.info(f"Завантаження базової моделі: {base_model_name}")

            model = AutoModelForCausalLM.from_pretrained(
                base_model_name, torch_dtype=torch.float16, device_map="auto"
            )

            # Завантажуємо токенізатор
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)

            # Застосовуємо LoRA адаптер
            logger.info("Застосування LoRA адаптера...")
            model = PeftModel.from_pretrained(model, str(self.adapter_path))

            # Зливаємо адаптер з базовою моделлю
            logger.info("Злиття адаптера...")
            merged_model = model.merge_and_unload()

            # Зберігаємо об'єднану модель
            logger.info(f"Збереження об'єднаної моделі в {output_path}")
            merged_model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)

            logger.info("Злиття завершено успішно!")
            return True

        except Exception as e:
            logger.error(f"Помилка при злитті: {e}")
            return False
