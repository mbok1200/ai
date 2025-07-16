"""
Клієнт для роботи з Ollama API
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, AsyncGenerator
import time
import os

logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, base_url: str = None):
        # Use environment variable or default to localhost
        self.base_url = base_url or os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _get_session(self):
        """Отримання сесії"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session
    async def _make_request(self, url: str, data: dict):
        """Виконати HTTP запит"""
        session = await self._get_session()
        return await session.post(url, json=data)
    async def wait_for_ready(self, timeout: int = 120):
        """Очікування готовності Ollama сервера"""
        logger.info("Очікування готовності Ollama сервера...")
        
        for attempt in range(timeout):
            try:
                if await self.check_health():
                    logger.info("Ollama сервер готовий!")
                    return True
                    
            except Exception as e:
                logger.debug(f"Спроба {attempt + 1}: {e}")
                
            await asyncio.sleep(1)
        
        raise Exception("Ollama сервер не готовий після очікування")
    
    async def check_health(self) -> bool:
        """Перевірка здоров'я Ollama"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/tags") as response:
                return response.status == 200
        except Exception:
            return False
    
    async def list_models(self) -> List[Dict]:
        """Список доступних моделей"""
        session = await self._get_session()
        async with session.get(f"{self.base_url}/api/tags") as response:
            if response.status == 200:
                data = await response.json()
                return data.get("models", [])
            else:
                raise Exception(f"Помилка отримання моделей: {response.status}")
    
    async def pull_model(self, model_name: str) -> bool:
        """Завантаження моделі"""
        logger.info(f"Завантаження моделі {model_name}...")
        
        session = await self._get_session()
        payload = {"name": model_name}
        
        async with session.post(
            f"{self.base_url}/api/pull",
            json=payload
        ) as response:
            if response.status == 200:
                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            if 'status' in data:
                                logger.info(f"Статус: {data['status']}")
                            if data.get('status') == 'success':
                                logger.info(f"Модель {model_name} завантажена успішно!")
                                return True
                        except json.JSONDecodeError:
                            continue
                return True
            else:
                raise Exception(f"Помилка завантаження моделі: {response.status}")
    async def close(self):
        """Закрити aiohttp сесію"""
        if self.session:
            await self.session.close()
            self.session = None
    async def create_model(self, name: str, modelfile_path: str) -> bool:
        """Створення кастомної моделі"""
        try:
            with open(modelfile_path, 'r', encoding='utf-8') as f:
                modelfile_content = f.read()
            
            logger.info(f"Створення моделі {name} з Modelfile:")
            logger.info(f"Вміст: {modelfile_content}")
            
            url = f"{self.base_url}/api/create"
            payload = {
                "name": name,
                "modelfile": modelfile_content
            }
            
            session = await self._get_session()
            async with session.post(url, json=payload) as response:
                logger.info(f"Response status: {response.status}")
                
                if response.status == 200:
                    # Читаємо відповідь
                    response_text = await response.text()
                    logger.info(f"Creation response: {response_text}")
                    
                    # Ollama може повертати кілька JSON об'єктів
                    lines = response_text.strip().split('\n')
                    for line in lines:
                        if line.strip():
                            try:
                                status_obj = json.loads(line)
                                if status_obj.get('error'):
                                    logger.error(f"Помилка створення моделі: {status_obj['error']}")
                                    return False
                                if 'status' in status_obj:
                                    logger.info(f"Статус: {status_obj['status']}")
                            except json.JSONDecodeError:
                                continue
                    
                    return True
                else:
                    logger.error(f"Помилка HTTP: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Помилка створення моделі: {e}")
            return False
    
    async def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Dict:
        """Генерація відповіді"""
        session = await self._get_session()
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        async with session.post(
            f"{self.base_url}/api/generate",
            json=payload
        ) as response:
            if response.status == 200:
                data = await response.json()
                return data
            else:
                raise Exception(f"Помилка генерації: {response.status}")
    
    async def generate_stream(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[Dict, None]:
        """Стрімінг генерації"""
        session = await self._get_session()
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature
            }
        }
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        async with session.post(
            f"{self.base_url}/api/generate",
            json=payload
        ) as response:
            if response.status == 200:
                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            yield data
                        except json.JSONDecodeError:
                            continue
            else:
                raise Exception(f"Помилка стрімінг генерації: {response.status}")
    
    async def generate_with_lora(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Dict:
        """Генерація з LoRA адаптером"""
        
        # If LoRA is loaded, enhance the prompt with HR context
        if hasattr(self, 'lora_manager') and self.lora_manager and self.lora_manager.is_loaded():
            # Add HR context to prompt
            enhanced_prompt = f"""You are an experienced HR assistant. You help with employee questions, policies, recruitment, and workplace issues. Provide professional, helpful, and empathetic responses.

User question: {prompt}

Please provide a comprehensive HR-focused response:"""
            
            logger.info("Using LoRA-enhanced prompt")
            prompt = enhanced_prompt
        
        return await self.generate(model, prompt, temperature, max_tokens)
