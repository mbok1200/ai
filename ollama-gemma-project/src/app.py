"""
FastAPI додаток для роботи з Ollama та LoRA адаптером
"""

import asyncio
import logging
import requests
import json
from typing import Optional, Dict, Any
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Налаштування логування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import custom modules with error handling
try:
    from lora_manager import LoRAManager
    from ollama_client import OllamaClient
    logger.info("Custom modules imported successfully")
    
    # Ініціалізація компонентів
    ollama_client = OllamaClient()
    lora_manager = LoRAManager()
    
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.warning("Creating dummy implementations...")
    
    # Create dummy classes for development/testing
    class DummyOllamaClient:
        async def wait_for_ready(self):
            logger.info("Dummy: Ollama ready")
            
        async def pull_model(self, model):
            logger.info(f"Dummy: Pulling model {model}")
            
        async def check_health(self):
            return True
            
        async def list_models(self):
            return ["gemma2:2b"]
            
        async def generate(self, **kwargs):
            return {"response": "This is a dummy response from the test API", "eval_count": 10}
            
        async def generate_stream(self, **kwargs):
            yield {"response": "Dummy streaming response"}
            
        async def create_model(self, name, modelfile):
            logger.info(f"Dummy: Creating model {name}")
    
    class DummyLoRAManager:
        async def load_adapter(self, path):
            logger.info(f"Dummy: Loading LoRA adapter from {path}")
            
        def is_loaded(self):
            return False
            
        async def reload_adapter(self, path):
            logger.info(f"Dummy: Reloading LoRA adapter from {path}")
    
    OllamaClient = DummyOllamaClient
    LoRAManager = DummyLoRAManager
    
    ollama_client = OllamaClient()
    lora_manager = LoRAManager()
    
except Exception as e:
    logger.error(f"Unexpected error during imports: {e}")
    raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Запуск застосунку...")
    
    try:
        # Чекаємо на готовність Ollama
        await ollama_client.wait_for_ready()
        
        # Завантажуємо базову модель
        await ollama_client.pull_model("gemma2:2b")
        
        # Створюємо кастомну модель з LoRA
        await setup_custom_model()
        
        logger.info("Застосунок готовий до роботи!")
        
    except Exception as e:
        logger.error(f"Помилка під час запуску: {e}")
        logger.warning("Продовжуємо запуск без повної ініціалізації")
    
    yield
    
    # Shutdown
    logger.info("Вимкнення застосунку...")
    if hasattr(ollama_client, 'close'):
        await ollama_client.close()

# Create FastAPI app with lifespan
app = FastAPI(
    title="Ollama Gemma LoRA API",
    description="API для роботи з Ollama, Gemma-2-2b та LoRA адаптером",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware FIRST - before any routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000
    stream: Optional[bool] = False

class ChatResponse(BaseModel):
    response: str
    model: str
    tokens_used: Optional[int] = None

async def setup_custom_model():
    """Налаштування кастомної моделі з LoRA"""
    try:
        # Спочатку спробуємо завантажити LoRA адаптер
        logger.info("Спроба завантаження LoRA адаптера...")
        lora_success = await lora_manager.load_adapter("/app/lora_adapter")
        logger.info(f"LoRA завантаження: {'успішно' if lora_success else 'неуспішно'}")
        
        # Перевіряємо чи є конвертована GGUF модель
        gguf_model_path = "/app/merged_model/model.gguf"
        logger.info(f"Перевірка GGUF моделі: {gguf_model_path}")
        
        if Path(gguf_model_path).exists():
            logger.info("GGUF модель знайдена, використовуємо її")
            modelfile_content = f"""FROM {gguf_model_path}

SYSTEM "Ти - професійний HR асистент з досвідом роботи в українських компаніях. Допомагай з HR питаннями українською мовою."

PARAMETER temperature 0.7
PARAMETER top_p 0.9
"""
        else:
            logger.info("GGUF модель не знайдена, використовуємо базову з HR промптом")
            modelfile_content = """FROM gemma2:2b

SYSTEM "Ти - професійний HR асистент. Твоя задача - допомагати з HR питаннями, співбесідами, оцінкою кандидатів та HR процесами. Відповідай конкретно та професійно українською мовою."

PARAMETER temperature 0.7
PARAMETER top_p 0.9
"""
        
        modelfile_path = "/app/Modelfile"
        logger.info(f"Створення Modelfile: {modelfile_path}")
        
        with open(modelfile_path, 'w', encoding='utf-8') as f:
            f.write(modelfile_content)
        
        # Перевіряємо, що файл створився
        if not Path(modelfile_path).exists():
            logger.error("Modelfile не створився!")
            return
            
        # Показуємо вміст файлу
        with open(modelfile_path, 'r', encoding='utf-8') as f:
            actual_content = f.read()
        logger.info(f"Фактичний вміст Modelfile:\n---\n{actual_content}\n---")
        
        # Перевіряємо чи існує базова модель
        models_before = await ollama_client.list_models()
        logger.info(f"Моделі перед створенням: {models_before}")
        
        # Використовуємо той же URL що і OllamaClient
        logger.info(f"OllamaClient base_url: {ollama_client.base_url}")
        
        # Спробуємо створити модель через OllamaClient
        logger.info("Створення моделі через OllamaClient...")
        result = await ollama_client.create_model("gemma2-hr-assistant", modelfile_path)
        logger.info(f"Результат створення моделі: {result}")
        
        if result:
            # Перевіряємо чи модель з'явилася
            await asyncio.sleep(3)  # Чекаємо трохи довше
            models_after = await ollama_client.list_models()
            logger.info(f"Моделі після створення: {models_after}")
            
            # Перевіряємо список моделей
            if isinstance(models_after, list):
                available_models = [model.get("name", "") if isinstance(model, dict) else str(model) for model in models_after]
            elif isinstance(models_after, dict) and "models" in models_after:
                available_models = [model.get("name", "") for model in models_after["models"]]
            else:
                available_models = []
                
            logger.info(f"Доступні моделі після створення: {available_models}")
            
            if "gemma2-hr-assistant" in available_models:
                logger.info("✅ Кастомна модель успішно створена!")
            else:
                logger.warning("❌ Кастомна модель не з'явилась в списку")
        else:
            logger.error("Помилка створення моделі через OllamaClient")
        
    except Exception as e:
        logger.error(f"Помилка в setup_custom_model: {e}", exc_info=True)
        logger.info("Продовжуємо з базовою моделлю")
async def get_best_available_model():
    """Вибір найкращої доступної моделі"""
    try:
        models = await ollama_client.list_models()
        logger.info(f"Тип відповіді list_models: {type(models)}")
        logger.info(f"Відповідь list_models: {models}")
        
        # Виправляємо обробку списку моделей
        if isinstance(models, list):
            available_models = [model.get("name", "") if isinstance(model, dict) else str(model) for model in models]
        elif isinstance(models, dict) and "models" in models:
            available_models = [model.get("name", "") for model in models["models"]]
        else:
            available_models = []
            
        logger.info(f"Доступні моделі: {available_models}")
        
        # Prefer custom model, fallback to base model
        if "gemma2-hr-assistant" in available_models:
            logger.info("Використовуємо кастомну модель: gemma2-hr-assistant")
            return "gemma2-hr-assistant"
        elif "gemma2:2b" in available_models:
            logger.info("Використовуємо базову модель: gemma2:2b")
            return "gemma2:2b"
        else:
            # Use first available model
            selected = available_models[0] if available_models else "gemma2:2b"
            logger.info(f"Використовуємо першу доступну модель: {selected}")
            return selected
    except Exception as e:
        logger.error(f"Помилка в get_best_available_model: {e}")
        return "gemma2:2b"  # Safe fallback
@app.get("/")
async def root():
    """Головна сторінка"""
    return {
        "message": "Ollama Gemma LoRA API",
        "status": "running",
        "available_endpoints": [
            "/chat",
            "/models",
            "/health"
        ]
    }

@app.get("/health")
async def health_check():
    """Перевірка здоров'я сервісу"""
    try:
        status = await ollama_client.check_health()
        return {
            "status": "healthy" if status else "unhealthy",
            "ollama_ready": status,
            "lora_loaded": lora_manager.is_loaded()
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/models")
async def list_models():
    """Список доступних моделей"""
    try:
        models = await ollama_client.list_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Чат з моделлю"""
    logger.info(f"Отримано запит: {request.message}")
    
    try:
        # Get best available model
        model_name = await get_best_available_model()
        logger.info(f"Using model: {model_name}")
        
        if request.stream:
            return StreamingResponse(
                stream_chat_response(request, model_name),
                media_type="text/plain"
            )
        else:
            response = await ollama_client.generate_with_lora(
                model=model_name,
                prompt=request.message,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            return ChatResponse(
                response=response["response"],
                model=model_name,
                tokens_used=response.get("eval_count", 0)
            )
            
    except Exception as e:
        logger.error(f"Помилка в чаті: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

async def stream_chat_response(request: ChatRequest, model_name: str):
    """Стрімінг відповіді"""
    try:
        async for chunk in ollama_client.generate_stream(
            model=model_name,
            prompt=request.message,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        ):
            yield f"data: {json.dumps(chunk)}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.post("/reload-lora")
async def reload_lora(background_tasks: BackgroundTasks):
    """Перезавантаження LoRA адаптера"""
    try:
        background_tasks.add_task(lora_manager.reload_adapter, "/app/lora_adapter")
        return {"message": "LoRA адаптер перезавантажується..."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug")
async def debug_info():
    """Debug information"""
    return {
        "ollama_client_type": str(type(ollama_client)),
        "lora_manager_type": str(type(lora_manager)),
        "is_dummy": "Dummy" in str(type(ollama_client))
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
