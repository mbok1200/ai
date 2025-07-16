#!/bin/bash

# Активація віртуального середовища
source /opt/venv/bin/activate

# Запуск Ollama в фоні
ollama serve &

# Очікування готовності Ollama
echo "Waiting for Ollama service..."
until curl -f http://ollama:11434/api/version; do
    echo "Ollama not ready, waiting..."
    sleep 2
done

echo "Ollama готова! Завантаження базової моделі..."

# Завантаження базової моделі
ollama pull gemma2:2b

echo "Створення кастомної моделі з LoRA..."

# Створення кастомної моделі
ollama create gemma2-hr-assistant -f /app/Modelfile

echo "Запуск FastAPI додатка..."
# Start the FastAPI application
cd /app/src
python app.py