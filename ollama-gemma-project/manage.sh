#!/bin/bash

# Скрипт для керування Ollama Gemma LoRA проектом

set -e

PROJECT_DIR="/home/mikola/projects/ai/ollama-gemma-project"
LORA_SOURCE="/home/mikola/projects/ai/hr-assistant-finetune-1/lora_adapter"

# Кольори для виводу
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    echo "Використання: $0 [КОМАНДА]"
    echo ""
    echo "Команди:"
    echo "  setup     - Копіювання LoRA адаптера та підготовка проекту"
    echo "  build     - Збірка Docker образу"
    echo "  start     - Запуск проекту (Docker Compose)"
    echo "  stop      - Зупинка проекту"
    echo "  restart   - Перезапуск проекту"
    echo "  logs      - Перегляд логів"
    echo "  status    - Статус сервісів"
    echo "  test      - Тестування API"
    echo "  clean     - Очищення (видалення контейнерів та образів)"
    echo "  help      - Показати це повідомлення"
}

setup_project() {
    print_status "Підготовка проекту..."
    
    cd "$PROJECT_DIR"
    
    # Копіюємо LoRA адаптер
    if [ -d "$LORA_SOURCE" ]; then
        print_status "Копіювання LoRA адаптера..."
        cp -r "$LORA_SOURCE" ./lora_adapter/
        print_success "LoRA адаптер скопійований"
    else
        print_error "LoRA адаптер не знайдений: $LORA_SOURCE"
        exit 1
    fi
    
    # Перевіряємо Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker не встановлений"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose не встановлений"
        exit 1
    fi
    
    print_success "Проект підготовлений!"
}

build_project() {
    print_status "Збірка Docker образу..."
    cd "$PROJECT_DIR"
    
    docker-compose build --no-cache
    
    print_success "Docker образ зібраний!"
}

start_project() {
    print_status "Запуск проекту..."
    cd "$PROJECT_DIR"
    
    docker-compose up -d
    
    print_success "Проект запущений!"
    print_status "API доступний на: http://localhost:8000"
    print_status "Ollama API: http://localhost:11434"
    print_status "Web UI: http://localhost:3000"
    
    # Чекаємо готовність сервісів
    print_status "Очікування готовності сервісів..."
    sleep 10
    
    # Перевіряємо API
    if curl -s http://localhost:8000/health > /dev/null; then
        print_success "API готовий!"
    else
        print_warning "API ще не готовий, може потрібно трохи почекати..."
    fi
}

stop_project() {
    print_status "Зупинка проекту..."
    cd "$PROJECT_DIR"
    
    docker-compose down
    
    print_success "Проект зупинений!"
}

restart_project() {
    print_status "Перезапуск проекту..."
    stop_project
    sleep 2
    start_project
}

show_logs() {
    print_status "Логи проекту:"
    cd "$PROJECT_DIR"
    
    docker-compose logs -f
}

show_status() {
    print_status "Статус сервісів:"
    cd "$PROJECT_DIR"
    
    docker-compose ps
    
    echo ""
    print_status "Перевірка доступності API:"
    
    # Перевіряємо API
    if curl -s http://localhost:8000/health > /dev/null; then
        print_success "✓ API доступний (http://localhost:8000)"
    else
        print_error "✗ API недоступний"
    fi
    
    # Перевіряємо Ollama
    if curl -s http://localhost:11434/api/tags > /dev/null; then
        print_success "✓ Ollama доступний (http://localhost:11434)"
    else
        print_error "✗ Ollama недоступний"
    fi
    
    # Перевіряємо Web UI
    if curl -s http://localhost:3000 > /dev/null; then
        print_success "✓ Web UI доступний (http://localhost:3000)"
    else
        print_warning "○ Web UI недоступний (можливо, не запущений)"
    fi
}

test_api() {
    print_status "Тестування API..."
    
    # Перевіряємо здоров'я
    print_status "Тест /health:"
    curl -s http://localhost:8000/health | python3 -m json.tool
    
    echo ""
    print_status "Тест /models:"
    curl -s http://localhost:8000/models | python3 -m json.tool
    
    echo ""
    print_status "Тест чату:"
    curl -s -X POST http://localhost:8000/chat \
        -H "Content-Type: application/json" \
        -d '{"message": "Привіт! Як справи?", "temperature": 0.7}' | \
        python3 -m json.tool
}

clean_project() {
    print_warning "Очищення проекту (видалення контейнерів та образів)..."
    read -p "Ви впевнені? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd "$PROJECT_DIR"
        
        # Зупиняємо та видаляємо контейнери
        docker-compose down -v
        
        # Видаляємо образи
        docker rmi $(docker images "ollama-gemma-project*" -q) 2>/dev/null || true
        
        # Очищаємо невикористані образи та томи
        docker system prune -f
        
        print_success "Очищення завершено!"
    else
        print_status "Очищення скасовано"
    fi
}

# Головна логіка
case "${1:-help}" in
    setup)
        setup_project
        ;;
    build)
        build_project
        ;;
    start)
        start_project
        ;;
    stop)
        stop_project
        ;;
    restart)
        restart_project
        ;;
    logs)
        show_logs
        ;;
    status)
        show_status
        ;;
    test)
        test_api
        ;;
    clean)
        clean_project
        ;;
    help|*)
        show_help
        ;;
esac
