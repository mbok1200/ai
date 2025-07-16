# QUICK START GUIDE
# Швидкий посібник по запуску Ollama Gemma LoRA проекту

echo "🚀 OLLAMA GEMMA LORA PROJECT - ШВИДКИЙ СТАРТ"
echo "=============================================="

# Перевірка системних вимог
echo "📋 Перевірка системних вимог..."

# Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker не встановлений. Встановіть Docker спочатку."
    echo "   Ubuntu/Debian: sudo apt install docker.io docker-compose"
    echo "   Або скачайте з: https://docs.docker.com/get-docker/"
    exit 1
fi

# Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose не встановлений."
    echo "   Ubuntu/Debian: sudo apt install docker-compose"
    exit 1
fi

# Перевірка пам'яті
MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
if [ "$MEMORY_GB" -lt 8 ]; then
    echo "⚠️  Увага: Рекомендується мінімум 8GB RAM (знайдено ${MEMORY_GB}GB)"
fi

echo "✅ Системні вимоги виконані"

# Перевірка LoRA адаптера
LORA_PATH="/home/mikola/projects/ai/hr-assistant-finetune-1/lora_adapter"
if [ ! -d "$LORA_PATH" ]; then
    echo "❌ LoRA адаптер не знайдено: $LORA_PATH"
    echo "   Переконайтеся, що шлях правильний"
    exit 1
fi

echo "✅ LoRA адаптер знайдено"

# Перехід до директорії проекту
PROJECT_DIR="/home/mikola/projects/ai/ollama-gemma-project"
cd "$PROJECT_DIR"

echo ""
echo "🔧 АВТОМАТИЧНИЙ ЗАПУСК"
echo "======================"

# Крок 1: Підготовка
echo "1️⃣  Підготовка проекту..."
./manage.sh setup

# Крок 2: Збірка
echo "2️⃣  Збірка Docker образу..."
./manage.sh build

# Крок 3: Запуск
echo "3️⃣  Запуск проекту..."
./manage.sh start

echo ""
echo "⏳ Очікування готовності сервісів (може зайняти кілька хвилин)..."
sleep 30

# Крок 4: Перевірка
echo "4️⃣  Перевірка статусу..."
./manage.sh status

echo ""
echo "🧪 ТЕСТУВАННЯ"
echo "=============="
python3 test_api.py

echo ""
echo "🎉 ПРОЕКТ ГОТОВИЙ!"
echo "=================="
echo ""
echo "📍 Доступні сервіси:"
echo "   🔗 FastAPI:     http://localhost:8000"
echo "   🔗 Ollama API:  http://localhost:11434" 
echo "   🔗 Web UI:      http://localhost:3000"
echo ""
echo "💡 Корисні команди:"
echo "   ./manage.sh status    - Статус сервісів"
echo "   ./manage.sh logs      - Перегляд логів"
echo "   ./manage.sh test      - Тестування API"
echo "   ./manage.sh stop      - Зупинка проекту"
echo ""
echo "🖥️  Приклади використання:"
echo "   python3 examples/chat_client.py     - Консольний чат"
echo "   firefox examples/web_client.html    - Веб-інтерфейс"
echo ""
echo "📚 Документація: README.md"
