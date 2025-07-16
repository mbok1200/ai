# Ollama Gemma LoRA Project

Проект для використання моделі Gemma-2-2b з LoRA адаптером через Ollama в Docker контейнері.

## 🚀 Особливості

- **Ollama**: Локальний LLM сервер для запуску моделей
- **Gemma-2-2b**: Компактна та ефективна мовна модель від Google
- **LoRA адаптер**: Ваш навчений адаптер для HR/менеджмент домену
- **Docker**: Повна контейнеризація для легкого розгортання
- **FastAPI**: RESTful API для взаємодії з моделлю
- **Web UI**: Веб-інтерфейс для зручної роботи
- **Стрімінг**: Підтримка потокових відповідей

## 📁 Структура проекту

```
ollama-gemma-project/
├── src/
│   ├── app.py              # Головний FastAPI додаток
│   ├── ollama_client.py    # Клієнт для роботи з Ollama
│   └── lora_manager.py     # Менеджер LoRA адаптерів
├── lora_adapter/           # Ваш LoRA адаптер (копіюється автоматично)
├── Dockerfile              # Docker образ
├── docker-compose.yml      # Оркестрація сервісів
├── Modelfile              # Конфігурація моделі для Ollama
├── requirements.txt        # Python залежності
├── manage.sh              # Скрипт керування проектом
└── README.md              # Цей файл
```

## 🛠️ Встановлення та запуск

### Передумови

- Docker та Docker Compose
- Мінімум 8GB RAM
- Навчений LoRA адаптер у `/home/mikola/projects/ai/hr-assistant-finetune-1/lora_adapter/`

### Швидкий старт

```bash
# 1. Підготовка проекту (копіювання LoRA адаптера)
./manage.sh setup

# 2. Збірка Docker образу
./manage.sh build

# 3. Запуск проекту
./manage.sh start
```

### Доступні команди

```bash
./manage.sh setup     # Підготовка проекту
./manage.sh build     # Збірка образу
./manage.sh start     # Запуск проекту
./manage.sh stop      # Зупинка проекту
./manage.sh restart   # Перезапуск
./manage.sh logs      # Перегляд логів
./manage.sh status    # Статус сервісів
./manage.sh test      # Тестування API
./manage.sh clean     # Очищення
./manage.sh help      # Довідка
```

## 🌐 Доступні сервіси

Після запуску доступні наступні сервіси:

- **FastAPI**: http://localhost:8000 - Головний API
- **Ollama API**: http://localhost:11434 - Прямий доступ до Ollama
- **Web UI**: http://localhost:3000 - Веб-інтерфейс для чату

## 📚 API Документація

### Основні ендпоінти

#### GET /
Інформація про API

#### GET /health
```json
{
  "status": "healthy",
  "ollama_ready": true,
  "lora_loaded": true
}
```

#### GET /models
Список доступних моделей

#### POST /chat
Чат з моделлю

**Запит:**
```json
{
  "message": "Привіт! Як справи?",
  "temperature": 0.7,
  "max_tokens": 1000,
  "stream": false
}
```

**Відповідь:**
```json
{
  "response": "Привіт! У мене все добре, дякую. Як я можу вам допомогти?",
  "model": "gemma2-hr-assistant",
  "tokens_used": 15
}
```

### Приклади використання

#### cURL
```bash
# Простий чат
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Розкажи про ефективне лідерство", "temperature": 0.7}'

# Стрімінг відповідь
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Що таке agile коучинг?", "stream": true}'
```

#### Python
```python
import requests

# Звичайний запит
response = requests.post("http://localhost:8000/chat", json={
    "message": "Поради з управління командою",
    "temperature": 0.7
})

result = response.json()
print(result["response"])

# Стрімінг
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={"message": "Розкажи про мотивацію співробітників", "stream": True},
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

#### JavaScript
```javascript
// Звичайний запит
fetch('http://localhost:8000/chat', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        message: 'Як провести ефективну зустріч?',
        temperature: 0.7
    })
})
.then(response => response.json())
.then(data => console.log(data.response));

// Стрімінг з fetch
fetch('http://localhost:8000/chat', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        message: 'Принципи lean менеджменту',
        stream: true
    })
})
.then(response => {
    const reader = response.body.getReader();
    function read() {
        reader.read().then(({done, value}) => {
            if (done) return;
            console.log(new TextDecoder().decode(value));
            read();
        });
    }
    read();
});
```

## ⚙️ Конфігурація

### Modelfile
Файл `Modelfile` містить конфігурацію моделі для Ollama:

```dockerfile
FROM gemma2:2b

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

SYSTEM """
Ви - корисний AI асистент, навчений на українських даних з HR та менеджменту.
"""
```

### Docker Compose
У `docker-compose.yml` можна налаштувати:
- Порти сервісів
- Змінні середовища
- Volumes для збереження даних

## 🔧 Налаштування LoRA

Ваш LoRA адаптер автоматично інтегрується з базовою моделлю. Конфігурація адаптера:

- **Базова модель**: google/gemma-2-2b
- **LoRA rank**: 8
- **LoRA alpha**: 16
- **Dropout**: 0.05
- **Цільові модулі**: q_proj, k_proj, gate_proj, o_proj, up_proj, v_proj, down_proj

## 📊 Моніторинг

### Перевірка статусу
```bash
./manage.sh status
```

### Перегляд логів
```bash
./manage.sh logs
```

### Тестування API
```bash
./manage.sh test
```

## 🚨 Усунення проблем

### Проблема: Ollama не запускається
```bash
# Перевірте логи
./manage.sh logs

# Перезапустіть проект
./manage.sh restart
```

### Проблема: Модель не завантажується
```bash
# Перевірте доступність Ollama
curl http://localhost:11434/api/tags

# Перезапустіть з очищенням
./manage.sh clean
./manage.sh setup
./manage.sh build
./manage.sh start
```

### Проблема: LoRA адаптер не знайдений
Переконайтеся, що адаптер знаходиться у:
```
/home/mikola/projects/ai/hr-assistant-finetune-1/lora_adapter/
```

## 📈 Оптимізація продуктивності

### Системні вимоги
- **Мінімум**: 8GB RAM, 4 CPU cores
- **Рекомендовано**: 16GB RAM, 8 CPU cores
- **GPU**: Опціонально, для прискорення

### Налаштування температури
- `0.1-0.3`: Консервативні відповіді
- `0.7`: Збалансовані відповіді (за замовчуванням)
- `0.9-1.0`: Креативні відповіді

## 🔐 Безпека

### Продакшн рекомендації
- Використовуйте HTTPS
- Додайте аутентифікацію
- Обмежте доступ до API
- Регулярно оновлюйте залежності

### Конфігурація для продакшн
```bash
# Змініть порти у docker-compose.yml
# Додайте nginx як reverse proxy
# Налаштуйте SSL сертифікати
```

## 📞 Підтримка

При виникненні проблем:

1. Перевірте логи: `./manage.sh logs`
2. Перевірте статус: `./manage.sh status`
3. Спробуйте перезапуск: `./manage.sh restart`
4. У крайньому випадку: `./manage.sh clean` та повторне встановлення

## 📝 Ліцензія

Цей проект використовується для навчальних та дослідницьких цілей.
