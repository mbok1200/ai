#!/usr/bin/env python3
"""
Простий чат-клієнт для Ollama Gemma LoRA API
"""

import requests
import json
import sys
from typing import Generator

class ChatClient:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.session = requests.Session()
    
    def check_connection(self) -> bool:
        """Перевірка підключення"""
        try:
            response = self.session.get(f"{self.api_url}/health")
            print(f"Response from {self.api_url}/health: {response}")
            return response.ollama_ready == True
        except:
            return False
    
    def send_message(self, message: str, temperature: float = 0.7) -> str:
        """Відправка повідомлення та отримання відповіді"""
        payload = {
            "message": message,
            "temperature": temperature,
            "max_tokens": 1000
        }
        
        response = self.session.post(f"{self.api_url}/chat", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('response', 'Помилка отримання відповіді')
        else:
            return f"Помилка API: {response.status_code}"
    
    def send_message_stream(self, message: str, temperature: float = 0.7) -> Generator[str, None, None]:
        """Відправка повідомлення з стрімінг відповіддю"""
        payload = {
            "message": message,
            "temperature": temperature,
            "stream": True
        }
        
        response = self.session.post(
            f"{self.api_url}/chat",
            json=payload,
            stream=True
        )
        
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    try:
                        line_text = line.decode('utf-8')
                        if line_text.startswith('data: '):
                            data = json.loads(line_text[6:])
                            if 'response' in data:
                                yield data['response']
                    except json.JSONDecodeError:
                        continue
        else:
            yield f"Помилка API: {response.status_code}"

def main():
    print("🤖 Ollama Gemma HR Assistant")
    print("=" * 40)
    
    # Ініціалізація клієнта
    client = ChatClient()
    
    # Перевірка підключення
    if not client.check_connection():
        print("❌ Не вдалося підключитися до API")
        print("Переконайтеся, що сервер запущений:")
        print("./manage.sh start")
        sys.exit(1)
    
    print("✅ Підключення успішне!")
    print("💡 Введіть 'exit' для виходу")
    print("💡 Введіть 'stream' для перемикання стрімінг режиму")
    print("-" * 40)
    
    streaming_mode = False
    
    while True:
        try:
            # Введення користувача
            user_input = input("\n👤 Ви: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'вихід']:
                print("👋 До побачення!")
                break
            
            if user_input.lower() == 'stream':
                streaming_mode = not streaming_mode
                mode = "увімкнений" if streaming_mode else "вимкнений"
                print(f"🔄 Стрімінг режим {mode}")
                continue
            
            if not user_input:
                continue
            
            # Відправка повідомлення
            print("🤖 Асистент: ", end="", flush=True)
            
            if streaming_mode:
                # Стрімінг режим
                for chunk in client.send_message_stream(user_input):
                    print(chunk, end="", flush=True)
                print()  # Новий рядок після завершення
            else:
                # Звичайний режим
                response = client.send_message(user_input)
                print(response)
                
        except KeyboardInterrupt:
            print("\n👋 До побачення!")
            break
        except Exception as e:
            print(f"\n❌ Помилка: {e}")

if __name__ == "__main__":
    main()
