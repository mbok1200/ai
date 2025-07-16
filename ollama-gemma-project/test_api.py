#!/usr/bin/env python3
"""
Скрипт для тестування Ollama Gemma LoRA API
"""

import requests
import json
import time
import sys
from typing import Dict, Any

class APITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def print_status(self, message: str):
        print(f"🔍 {message}")
    
    def print_success(self, message: str):
        print(f"✅ {message}")
    
    def print_error(self, message: str):
        print(f"❌ {message}")
    
    def print_warning(self, message: str):
        print(f"⚠️  {message}")
    
    def test_connection(self) -> bool:
        """Тест підключення до API"""
        self.print_status("Тестування підключення...")
        
        try:
            response = self.session.get(f"{self.base_url}/")
            if response.status_code == 200:
                self.print_success("Підключення успішне!")
                return True
            else:
                self.print_error(f"Помилка підключення: {response.status_code}")
                return False
        except Exception as e:
            self.print_error(f"Не вдалося підключитися: {e}")
            return False
    
    def test_health(self) -> bool:
        """Тест здоров'я сервісу"""
        self.print_status("Перевірка здоров'я сервісу...")
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"📊 Статус: {data.get('status')}")
                print(f"🤖 Ollama готовий: {data.get('ollama_ready')}")
                print(f"🔧 LoRA завантажений: {data.get('lora_loaded')}")
                
                if data.get('status') == 'healthy':
                    self.print_success("Сервіс здоровий!")
                    return True
                else:
                    self.print_warning("Сервіс не повністю готовий")
                    return False
            else:
                self.print_error(f"Помилка health check: {response.status_code}")
                return False
        except Exception as e:
            self.print_error(f"Помилка при перевірці здоров'я: {e}")
            return False
    
    def test_models(self) -> bool:
        """Тест списку моделей"""
        self.print_status("Отримання списку моделей...")
        
        try:
            response = self.session.get(f"{self.base_url}/models")
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])
                
                print(f"📋 Знайдено моделей: {len(models)}")
                for i, model in enumerate(models, 1):
                    name = model.get('name', 'Невідома')
                    size = model.get('size', 0)
                    print(f"   {i}. {name} ({size // (1024**3):.1f}GB)")
                
                self.print_success("Список моделей отримано!")
                return True
            else:
                self.print_error(f"Помилка отримання моделей: {response.status_code}")
                return False
        except Exception as e:
            self.print_error(f"Помилка при отриманні моделей: {e}")
            return False
    
    def test_simple_chat(self) -> bool:
        """Тест простого чату"""
        self.print_status("Тестування простого чату...")
        
        test_message = "Привіт! Як справи?"
        
        try:
            payload = {
                "message": test_message,
                "temperature": 0.7,
                "max_tokens": 100
            }
            
            start_time = time.time()
            response = self.session.post(f"{self.base_url}/chat", json=payload)
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                response_text = data.get('response', '')
                model_name = data.get('model', '')
                tokens_used = data.get('tokens_used', 0)
                
                print(f"💬 Запит: {test_message}")
                print(f"🤖 Відповідь: {response_text}")
                print(f"📊 Модель: {model_name}")
                print(f"🔢 Токенів: {tokens_used}")
                print(f"⏱️  Час відповіді: {end_time - start_time:.2f}с")
                
                self.print_success("Простий чат працює!")
                return True
            else:
                self.print_error(f"Помилка чату: {response.status_code}")
                print(f"Відповідь: {response.text}")
                return False
        except Exception as e:
            self.print_error(f"Помилка при тестуванні чату: {e}")
            return False
    
    def test_streaming_chat(self) -> bool:
        """Тест стрімінг чату"""
        self.print_status("Тестування стрімінг чату...")
        
        test_message = "Розкажи коротко про ефективне лідерство"
        
        try:
            payload = {
                "message": test_message,
                "temperature": 0.7,
                "stream": True
            }
            
            response = self.session.post(
                f"{self.base_url}/chat",
                json=payload,
                stream=True
            )
            
            if response.status_code == 200:
                print(f"💬 Запит: {test_message}")
                print("🔄 Стрімінг відповідь:")
                
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            line_text = line.decode('utf-8')
                            if line_text.startswith('data: '):
                                data = json.loads(line_text[6:])
                                if 'response' in data:
                                    chunk = data['response']
                                    print(chunk, end='', flush=True)
                                    full_response += chunk
                        except json.JSONDecodeError:
                            continue
                
                print("\n")
                self.print_success("Стрімінг чат працює!")
                return True
            else:
                self.print_error(f"Помилка стрімінг чату: {response.status_code}")
                return False
        except Exception as e:
            self.print_error(f"Помилка при тестуванні стрімінг чату: {e}")
            return False
    
    def test_hr_domain_questions(self) -> bool:
        """Тест питань з HR домену"""
        self.print_status("Тестування питань з HR домену...")
        
        hr_questions = [
            "Що таке agile коучинг?",
            "Як мотивувати команду?",
            "Принципи ефективного лідерства",
            "Як провести ефективну зустріч?",
            "Що таке lean менеджмент?"
        ]
        
        successful_tests = 0
        
        for i, question in enumerate(hr_questions, 1):
            print(f"\n📝 Тест {i}/{len(hr_questions)}: {question}")
            
            try:
                payload = {
                    "message": question,
                    "temperature": 0.8,
                    "max_tokens": 200
                }
                
                response = self.session.post(f"{self.base_url}/chat", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get('response', '')
                    
                    print(f"🤖 Відповідь: {answer[:200]}{'...' if len(answer) > 200 else ''}")
                    print(f"✅ Тест {i} пройдений")
                    successful_tests += 1
                else:
                    print(f"❌ Тест {i} провалений: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Тест {i} провалений: {e}")
        
        success_rate = (successful_tests / len(hr_questions)) * 100
        print(f"\n📊 Результат HR тестів: {successful_tests}/{len(hr_questions)} ({success_rate:.1f}%)")
        
        if success_rate >= 80:
            self.print_success("HR домен тести пройдені успішно!")
            return True
        else:
            self.print_warning("Деякі HR тести провалилися")
            return False
    
    def run_all_tests(self) -> bool:
        """Запуск всіх тестів"""
        print("🚀 Початок тестування Ollama Gemma LoRA API")
        print("=" * 50)
        
        tests = [
            ("Підключення", self.test_connection),
            ("Здоров'я сервісу", self.test_health),
            ("Список моделей", self.test_models),
            ("Простий чат", self.test_simple_chat),
            ("Стрімінг чат", self.test_streaming_chat),
            ("HR домен", self.test_hr_domain_questions)
        ]
        
        passed_tests = 0
        
        for test_name, test_func in tests:
            print(f"\n{'='*20} {test_name} {'='*20}")
            
            try:
                if test_func():
                    passed_tests += 1
                    time.sleep(1)  # Пауза між тестами
                else:
                    print(f"⚠️  Тест '{test_name}' провалений")
            except Exception as e:
                print(f"❌ Критична помилка в тесті '{test_name}': {e}")
        
        print(f"\n{'='*50}")
        print(f"📊 РЕЗУЛЬТАТИ ТЕСТУВАННЯ")
        print(f"{'='*50}")
        print(f"✅ Пройдено: {passed_tests}/{len(tests)}")
        print(f"📈 Успішність: {(passed_tests/len(tests)*100):.1f}%")
        
        if passed_tests == len(tests):
            print("🎉 Всі тести пройдені успішно!")
            return True
        elif passed_tests >= len(tests) * 0.8:
            print("⚠️  Більшість тестів пройдена, але є проблеми")
            return False
        else:
            print("❌ Критичні проблеми - багато тестів провалено")
            return False

def main():
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = "http://localhost:8000"
    
    print(f"🎯 Тестування API: {base_url}")
    
    tester = APITester(base_url)
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
