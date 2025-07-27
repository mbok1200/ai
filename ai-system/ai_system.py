import os
from urllib import response
from dotenv import load_dotenv
from rag_engine import RAGEngine
from function_agent import FunctionAgent
from tools.google_search import GoogleSearchTool
from workflow import DialogueState, Workflow
from openai import OpenAI
from typing import Dict, Any, List

import workflow
load_dotenv(".env")

class AISystem:
    def __init__(self):
        self.rag_engine = RAGEngine(
            pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "streamlit")
        )
        self.function_agent = FunctionAgent()
        self.google_search = GoogleSearchTool()
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.workflow = Workflow(self.openai_client)
        self.state = self.workflow.state
    def process_query(self, user_query: str, mode: str = "hybrid") -> Dict[str, Any]:
        """Основна логіка обробки запиту з режимами роботи"""
        if not user_query.strip():
            return {
                'response': "❓ Будь ласка, введіть запит",
                'source': 'System',
                'metadata': {}
            }
        
        # Логіка залежно від режиму
        if mode == "rag_only":
            return self._process_rag_only(user_query)
        elif mode == "web_only":
            return self._search_and_analyze_web(user_query)
        elif mode == "research":
            return self._research_mode(user_query)
        else:  # hybrid (default)
            return self._process_hybrid(user_query)
    
    def _process_rag_only(self, user_query: str) -> Dict[str, Any]:
        try:
            response = self.workflow.process_user_input("які в мене завдання на сьогодні")
            print(response)
            return {}
        except Exception as e:
            print(f"Function calling помилка: {e}")

        # 2. RAG пошук
        try:
            rag_result = self.rag_engine.search(user_query)
            
            if rag_result['success']:
                answer = self.rag_engine.generate_answer(user_query, rag_result['context'])
                
                return {
                    'response': answer,
                    'source': 'RAG (База знань)',
                    'metadata': {
                        'score': rag_result['score'],
                        'sources': rag_result['sources'],
                        'mode': 'rag_only'
                    }
                }
            else:
                return {
                    'response': "❌ Не знайдено релевантної інформації в базі знань. Спробуйте інший режим пошуку.",
                    'source': 'RAG (Не знайдено)',
                    'metadata': {'mode': 'rag_only', 'no_results': True}
                }
        except Exception as e:
            return {
                'response': f"❌ Помилка бази знань: {str(e)}",
                'source': 'RAG Error',
                'metadata': {'mode': 'rag_only', 'error': True}
            }
    
    def _process_hybrid(self, user_query: str) -> Dict[str, Any]:
        
        # 1. Спробуємо function calling
        try:
            return {}
        except Exception as func_error:
            print(f"Function calling помилка: {func_error}")
        try:
            rag_result = self.rag_engine.search(user_query)
            
            if rag_result['success'] and rag_result['score'] > 0.75:
                answer = self.rag_engine.generate_answer(user_query, rag_result['context'])
                
                return {
                    'response': answer,
                    'source': 'RAG (Pinecone)',
                    'metadata': {
                        'score': rag_result['score'],
                        'sources': rag_result['sources'],
                        'mode': 'hybrid'
                    }
                }
        except Exception as e:
            print(f"RAG пошук помилка: {e}")
        
        result = self._search_and_analyze_web(user_query)
        result['metadata']['mode'] = 'hybrid'
        return result
    
    def _research_mode(self, user_query: str) -> Dict[str, Any]:
        
        try:
            # Розширений Google пошук
            search_result = self.google_search.search_with_analysis(user_query, num_results=5)
            
            if not search_result['success']:
                return {
                    'response': f"❌ Помилка Google пошуку: {search_result.get('error', 'Невідома помилка')}",
                    'source': 'Google Search Error',
                    'metadata': {'error': True, 'mode': 'research'}
                }
            
            # Збираємо контент з усіх джерел
            all_content = []
            valid_sources = []
            
            for source in search_result['sources']:
                if source['success'] and source['content'] and len(source['content']) > 50:
                    all_content.append(f"Джерело: {source['title']}\n{source['content']}")
                    valid_sources.append(source)
            
            if not valid_sources:
                return {
                    'response': "❌ Не вдалося отримати достатньо інформації для дослідження",
                    'source': 'Research Mode (No Data)',
                    'metadata': {'mode': 'research', 'no_data': True}
                }
            
            # Поглиблений аналіз
            research_analysis = self._generate_research_analysis(
                user_query, 
                all_content, 
                valid_sources
            )
            
            return {
                'response': research_analysis,
                'source': 'Research Mode',
                'metadata': {
                    'sources_analyzed': len(valid_sources),
                    'total_sources': len(search_result['sources']),
                    'analysis_type': 'research',
                    'mode': 'research'
                }
            }
            
        except Exception as e:
            return {
                'response': f"❌ Помилка дослідницького режиму: {str(e)}",
                'source': 'Research Mode Error',
                'metadata': {'error': str(e), 'mode': 'research'}
            }

    def _generate_research_analysis(self, query: str, contents: List[str], sources: List[Dict]) -> str:
        """Генерація дослідницького аналізу"""
        try:
            combined_content = "\n\n---\n\n".join(contents)
            
            # Обмежуємо розмір
            max_tokens = 15000
            if len(combined_content) > max_tokens:
                combined_content = combined_content[:max_tokens] + "\n\n[Контент обрізано...]"
            
            system_prompt = """Ти - експертний дослідник та аналітик. Твоє завдання провести глибокий аналіз теми на основі кількох джерел.

Створи дослідницький звіт що містить:
1. 📋 Резюме теми
2. 🔍 Детальний аналіз ключових аспектів
3. 📊 Статистичні дані та факти (якщо є)
4. 🎯 Висновки та рекомендації
5. 🔮 Прогнози та тренди (якщо доречно)
6. ❓ Питання для подальшого дослідження

Використовуй структуровану подачу з заголовками, списками та виділенням важливих моментів."""

            user_prompt = f"""Тема дослідження: "{query}"

Проаналізуй наступну інформацію та створи детальний дослідницький звіт:

{combined_content}
"""

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo-16k",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2000,
                temperature=0.3
            )
            
            research_analysis = response.choices[0].message.content
            
            # Форматуємо звіт
            formatted_response = f"""# 🔬 Дослідницький звіт

{research_analysis}

---

## 📚 Джерела дослідження:

"""
            
            for i, source in enumerate(sources, 1):
                formatted_response += f"""**{i}. {source['title']}**
🔗 {source['url']}
📊 Проаналізовано слів: {source['word_count']}
⭐ Релевантність: {"🟢 Висока" if source['word_count'] > 500 else "🟡 Середня"}

"""
            
            formatted_response += f"\n🎯 *Дослідження базується на {len(sources)} джерелах*"
            
            return formatted_response
            
        except Exception as e:
            return f"❌ Помилка генерації дослідження: {str(e)}"

    def _search_and_analyze_web(self, user_query: str) -> Dict[str, Any]:
        
        try:
            # Google пошук
            search_result = self.google_search.search_with_analysis(user_query, num_results=3)
            
            if not search_result['success']:
                return {
                    'response': f"❌ Помилка Google пошуку: {search_result.get('error', 'Невідома помилка')}",
                    'source': 'Google Search Error',
                    'metadata': {'error': True}
                }
            
            # Збираємо контент з джерел
            valid_sources = []
            all_content = []
            
            for source in search_result['sources']:
                if source['success'] and source['content'] and len(source['content']) > 50:
                    all_content.append(source['content'][:2000])
                    valid_sources.append(source)
            
            if not valid_sources:
                return {
                    'response': "❌ Не вдалося отримати інформацію з веб-джерел",
                    'source': 'Web Search (No Data)',
                    'metadata': {'no_data': True}
                }
            
            # Генеруємо відповідь на основі веб-контенту
            combined_content = "\n\n---\n\n".join(all_content)
            
            # Обмежуємо розмір контенту
            max_tokens = 6000
            if len(combined_content) > max_tokens:
                combined_content = combined_content[:max_tokens] + "\n\n[Контент обрізано...]"
            
            # Генеруємо відповідь через OpenAI
            system_prompt = """Ти - експертний аналітик інформації. На основі наданого контенту з веб-джерел дай вичерпну та точну відповідь на запит користувача.

Використовуй структуровану подачу інформації з заголовками та списками. Посилайся на конкретні факти з джерел."""

            user_prompt = f"""Запит користувача: "{user_query}"

Контент з веб-джерел:
{combined_content}

Дай детальну відповідь на основі цієї інформації."""

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo-16k",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            web_analysis = response.choices[0].message.content
            
            # Форматуємо відповідь з джерелами
            formatted_response = f"""{web_analysis}

## 📚 Джерела інформації:

"""
            
            for i, source in enumerate(valid_sources, 1):
                formatted_response += f"""**{i}. {source['title']}**
🔗 {source['url']}
📊 Слів: {source.get('word_count', 'N/A')}

"""
            
            return {
                'response': formatted_response,
                'source': 'Web Search Analysis',
                'metadata': {
                    'sources_count': len(valid_sources),
                    'total_searched': len(search_result['sources']),
                    'search_query': user_query
                }
            }
            
        except Exception as e:
            return {
                'response': f"❌ Помилка веб-пошуку: {str(e)}",
                'source': 'Web Search Error',
                'metadata': {'error': str(e)}
            }
    def process_user_message(self, user_input: str, user_id: str = "default") -> str:
        """Головний метод обробки повідомлень"""
        
        # Створюємо початковий стан
        initial_state = DialogueState(
            user_input=user_input,
            user_id=user_id
        )
        
        # Виконуємо workflow
        final_state = self.app.ainvoke(initial_state)
        
        return final_state.response_message