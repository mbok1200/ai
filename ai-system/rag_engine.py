import os
import logging
from typing import List, Dict, Optional
from openai import OpenAI
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

class RAGEngine:
    """Система пошуку через Pinecone RAG"""
    
    def __init__(self, pinecone_index_name: str):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Ініціалізація Pinecone (новий API)
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
        self.index_name = pinecone_index_name
        self.index = self.pc.Index(pinecone_index_name)
        self._detect_embedding_model()
        self._log_index_stats()
    
    def _detect_embedding_model(self):
        """Автоматичне визначення моделі embedding"""
        try:
            stats = self.index.describe_index_stats()
            dimension = stats.dimension
            
            if dimension == 1536:
                self.embedding_model = "text-embedding-ada-002"
            elif dimension == 768:
                self.embedding_model = "local"  # Використовуємо локальну модель
            elif dimension == 3072:
                self.embedding_model = "text-embedding-3-large"
            else:
                self.embedding_model = "local"  # fallback
                
        except Exception as e:
            self.embedding_model = "local"
    
    def _log_index_stats(self):
        try:
            stats = self.index.describe_index_stats()
            if stats.namespaces:
                for ns, data in stats.namespaces.items():
                    ns_name = ns if ns else "''"
                    logging.info(f"   Namespace {ns_name}: {data['vector_count']} векторів")
        except Exception as e:
            logging.error(f"❌ Помилка отримання статистики: {e}")
    
    def search(self, query: str, top_k: int = 5) -> Dict:
        try:
            stats = self.index.describe_index_stats()
            total_vectors = stats.total_vector_count                   
            if total_vectors == 0:
                return {
                    'success': False,
                    'context': '',
                    'score': 0.0,
                    'sources': [],
                    'raw_results': [],
                    'message': 'База знань порожня'
                }
            
            # Генеруємо embedding для запиту
            embedding = self._get_embedding(query)
            
            # Шукаємо в default namespace (де більше векторів)
            results = self.index.query(
                vector=embedding,
                top_k=top_k,
                include_metadata=True,
                namespace="default"  # ← Ключове виправлення!
            )
                    
            # Якщо в default мало результатів, спробуємо порожній namespace
            if len(results.matches) < top_k // 2:
                empty_results = self.index.query(
                    vector=embedding,
                    top_k=top_k,
                    include_metadata=True,
                    namespace=""  # Порожній namespace
                )                
                # Об'єднуємо результати
                all_matches = list(results.matches) + list(empty_results.matches)
                # Сортуємо за score
                all_matches.sort(key=lambda x: x.score, reverse=True)
                results.matches = all_matches[:top_k]
            
            if not results.matches:
                return {
                    'success': False,
                    'context': '',
                    'score': 0.0,
                    'sources': [],
                    'raw_results': [],
                    'message': 'В базі знань нічого не знайдено'
                }
            
            # Збираємо всі результати для відображення
            all_results = []
            context_parts = []
            sources = []
            total_score = 0
            
            for i, match in enumerate(results.matches, 1):
                result_info = {
                    'rank': i,
                    'score': round(match.score, 3),
                    'id': match.id,
                    'text': match.metadata.get('text', ''),
                    'source': match.metadata.get('source', 'Unknown'),
                    'title': match.metadata.get('title', 'Без назви'),
                    'is_relevant': match.score > 0.7
                }
                all_results.append(result_info)
                
                # Додаємо до контексту тільки релевантні
                if match.score > 0.7:
                    context_parts.append(match.metadata.get('text', ''))
                    sources.append({
                        'id': match.id,
                        'score': match.score,
                        'source': match.metadata.get('source', 'Unknown'),
                        'title': match.metadata.get('title', 'Без назви')
                    })
                    total_score += match.score
            
            avg_score = total_score / len(sources) if sources else 0
            
            return {
                'success': len(context_parts) > 0,
                'context': '\n\n'.join(context_parts),
                'score': avg_score,
                'sources': sources,
                'raw_results': all_results,
                'total_found': len(results.matches),
                'relevant_count': len(context_parts),
                'message': f'Знайдено {len(results.matches)} документів, {len(context_parts)} релевантних'
            }
            
        except Exception as e:
            return {
                'success': False,
                'context': '',
                'score': 0.0,
                'sources': [],
                'raw_results': [],
                'error': str(e),
                'message': f'Помилка пошуку: {str(e)}'
            }
    
    def _get_embedding(self, text: str) -> List[float]:
        """Отримання embedding (той же метод що і в DocumentLoader)"""
        try:
            if self.embedding_model == "local":
                return self._get_local_embedding(text)
            else:
                response = self.openai_client.embeddings.create(
                    input=text,
                    model=self.embedding_model
                )
                return response.data[0].embedding
        except Exception as e:
            return self._get_local_embedding(text)
    
    def _get_local_embedding(self, text: str) -> List[float]:
        """Локальний embedding (точно той же що DocumentLoader)"""
        try:            
            # Ініціалізуємо модель якщо ще не ініціалізована
            if not hasattr(self, '_local_model'):
                # Використовуємо ТУ Ж САМУ модель що і DocumentLoader
                self._local_model = SentenceTransformer('intfloat/multilingual-e5-base')
            
            embedding = self._local_model.encode(text).tolist()
            
            # Перевіряємо розмірність
            stats = self.index.describe_index_stats()
            expected_dim = stats.dimension
            
            if len(embedding) != expected_dim:                
                if len(embedding) > expected_dim:
                    embedding = embedding[:expected_dim]
                elif len(embedding) < expected_dim:
                    # Розширюємо вектор
                    while len(embedding) < expected_dim:
                        remaining = expected_dim - len(embedding)
                        if remaining >= len(embedding):
                            embedding.extend(embedding)
                        else:
                            embedding.extend(embedding[:remaining])
                    embedding = embedding[:expected_dim]
            
            return embedding
            
        except ImportError:
            raise Exception("Встановіть: pip install sentence-transformers")
        except Exception as e:
            raise e
    
    def generate_answer(self, query: str, context: str) -> str:
        """Генерація відповіді на основі контексту"""
        try:
            system_prompt = f"""
Ти корисний асистент. Відповідай українською мовою на основі наданого контексту.

Контекст:
{context}

Якщо в контексті немає інформації для відповіді на запит, скажи що не знаєш.
Відповідай детально та структуровано.
"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return "Вибачте, сталася помилка під час генерації відповіді."
    
    def format_search_results(self, rag_result: Dict) -> str:
        """Форматування результатів пошуку для відображення"""
        if not rag_result.get('raw_results'):
            return "🔍 Результатів пошуку немає"
        
        output = []
        output.append(f"🔍 **Результати пошуку в базі знань:**")
        output.append(f"📊 {rag_result.get('message', '')}")
        output.append("")
        
        for result in rag_result['raw_results']:
            relevance_icon = "✅" if result['is_relevant'] else "⚠️"
            score_percent = f"{result['score']*100:.1f}%"
            
            output.append(f"{relevance_icon} **#{result['rank']} - {result['title']}** ({score_percent})")
            
            # Показуємо перші 150 символів тексту
            text_preview = result['text'][:150] + "..." if len(result['text']) > 150 else result['text']
            output.append(f"📄 {text_preview}")
            output.append(f"🏷️ Джерело: {result['source']}")
            output.append("")
        
        return "\n".join(output)