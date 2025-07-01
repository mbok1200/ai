import os

class Config:
    PERSIST_DIR = "db"
    CHUNK_SIZE = 1300
    CHUNK_OVERLAP = 150
    EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
    NUM_CHUNKS = 6
    BEDROCK_MODEL = "anthropic.claude-3-sonnet-20240229-v1:0"
    AWS_REGION = "us-east-1"
    TEMPERATURE = 0.3
    MAX_HISTORY = 3
    DATA_DIR = "/structured"
    # --- Multilanguage support ---
    LANGUAGES = {
        "uk": {
            "title": "🤖 AI Асистент & Чат з Документами",
            "description": """
                Цей AI асистент може:
                - Відповідати на питання щодо ваших завантажених документів
                - Запам'ятовувати деталі розмови
                - Допомагати з загальними питаннями
                - Утримувати контекст у чаті
                """,
            "sidebar_header": "📄 Завантаження документів",
            "index_btn": "Індексувати локальну папку data/drive",
            "indexing": "Індексація файлів у /data/gdrive...",
            "index_done": "Індексація завершена. Оброблено файлів: {count}",
            "clear_chat": "Очистити історію чату",
            "check_docs": "Перевірити кількість документів у ChromaDB",
            "docs_count": "Кількість документів у ChromaDB: {count}",
            "sources": "Джерела",
            "ask": "Поставте питання про документи або поспілкуйтесь зі мною",
            "no_info": "🤔 Не знайдено релевантної інформації у документах. Спробуйте переформулювати питання.",
            "from": "**✅З файлу**: {filename}",
            "thinking": "⏳ Думаю...",
            "google": "\n\n🌐 **Google Search:**\n{answer}",
            "error": "Помилка: {error}"
        },
        "en": {
            "title": "🤖 AI Assistant & Document Chat",
            "description": """
                This AI assistant can:
                - Answer questions about your uploaded documents
                - Remember details from your conversation
                - Help with general questions
                - Maintain context across the chat
                """,
            "sidebar_header": "📄 Document Upload",
            "index_btn": "Index local folder data/drive",
            "indexing": "Indexing files in /data/gdrive...",
            "index_done": "Indexing complete. Files processed: {count}",
            "clear_chat": "Clear Chat History",
            "check_docs": "Check number of documents in ChromaDB",
            "docs_count": "Number of documents in ChromaDB: {count}",
            "sources": "Sources",
            "ask": "Ask a question about your documents or chat with me",
            "no_info": "🤔 No relevant information found in documents. Try rephrasing your question.",
            "from": "**✅From**: {filename}",
            "thinking": "⏳ Thinking...",
            "google": "\n\n🌐 **Google Search:**\n{answer}",
            "error": "Error: {error}"
        }
    }

    def __init__(self):
        os.makedirs(self.PERSIST_DIR, exist_ok=True)
        os.chmod(self.PERSIST_DIR, 0o770)
