import json
import gradio as gr
import os
import logging
from typing import Dict
from dotenv import load_dotenv
from ai_system import AISystem

load_dotenv(".env")

# Налаштування логування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def load_file_map() -> Dict[str, str]:
    """Завантаження мапи файлів з конфігурації"""
    file_map = {}
    try:
        with open("data/gdrive_file_map.json", "r") as f:
            file_map = json.load(f)
    except FileNotFoundError:
        logger.warning("Файл file_map.json не знайдено, використовується порожня мапа")
    return file_map
file_map = load_file_map()


# Глобальний екземпляр системи
ai_system = AISystem()

def chat_interface(message: str, history: list, mode: str) -> tuple:
    """Інтерфейс чату для Gradio з режимами роботи"""
    try:
        result = ai_system.process_query(message, mode)
        
        # Форматуємо відповідь
        response = result['response']
        source_info = f"\n\n---\n📍 **Джерело:** {result['source']}"
        
        # Додаємо інформацію про режим
        mode_emoji = {
            'rag_only': '📚',
            'hybrid': '🔍', 
            'web_only': '🌐',
            'research': '🔬'
        }
        source_info += f"\n🎯 **Режим:** {mode_emoji.get(mode, '❓')} {mode}"
        
        # Додаємо метадані якщо є
        if result['metadata']:
            metadata = result['metadata']
            if 'confidence' in metadata:
                source_info += f"\n💯 **Впевненість:** {metadata['confidence']:.1%}"
            if 'functions' in metadata:
                source_info += f"\n🔧 **Функції:** {', '.join(metadata['functions'])}"
            if 'score' in metadata:
                source_info += f"\n📊 **Релевантність:** {metadata['score']:.1%}"
            if 'sources_analyzed' in metadata:
                source_info += f"\n🔍 **Джерел проаналізовано:** {metadata['sources_analyzed']}/{metadata.get('total_sources', 'N/A')}"
            if 'analysis_type' in metadata:
                source_info += f"\n🧠 **Тип аналізу:** {metadata['analysis_type']}"
            if 'sources_count' in metadata:
                source_info += f"\n📄 **Веб-джерел:** {metadata['sources_count']}"
            if 'sources' in metadata and isinstance(metadata['sources'], list):
                source_info += f"\n📚 **RAG джерел:** {len(metadata['sources'])}"
                source_info += "\n\n**Джерела:**"
                for src in metadata['sources']:
                    # Безпечне отримання ключа для file_map
                    source_key = src['source'].split('.')[0] if '.' in src['source'] else src['source']
                    
                    if source_key in file_map:
                        source_info += f"\n- <span style='color: #2563eb;'>[{src['title']}]({file_map[source_key]})</span>"
                    else:
                        source_info += f"\n- <span style='color: #dc2626;'>{src['title']}</span> (джерело: {src['source']})"
            if 'context_quality' in metadata:
                quality = metadata['context_quality']
                quality_emoji = "✅" if quality['is_good'] else "⚠️"
                source_info += f"\n{quality_emoji} **Якість контенту:** {quality['reason']}"
        
        full_response = response + source_info
        
        # Оновлюємо історію в новому форматі "messages"
        new_message = {"role": "user", "content": message}
        assistant_message = {"role": "assistant", "content": full_response}
        
        # Додаємо повідомлення до історії
        history.append(new_message)
        history.append(assistant_message)
        
        return "", history
        
    except Exception as e:
        error_response = f"❌ Вибачте, сталася помилка: {str(e)}"
        
        # Форматуємо помилку в новому форматі
        new_message = {"role": "user", "content": message}
        error_message = {"role": "assistant", "content": error_response}
        
        history.append(new_message)
        history.append(error_message)
        
        return "", history

def create_interface():
    """Створення Gradio інтерфейсу"""
    
    # Кастомний CSS
    css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: 0 auto;
        width: 100% !important;
        height: 100% !important;
    }
    .chat-message {
        padding: 10px;
        margin: 5px 0;
        border-radius: 10px;
    }
    """
    
    with gr.Blocks(title="🤖 AI Assistant", css=css, theme=gr.themes.Soft(), fill_height=True, fill_width=True) as app:
        gr.Markdown(
            """
            # 🤖 AI Assistant ### RAG + Function Calling + Google Search
            Система поєднує: 📚 **RAG**, 🔧 **Function Calling**, 🔍 **Google Search** 
            """
        )
        with gr.Row(scale=2):
            with gr.Column():
                chatbot = gr.Chatbot(
                    height=500,
                    placeholder="Напишіть запит і я допоможу знайти інформацію...",
                    avatar_images=(
                        "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",  # користувач
                        "https://cdn-icons-png.flaticon.com/512/4712/4712027.png"   # бот
                    ),
                    type="messages"
                )

                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Введіть ваш запит тут...",
                        container=False,
                        scale=4
                    )
                    send_btn = gr.Button("Відправити", variant="primary", scale=1)

                # Кнопка очищення
                clear_btn = gr.ClearButton([msg, chatbot], value="🗑️ Очистити")
        with gr.Sidebar(position="left"):
            with gr.Column(scale=1):
                gr.Markdown("### 💡 Приклади запитів:")
                
                examples = [
                    "🎯 завдання #12345",
                    "📅 завдання на сьогодні", 
                    "⏰ заповнити 4 години для #12345",
                    "📊 статус завдання Bug 123",
                    "👤 мої завдання",
                    "🔍 пошук у Redmine API",
                    "📚 wiki інформація про проект",
                    "❓ що таке штучний інтелект?"
                ]
                
                for example in examples:
                    gr.Button(
                        example, 
                        size="sm"
                    ).click(
                        lambda x=example: (x, []),
                        outputs=[msg, chatbot],
                        queue=False
                    )
                with gr.Group():
                    gr.Markdown("### 🚀 Швидкі режими:")
                    
                    mode_buttons = gr.Radio(
                        choices=[
                            ("📚 Тільки база знань", "rag_only"),
                            ("🔍 База + веб-пошук", "hybrid"),
                            ("🌐 Тільки веб-пошук", "web_only"),
                            # ("🔬 Дослідницький режим", "research")
                        ],
                        value="hybrid",
                        label="Режим роботи:",
                        info="Оберіть стратегію пошуку"
                    )
                gr.Markdown("### ℹ️ Статус системи:")
                
                # Перевірка статусу компонентів
                status_info = []
                
                # Pinecone
                pinecone_status = "✅" if os.getenv("PINECONE_API_KEY") else "❌"
                status_info.append(f"{pinecone_status} Pinecone RAG")
                
                # OpenAI
                openai_status = "✅" if os.getenv("OPENAI_API_KEY") else "❌"
                status_info.append(f"{openai_status} OpenAI")
                
                # Redmine
                redmine_status = "✅" if os.getenv("REDMINE_API_KEY") else "❌"
                status_info.append(f"{redmine_status} Redmine API")
                
                # Google
                google_status = "✅" if os.getenv("GOOGLE_API_KEY") else "❌"
                status_info.append(f"{google_status} Google Search")
                
                gr.Markdown("\n\n".join(status_info))
        
        # Прив'язуємо події з режимом
        msg.submit(chat_interface, [msg, chatbot, mode_buttons], [msg, chatbot], queue=True)
        send_btn.click(chat_interface, [msg, chatbot, mode_buttons], [msg, chatbot], queue=True)
    
    return app

if __name__ == "__main__":
    required_vars = [
        "OPENAI_API_KEY",
        "PINECONE_API_KEY", 
        "PINECONE_ENV",
        "PINECONE_INDEX_NAME"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    
    # Запускаємо інтерфейс
    app = create_interface()
    app.queue(max_size=20)
    app.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True
    )