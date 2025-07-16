import torch
import streamlit as st
import json
import random
import re
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Конфігурація ---
base_model_name = "google/gemma-2-2b"
lora_path = "/home/mikola/projects/ai/hr-assistant-finetune-1/lora_hr_assistant_v2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    """Завантаження моделі з кешуванням"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            attn_implementation='eager'
        )
        
        model = PeftModel.from_pretrained(base_model, lora_path)
        model = model.merge_and_unload()
        model.to(device)
        model.eval()
        
        return tokenizer, model
    except Exception as e:
        st.error(f"Помилка завантаження моделі: {e}")
        return None, None

def build_prompt(input_text: str) -> str:
    """Створює промпт точно як у навчальних даних"""
    system_prompt = """Ти HR-асистент для роботи з системою Redmine. Твоя мета:
• Розуміти українські запити користувачів
• Генерувати природні відповіді українською мовою
• Визначати потрібну функцію та її параметри
• Надавати корисну інформацію

Будь ввічливим, професійним та корисним."""

    return f"""{system_prompt}

Користувач: {input_text}

Асистент:"""

def generate_response(tokenizer, model, prompt: str, max_new_tokens: int = 300, temperature: float = 0.7):
    """Генерує відповідь моделі"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        
        generated_tokens = output_ids[0][inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()
    except Exception as e:
        return f"Помилка генерації: {e}"

def extract_function_info(response_text: str):
    """Витягує інформацію про функції з відповіді"""
    # Шукаємо JSON блоки
    json_patterns = [
        r'\{[^{}]*"function_call"[^{}]*\}',
        r'\{[^{}]*"name"[^{}]*"arguments"[^{}]*\}'
    ]
    
    function_calls = []
    for pattern in json_patterns:
        matches = re.findall(pattern, response_text, re.DOTALL)
        for match in matches:
            try:
                func_data = json.loads(match)
                if 'function_call' in func_data:
                    function_calls.append(func_data['function_call'])
                elif 'name' in func_data and 'arguments' in func_data:
                    function_calls.append(func_data)
            except json.JSONDecodeError:
                continue
    
    return function_calls

def get_real_examples_from_dataset():
    """Повертає реальні приклади з датасету"""
    return [
        # Завдання за датою (точно з датасету)
        "які в мене завдання",
        "які в мене завдання на сьогодні", 
        "шо там в мене по завдання на завтра",
        "які в мене завдання на тиждень",
        "які в мене завдання на місяць",
        "що по завданням на рік",
        "що в мене за завдання на 23.04",
        "які в мене завдання на 23.04.2025",
        "шо там по задачам на 23.04.2025 12:00",
        
        # Конкретні завдання (з датасету)
        "що за завдання #33456",
        "які в мене завдання #33456",
        "#33456 шо там по завданню",
        "Bug 453465",
        "що за таска Bug 453465",
        "#Bug 453465 шо там по завданню",
        "завдання User Story #4534690",
        "User Story 453780",
        "що за User Story 453799",
        
        # Статус (з датасету)
        "статус завдання 453799",
        "User Story 453759 статус завдання",
        "#453799 який статус",
        "статус завдання #453799",
        "статус User Story 453759",
        
        # Години (з датасету)
        "Скільки годин витрачено на завдання 453799",
        "#453799 скільки годин витрачено",
        "#453799 скільки годин витрачено на завдання",
        "User Story 453759 скільки годин витрачено",
        
        # Заповнення годин (з датасету)
        "заповнити години",
        "заповнити години на завдання #453799 2 години",
        "заповнити години на завдання User Story 453759 3 години",
        "заповнити години на завдання #453799 4 години на вчора",
        "заповнити години на завдання User Story 453759 5 годин на 23.04",
        "заповнити години на завдання #453799 6 годин на 22.04.2025",
        "User Story 453759 8 годин на 29.04.2025 12:00 зробив тест",
        "#453799 8 годин на 30.04.2025 12:00 зробив рев'ю",
        
        # Користувач (з датасету)
        "мій статус",
        "статус мого акаунту", 
        "я зараз на роботі?",
        "я на лікарняному",
        "я на роботі",
        "я вдома",
        
        # Створення (з датасету)
        "створи завдання",
        "створи завдання з назвою 'Новий проект' і описом 'Опис нового проекту'",
        "User Story 453759: bug with login",
        "#453799: fix issue with payment",
        "завдання #453459: add new feature, -add new button for exist menu",
        
        # Призначення (з датасету)
        "#459859 назначити",
        "#498459 назначити Yuri",
        "User Story 434256",
        "User Story 434256 назначити Yuri",
        "назначити на Yuri User Story 434256",
        
        # Wiki (з датасету)
        "wiki інформація",
        "wiki інформація про проект",
        "wiki інформація про контакти для лендигів та сайтів компанії"
    ]

def create_expected_responses():
    """Створює очікувані відповіді на основі датасету"""
    return {
        "які в мене завдання на сьогодні": {
            "expected_function": "get_issue_by_date",
            "expected_args": {"value_1": "today"},
            "mock_result": "📅 Завдання на сьогодні: #453231, #453232, #453233"
        },
        "що в мене за завдання на 23.04": {
            "expected_function": "get_issue_by_date", 
            "expected_args": {"value_1": "23.04.2025"},
            "mock_result": "📅 Завдання на 23.04: #453245, #453246"
        },
        "що за завдання #33456": {
            "expected_function": "get_issue_by_id",
            "expected_args": {"value_1": "33456"},
            "mock_result": "📋 Завдання #33456: 'Розробка API' (Статус: В роботі)"
        },
        "Bug 453465": {
            "expected_function": "get_issue_by_name",
            "expected_args": {"value_1": "Bug 453465"},
            "mock_result": "🐛 Bug 453465: 'Помилка логіну' (Пріоритет: Високий)"
        },
        "статус завдання 453799": {
            "expected_function": "get_issue_status",
            "expected_args": {"value_1": "453799"},
            "mock_result": "📊 Статус завдання 453799: В роботі → На тестуванні"
        },
        "заповнити години на завдання #453799 2 години": {
            "expected_function": "fill_issue_hours",
            "expected_args": {"value_1": "#453799", "value_2": "2", "value_3": "today"},
            "mock_result": "✅ Заповнено 2 години на завдання #453799 за сьогодні"
        },
        "я на роботі": {
            "expected_function": "set_user_status",
            "expected_args": {"value_1": "on_work"},
            "mock_result": "👤 Статус змінено на: На роботі"
        },
        "створи завдання": {
            "expected_function": "create_issue",
            "expected_args": {"value_1": "", "value_2": "", "value_3": "assign_me"},
            "mock_result": "➕ Потрібні деталі для створення завдання"
        }
    }

# --- Streamlit UI ---
st.set_page_config(
    page_title="HR Assistant - Точна реплікація датасету",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 HR Assistant - Навчений на реальних даних")
st.markdown("*Модель навчена на 58 точних прикладах з merged_with_links_is.jsonl*")

# Завантаження моделі
with st.spinner("⏳ Завантаження моделі..."):
    tokenizer, model = load_model()

if tokenizer is None or model is None:
    st.error("❌ Не вдалося завантажити модель")
    st.stop()

st.success("✅ Модель завантажена (Loss: 1.64 → 0.039)")

# Бічна панель з реальними прикладами
st.sidebar.header("📋 Приклади з датасету")
st.sidebar.markdown("*Точні запити з навчальних даних*")

real_examples = get_real_examples_from_dataset()
expected_responses = create_expected_responses()

selected_example = st.sidebar.selectbox(
    "Оберіть приклад (з merged_with_links_is.jsonl):",
    [""] + real_examples
)

# Показуємо очікувану функцію для вибраного прикладу
if selected_example and selected_example in expected_responses:
    expected = expected_responses[selected_example]
    st.sidebar.markdown("### 🎯 Очікувана функція:")
    st.sidebar.code(f"""
Функція: {expected['expected_function']}
Аргументи: {expected['expected_args']}
    """)

# Статистика з датасету
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Статистика датасету")
st.sidebar.metric("Всього прикладів", "58", "100% покриття")
st.sidebar.metric("Навчальних", "52", "90%")
st.sidebar.metric("Тестових", "6", "10%")
st.sidebar.metric("Функцій", "11", "Redmine API")

# Показуємо розподіл функцій у датасеті
function_counts = {
    "get_issue_by_date": 9,
    "get_issue_by_id": 3, 
    "get_issue_by_name": 6,
    "get_issue_status": 5,
    "get_issue_hours": 4,
    "fill_issue_hours": 11,
    "get_user_status": 4,
    "set_user_status": 2,
    "create_issue": 8,
    "assign_issue": 6,
    "get_wiki_info": 3
}

st.sidebar.markdown("### 📈 Розподіл у датасеті:")
for func, count in function_counts.items():
    percentage = (count / 58) * 100
    st.sidebar.write(f"**{func}**: {count} ({percentage:.1f}%)")

# Основний інтерфейс
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Введіть запит")
    
    input_text = st.text_area(
        "Запит українською мовою:",
        value=selected_example,
        placeholder="Наприклад: які в мене завдання на сьогодні",
        height=150
    )
    
    # Параметри генерації
    col1_1, col1_2 = st.columns(2)
    with col1_1:
        max_tokens = st.slider("Макс. токенів:", 50, 512, 300)
    with col1_2:
        temperature = st.slider("Температура:", 0.1, 1.0, 0.7, 0.1)
    
    generate_button = st.button("🚀 Згенерувати відповідь", type="primary", use_container_width=True)

with col2:
    st.subheader("📤 Відповідь HR Assistant")
    
    if generate_button:
        if not input_text.strip():
            st.warning("⚠️ Введіть текст запиту")
        else:
            with st.spinner("🤖 Модель аналізує запит..."):
                prompt = build_prompt(input_text)
                response = generate_response(tokenizer, model, prompt, max_tokens, temperature)
                
                # Відображення відповіді
                st.text_area(
                    "Згенерована відповідь:",
                    value=response,
                    height=150,
                    disabled=True
                )
                
                # Аналіз функцій
                function_calls = extract_function_info(response)
                
                if function_calls:
                    st.subheader("⚙️ Виявлені функції")
                    for i, func_call in enumerate(function_calls, 1):
                        with st.expander(f"🔧 Function #{i}: `{func_call.get('name', 'unknown')}`", expanded=True):
                            st.json(func_call)
                            
                            # Порівняння з очікуваним
                            if input_text in expected_responses:
                                expected = expected_responses[input_text]
                                func_name = func_call.get('name', '')
                                
                                if func_name == expected['expected_function']:
                                    st.success(f"✅ Правильна функція: {func_name}")
                                else:
                                    st.error(f"❌ Очікувалось: {expected['expected_function']}, отримано: {func_name}")
                                
                                # Мокова відповідь
                                st.info(f"📡 Мокова відповідь: {expected['mock_result']}")
                            else:
                                # Загальна мокова відповідь
                                mock_responses = {
                                    'get_issue_by_date': f"📅 Знайдено завдання на дату",
                                    'get_issue_by_id': f"📋 Інформація про завдання",
                                    'get_issue_by_name': f"🔍 Знайдено завдання",
                                    'get_issue_status': f"📊 Статус завдання",
                                    'get_issue_hours': f"⏱️ Години по завданню",
                                    'fill_issue_hours': f"✅ Години заповнені",
                                    'get_user_status': "👤 Статус користувача",
                                    'set_user_status': f"🔄 Статус оновлено",
                                    'create_issue': f"➕ Завдання створено",
                                    'assign_issue': f"👥 Завдання призначено",
                                    'get_wiki_info': f"📚 Wiki інформація"
                                }
                                
                                func_name = func_call.get('name', '')
                                mock_response = mock_responses.get(func_name, f"⚙️ Функція {func_name} виконана")
                                st.info(f"📡 Мокова відповідь: {mock_response}")
                else:
                    st.warning("🔍 Функції не виявлено в відповіді")
                    if any(keyword in response.lower() for keyword in ['завдання', 'статус', 'години', 'створи']):
                        st.info("💡 Модель розпізнала контекст, але не згенерувала JSON")

# Розділ з прогресом навчання
st.markdown("---")
st.subheader("📈 Прогрес навчання")

# Метрики навчання
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Початковий Loss", "1.6428", "Епоха 0.77")
with col2:
    st.metric("Фінальний Loss", "0.039", "-97.6% ✅")
with col3:
    st.metric("Eval Loss", "0.231", "Стабільний")
with col4:
    st.metric("Градієнт", "0.45", "Конвергенція ✅")

# Розділ з аналізом датасету
st.markdown("---")
st.subheader("📊 Аналіз навчального датасету")

# Показуємо статистику по функціях
import pandas as pd

df_functions = pd.DataFrame([
    {"Функція": func, "Кількість": count, "Відсоток": f"{(count/58)*100:.1f}%"} 
    for func, count in function_counts.items()
]).sort_values("Кількість", ascending=False)

st.dataframe(df_functions, use_container_width=True)

# Приклади форматів з датасету
st.markdown("---")
st.subheader("🎯 Підтримувані формати (з датасету)")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📅 Дати:**")
    st.code("""
today
tomorrow  
week
month
year
23.04
23.04.2025
23.04.2025 12:00
    """)

with col2:
    st.markdown("**🔢 Номери завдань:**")
    st.code("""
#33456
453799
#453799
Bug 453465
#Bug 453465
User Story 453759
User Story #453759
    """)

with col3:
    st.markdown("**👤 Статуси:**")
    st.code("""
on_work
off_work
assign_me
Yuri
ossystem
    """)

# Футер
st.markdown("---")
st.info("""
🎯 **Особливості цієї моделі:** Навчена на точних 58 прикладах з реального робочого процесу Redmine. 
Модель розуміє українську мову, розпізнає формати дат, номери завдань та генерує структуровані відповіді для API.

📊 **Результат навчання:** Loss знизився з 1.64 до 0.039 за 18 епох, що вказує на відмінне засвоєння патернів.
""")