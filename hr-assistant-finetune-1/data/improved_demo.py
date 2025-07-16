import torch
import streamlit as st
import json
import random
import re
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Конфігурація ---
base_model_name = "google/gemma-2-2b"
lora_path = "/content/drive/MyDrive/lora_hr_assistant_v2"
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
    """Створює промпт у форматі навчальних даних для кращої генерації JSON"""
    system_prompt = """Ти HR-асистент для роботи з системою Redmine. Твоя мета:
• Розуміти українські запити користувачів
• Генерувати природні відповіді українською мовою
• Визначати потрібну функцію та її параметри
• Надавати корисну інформацію

Будь ввічливим, професійним та корисним."""

    return f"""{system_prompt}

Користувач: {input_text}

Асистент:"""

def generate_response(tokenizer, model, prompt: str, max_new_tokens: int = 300, temperature: float = 0.3):
    """Генерує відповідь моделі з оптимізованими параметрами для JSON"""
    if tokenizer is None or model is None:
        return "Помилка: модель не завантажена"
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        generated_tokens = output_ids[0][inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()
    except Exception as e:
        return f"Помилка генерації: {str(e)}"

def extract_function_info(response_text: str):
    """Покращена функція витягування JSON з відповіді"""
    function_calls = []
    
    # Шукаємо різні варіанти JSON блоків
    json_patterns = [
        r'\{[^{}]*"function_call"[^{}]*\}',
        r'\{[^{}]*"name"[^{}]*"arguments"[^{}]*\}',
        r'Function Call:\s*\{[^{}]*\}',
        r'"function_call":\s*\{[^{}]*\}'
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                # Очищуємо та нормалізуємо JSON
                clean_match = match.strip()
                if clean_match.startswith('Function Call:'):
                    clean_match = clean_match.replace('Function Call:', '').strip()
                
                func_data = json.loads(clean_match)
                if 'function_call' in func_data:
                    function_calls.append(func_data['function_call'])
                elif 'name' in func_data and 'arguments' in func_data:
                    function_calls.append(func_data)
            except json.JSONDecodeError:
                continue
    
    # Якщо JSON не знайдено, спробуємо витягнути інформацію про функції текстом
    if not function_calls:
        text_function = extract_function_from_text(response_text)
        if text_function:
            function_calls.append(text_function)
    
    return function_calls
def extract_function_from_text(response_text: str):
    """Витягує функцію з тексту, якщо JSON не згенеровано"""
    response_lower = response_text.lower()
    
    # Розширені патерни для визначення функцій
    function_patterns = {
        'get_issue_by_date': ['завдання', 'дату', 'сьогодні', 'завтра', 'тиждень', 'місяць', 'на дату', 'my tasks', 'get_my_tasks'],
        'get_issue_by_id': ['#', 'номер', 'завдання', 'id', 'вказаним id', 'за id', '456789', '445432', '453213', 'calling_issue', 'get_issue'],
        'get_issue_by_name': ['user story', 'bug', 'назвою', 'з назвою'],
        'get_issue_status': ['статус', 'стан'],
        'get_issue_hours': ['годин', 'витрачено', 'скільки', 'getting_hours', 'get_assigned_hours'],
        'fill_issue_hours': ['заповнити', 'години', 'record time', 'запіс часу', 'fill_hours_for_days', 'filling_hours', 'updating_hours'],
        'get_user_status': ['мій статус', 'статус акаунту'],
        'set_user_status': ['я на роботі', 'я вдома'],
        'create_issue': ['створи', 'створити', 'create ticket', 'новий наклад', 'create_issue', 'creating_issue'],
        'assign_issue': ['назначити', 'призначити', 'assign', 'прыпісьце', 'assign_me_to_issue', 'assigning_me'],
        'get_wiki_info': ['wiki', 'інформація'],
        'resolve_issue': ['вырішити', 'resolve issue', 'закрити']
    }
    for func_name, keywords in function_patterns.items():
        if any(keyword in response_lower for keyword in keywords):
            args = extract_arguments_from_response(response_text, func_name)
            return {
                "name": func_name,
                "arguments": args
            }

def extract_arguments_from_response(text: str, func_name: str):
    """Витягує аргументи з відповіді що містить 'Виклик функції:'"""
    args = {}
    
    # Шукаємо параметри після назви функції
    if 'params:' in text.lower():
        params_match = re.search(r"params:\s*['\"]([^'\"]+)['\"]", text)
        if params_match:
            args['value_1'] = params_match.group(1)
    
    # Альтернативний пошук параметрів
    param_patterns = [
        r"параметр:\s*['\"]([^'\"]+)['\"]",
        r"ID[:\s]+['\"]?(\d+)['\"]?",
        r"#(\d+)",
        r"(\d{5,6})"
    ]
    
    for pattern in param_patterns:
        match = re.search(pattern, text)
        if match:
            args['value_1'] = match.group(1)
            break
    
    return args

def extract_arguments_from_text(text: str, func_name: str):
    """Витягує аргументи з тексту на основі типу функції"""
    args = {}
    text_lower = text.lower()
    
    if func_name == 'get_issue_by_date':
        if 'сьогодні' in text_lower or 'today' in text_lower:
            args['value_1'] = 'today'
        elif 'завтра' in text_lower or 'tomorrow' in text_lower:
            args['value_1'] = 'tomorrow'
        elif 'тиждень' in text_lower or 'week' in text_lower:
            args['value_1'] = 'week'
        elif 'місяць' in text_lower or 'month' in text_lower:
            args['value_1'] = 'month'
        else:
            # Шукаємо дати у форматі 23.04 або 23.04.2025
            date_match = re.search(r'\d{1,2}\.\d{1,2}\.?\d{0,4}', text)
            if date_match:
                args['value_1'] = date_match.group()
            else:
                args['value_1'] = 'all'  # За замовчуванням всі завдання
    
    elif func_name in ['get_issue_by_id', 'get_issue_status', 'get_issue_hours']:
        # Розширений пошук номерів завдань
        id_patterns = [
            r"['\"](\d{5,6})['\"]",  # У лапках
            r"#(\d{5,6})",           # З решіткою
            r"ID[:\s]+(\d{5,6})",    # Після ID:
            r"(\d{5,6})"             # Просто число
        ]
        
        for pattern in id_patterns:
            id_match = re.search(pattern, text)
            if id_match:
                args['value_1'] = id_match.group(1)
                break
    
    elif func_name == 'get_issue_by_name':
        # Шукаємо User Story, Bug або інші назви
        name_patterns = [
            r"(User Story \d+)",
            r"(Bug \d+)",
            r"з назвою\s+['\"]([^'\"]+)['\"]",
            r"назвою\s+['\"]([^'\"]+)['\"]"
        ]
        
        for pattern in name_patterns:
            name_match = re.search(pattern, text)
            if name_match:
                args['value_1'] = name_match.group(1)
                break
    
    elif func_name == 'fill_issue_hours':
        # Шукаємо завдання та години
        id_match = re.search(r'#?(\d{5,6})', text)
        hours_match = re.search(r'(\d+)\s*годин', text)
        if id_match:
            args['value_1'] = f"#{id_match.group(1)}"
        if hours_match:
            args['value_2'] = hours_match.group(1)
            args['value_3'] = 'today'
    
    elif func_name == 'set_user_status':
        if 'на роботі' in text_lower or 'on_work' in text_lower:
            args['value_1'] = 'on_work'
        elif 'вдома' in text_lower or 'off_work' in text_lower:
            args['value_1'] = 'off_work'
    
    elif func_name == 'create_issue':
        # Шукаємо назву нового завдання
        title_match = re.search(r"назвою\s+['\"]([^'\"]+)['\"]", text)
        if title_match:
            args['value_1'] = title_match.group(1)
        else:
            args['value_1'] = 'Нове завдання'
    
    elif func_name == 'assign_issue':
        # Шукаємо завдання та користувача
        id_match = re.search(r'#?(\d{5,6})', text)
        user_match = re.search(r'назначити\s+(\w+)', text)
        if id_match:
            args['value_1'] = f"#{id_match.group(1)}"
        if user_match:
            args['value_2'] = user_match.group(1)
    
    return args

def get_real_examples_from_dataset():
    """Повертає реальні приклади з датасету"""
    return [
        "які в мене завдання",
        "які в мене завдання на сьогодні", 
        "шо там в мене по завдання на завтра",
        "які в мене завдання на тиждень",
        "які в мене завдання на місяць",
        "що по завданням на рік",
        "що в мене за завдання на 23.04",
        "які в мене завдання на 23.04.2025",
        "що за завдання #33456",
        "які в мене завдання #33456",
        "#33456 шо там по завданню",
        "Bug 453465",
        "що за таска Bug 453465",
        "User Story 453780",
        "що за User Story 453799",
        "статус завдання 453799",
        "User Story 453759 статус завдання",
        "#453799 який статус",
        "Скільки годин витрачено на завдання 453799",
        "#453799 скільки годин витрачено",
        "User Story 453759 скільки годин витрачено",
        "заповнити години",
        "заповнити години на завдання #453799 2 години",
        "заповнити години на завдання User Story 453759 3 години",
        "мій статус",
        "статус мого акаунту",
        "я на роботі",
        "я вдома",
        "створи завдання",
        "створи завдання з назвою 'Новий проект' і описом 'Опис нового проекту'",
        "#453799 назначити Yuri",
        "User Story 434256 назначити Yuri",
        "wiki інформація про проект"
    ]

def suggest_similar_examples(input_text: str):
    """Пропонує схожі приклади з датасету"""
    text_lower = input_text.lower()
    
    suggestions = []
    if 'завдання' in text_lower:
        suggestions.extend([
            "які в мене завдання на сьогодні",
            "що за завдання #33456",
            "User Story 453780"
        ])
    if 'статус' in text_lower:
        suggestions.extend([
            "статус завдання 453799",
            "мій статус"
        ])
    if 'годин' in text_lower:
        suggestions.extend([
            "заповнити години на завдання #453799 2 години",
            "Скільки годин витрачено на завдання 453799"
        ])
    
    return list(set(suggestions))

def handle_special_cases(input_text: str, response: str, function_calls: list):
    """Обробляє спеціальні випадки, коли модель не згенерувала JSON"""
    input_lower = input_text.lower()
    response_lower = response.lower()
    
    # Випадок "які в мене завдання"
    if 'які в мене завдання' in input_lower and not function_calls:
        # Якщо не вказана конкретна дата, це загальний запит всіх завдань
        if not any(word in input_lower for word in ['сьогодні', 'завтра', 'тиждень', 'місяць']):
            return [{
                "name": "get_issue_by_date",
                "arguments": {"value_1": "all"}
            }]
    
    # Випадок пошуку за ID з відповіді
    if 'виклик функції' in response_lower and 'get_issue_by_id' in response_lower:
        id_match = re.search(r"['\"](\d{5,6})['\"]", response)
        if id_match:
            return [{
                "name": "get_issue_by_id", 
                "arguments": {"value_1": id_match.group(1)}
            }]
    
    # Випадок, коли згадується "завдання" але функція не розпізнана
    if 'завдання' in input_lower and not function_calls:
        # Перевіряємо чи є в запиті номер завдання
        id_match = re.search(r'#?(\d{5,6})', input_text)
        if id_match:
            return [{
                "name": "get_issue_by_id",
                "arguments": {"value_1": id_match.group(1)}
            }]
        # Перевіряємо чи є User Story або Bug
        name_match = re.search(r'(User Story \d+|Bug \d+)', input_text, re.IGNORECASE)
        if name_match:
            return [{
                "name": "get_issue_by_name",
                "arguments": {"value_1": name_match.group(1)}
            }]
    
    # Випадок статусу
    if 'статус' in input_lower and not function_calls:
        if 'мій' in input_lower or 'акаунт' in input_lower:
            return [{
                "name": "get_user_status",
                "arguments": {}
            }]
        else:
            id_match = re.search(r'#?(\d{5,6})', input_text)
            if id_match:
                return [{
                    "name": "get_issue_status",
                    "arguments": {"value_1": id_match.group(1)}
                }]
    
    # Випадок установки статусу
    if ('я на роботі' in input_lower or 'я вдома' in input_lower) and not function_calls:
        status = 'on_work' if 'на роботі' in input_lower else 'off_work'
        return [{
            "name": "set_user_status",
            "arguments": {"value_1": status}
        }]
    
    # Випадок створення завдання
    if ('створи' in input_lower or 'створити' in input_lower) and 'завдання' in input_lower and not function_calls:
        title_match = re.search(r"назвою\s+['\"]([^'\"]+)['\"]", input_text)
        title = title_match.group(1) if title_match else 'Нове завдання'
        return [{
            "name": "create_issue",
            "arguments": {"value_1": title}
        }]
    
    # Випадок призначення
    if ('назначити' in input_lower or 'призначити' in input_lower) and not function_calls:
        id_match = re.search(r'#?(\d{5,6})', input_text)
        user_match = re.search(r'(назначити|призначити)\s+(\w+)', input_text, re.IGNORECASE)
        if id_match and user_match:
            return [{
                "name": "assign_issue",
                "arguments": {
                    "value_1": f"#{id_match.group(1)}",
                    "value_2": user_match.group(2)
                }
            }]
    
    # Випадок wiki
    if 'wiki' in input_lower and not function_calls:
        return [{
            "name": "get_wiki_info",
            "arguments": {"value_1": "проект"}
        }]
    
    # Випадок заповнення годин
    if ('заповнити' in input_lower and 'годин' in input_lower) and not function_calls:
        id_match = re.search(r'#?(\d{5,6})', input_text)
        hours_match = re.search(r'(\d+)\s*годин', input_text)
        if id_match:
            args = {"value_1": f"#{id_match.group(1)}"}
            if hours_match:
                args["value_2"] = hours_match.group(1)
                args["value_3"] = "today"
            return [{
                "name": "fill_issue_hours",
                "arguments": args
            }]
    
    # Випадок перегляду витрачених годин
    if ('скільки' in input_lower and 'годин' in input_lower) and not function_calls:
        id_match = re.search(r'#?(\d{5,6})', input_text)
        if id_match:
            return [{
                "name": "get_issue_hours",
                "arguments": {"value_1": id_match.group(1)}
            }]
    
    return function_calls

# --- Streamlit UI ---
st.set_page_config(
    page_title="HR Assistant - Покращена версія",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 HR Assistant - Навчена модель (Loss: 1.64 → 0.039)")
st.markdown("*Протестуйте модель з покращеною генерацією JSON*")

# Завантаження моделі
with st.spinner("⏳ Завантаження моделі..."):
    tokenizer, model = load_model()

if tokenizer is None or model is None:
    st.error("❌ Не вдалося завантажити модель")
    st.stop()

st.success("✅ Модель завантажена та готова до роботи!")

# Бічна панель
st.sidebar.header("📋 Приклади з датасету")
real_examples = get_real_examples_from_dataset()
selected_example = st.sidebar.selectbox(
    "Оберіть приклад:",
    [""] + real_examples
)

# Статистика
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Статистика навчання")
st.sidebar.metric("Train Loss", "0.039", "-97.6%")
st.sidebar.metric("Eval Loss", "0.231", "-59.2%")
st.sidebar.metric("Епох", "18", "126 кроків")

# Показуємо функції з датасету
st.sidebar.markdown("### 🔧 Підтримувані функції:")
functions_info = {
    "📅 get_issue_by_date": "today, tomorrow, week, month",
    "🔍 get_issue_by_id": "#33456, 453799",
    "📝 get_issue_by_name": "User Story, Bug",
    "📊 get_issue_status": "статус завдання",
    "⏱️ get_issue_hours": "витрачені години",
    "✏️ fill_issue_hours": "заповнити години",
    "👤 get_user_status": "мій статус",
    "🔄 set_user_status": "на роботі, вдома",
    "➕ create_issue": "створити завдання",
    "👥 assign_issue": "призначити користувачу",
    "📚 get_wiki_info": "wiki інформація"
}

for func, desc in functions_info.items():
    st.sidebar.write(f"**{func}**: {desc}")

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
        max_tokens = st.slider("Макс. токенів:", 50, 512, 250)
    with col1_2:
        temperature = st.slider("Температура:", 0.1, 1.0, 0.3, 0.1)
    
    # Опції генерації
    st.markdown("**🔧 Опції генерації:**")
    force_json = st.checkbox("Примусова генерація JSON", value=True)
    analyze_text = st.checkbox("Аналіз тексту при відсутності JSON", value=True)
    show_prompt = st.checkbox("Показати промпт", value=False)
    
    generate_button = st.button("🚀 Згенерувати відповідь", type="primary", use_container_width=True)

with col2:
    st.subheader("📤 Відповідь HR Assistant")
    
    if generate_button:
        if not input_text.strip():
            st.warning("⚠️ Введіть текст запиту")
        else:
            # Перевіряємо чи модель завантажена
            if tokenizer is None or model is None:
                st.error("❌ Модель не завантажена. Перезавантажте сторінку.")
            else:
                with st.spinner("🤖 Модель аналізує запит..."):
                    try:
                        # Модифікуємо промпт для кращої JSON генерації
                        base_prompt = build_prompt(input_text)
                        
                        if show_prompt:
                            with st.expander("🔍 Показати промпт"):
                                st.code(base_prompt)
                        
                        if force_json:
                            enhanced_prompt = base_prompt + "\n\nFunction Call:\n{"
                            response = generate_response(tokenizer, model, enhanced_prompt, max_tokens, temperature)
                            if not response.strip().startswith('{'):
                                response = "{\n" + response
                        else:
                            response = generate_response(tokenizer, model, base_prompt, max_tokens, temperature)
                        
                        # Перевіряємо чи є помилка в відповіді
                        if response.startswith("Помилка"):
                            st.error(f"❌ {response}")
                        else:
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
                                for func in function_calls:
                                    st.code(json.dumps(func, ensure_ascii=False, indent=2))
                            else:
                                st.warning("🔍 Функції не виявлено в відповіді")
                                
                                if analyze_text and any(keyword in response.lower() for keyword in ['завдання', 'статус', 'години', 'створи']):
                                    st.info("💡 Модель розпізнала контекст. Спробуйте:")
                                    st.markdown("""
                                    - Увімкніть 'Примусова генерація JSON'
                                    - Зменшіть температуру до 0.1-0.3
                                    - Або використайте точний приклад з датасету
                                    """)
                                    
                                    # Пропонуємо схожі приклади
                                    suggestions = []
                                    if suggestions:
                                        st.markdown("**🎯 Схожі приклади з датасету:**")
                                        for suggestion in suggestions[:3]:
                                            st.code(suggestion)
                    
                    except Exception as e:
                        st.error(f"❌ Помилка під час генерації: {str(e)}")
                        st.info("💡 Спробуйте перезавантажити сторінку або зменшити кількість токенів")

# Додаткова інформація про модель
st.markdown("---")
st.subheader("📊 Інформація про модель")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🎯 Навчання:**")
    st.write("• Датасет: 58 прикладів")
    st.write("• Train/Eval: 52/6")
    st.write("• Епох: 18 (126 кроків)")
    st.write("• Final Loss: 0.039")

with col2:
    st.markdown("**⚙️ Архітектура:**")
    st.write("• База: Gemma-2-2b")
    st.write("• LoRA адаптер")
    st.write("• Параметрів: 20.7M")
    st.write("• Dtype: float16")

with col3:
    st.markdown("**🔧 Функції:**")
    st.write("• 11 типів функцій")
    st.write("• JSON генерація")
    st.write("• Українська мова")
    st.write("• Redmine API")

# Приклад використання
with st.expander("📖 Приклад роботи з моделлю"):
    st.markdown("""
    **Запит користувача:** `які в мене завдання на сьогодні`
    
    **Відповідь моделі:**
    ```
    Перевіряю завдання за вказаною датою.
    Виклик функції: get_issue_by_date, параметр: 'today'
    ```
    
    **Виявлена функція:**
    ```json
    {
        "name": "get_issue_by_date",
        "arguments": {
            "value_1": "today"
        }
    }
    ```
    
    **Мокова відповідь API:**
    📅 Завдання на today: #453231, #453232, #453233 (3 активних завдання)
    """)

st.info("""
🎯 **Результат:** Модель успішно навчилась розуміти українські запити та генерувати структуровані виклики функцій для Redmine API.
Фінальний loss 0.039 показує відмінну конвергенцію!
""")