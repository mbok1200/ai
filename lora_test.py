import regex,torch, json
import streamlit as st
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

base_model_name = "google/gemma-2-2b"
lora_path = "lora_only"  # Змініть на свій шлях
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEMPERATURE=0.3
TOP_K=5
TOP_P=0.3
REPETITION_PENALTY=1.2
MAX_NEW_TOKENS=256
@st.cache_resource
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
            attn_implementation='eager'  # Рекомендації для Gemma-2
        )
        model = PeftModel.from_pretrained(base_model, lora_path)
        model = model.merge_and_unload()
        model.to(device)
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Помилка завантаження моделі: {e}")
        return None, None

# Формат промпта, щоб модель виводила валідний JSON з полями function_call і text
PROMPT_TEMPLATE = """
Інструкція: Ти HR-асистент для Redmine.
Твоя задача - відповідати на запити користувачів, використовуючи функції API Redmine.

- Якщо запит потребує виконання функції, поверни JSON з полями:
  - "function_calls" — список об'єктів, де кожен має:
    - "name" — назва функції,
    - "arguments" — словник аргументів функції.
  - "text" — текстова відповідь користувачу.

- Якщо функція не потрібна, поверни JSON з порожнім списком "function_calls" і відповідним "text".

**ВАЖЛИВО:** Відповідь має бути строго валідним JSON і містити валідні параметри "function_calls", "name", "arguments", "text".

Приклади:

Запит: "Покажи статус завдання #343544"
Відповідь:
{{
  "function_calls": [
    {{
      "name": "get_issue_status",
      "arguments": {{"issue_id": "343544"}}
    }}
  ],
  "text": "Статус завдання #343544: "
}}

Запит: "Привіт, як справи?"
Відповідь:
{{
  "function_calls": [],
  "text": "Привіт! Чим можу допомогти?"
}}

---

Запит:
{input_text}

Відповідь у форматі JSON:
"""

def build_prompt(input_text: str) -> str:
    return PROMPT_TEMPLATE.format(input_text=input_text)

def generate_response(
    tokenizer, 
    model, 
    prompt: str, 
    max_new_tokens=MAX_NEW_TOKENS, 
    temperature=TEMPERATURE, 
    top_k=TOP_K, 
    top_p=TOP_P, 
    repetition_penalty=REPETITION_PENALTY
    ):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    generated_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response_text.strip()
def parse_response(response_text):
    """
    Парсить JSON-відповідь у вигляді рядка.
    Повертає словник з ключами 'function_calls' та 'text' або None при помилці.
    """
    try:
        # Видаляємо зайві коми перед закриваючими дужками (об’єктів і масивів)
        cleaned_text = response_text
        # Приблизний спосіб видалити зайві коми (можна покращити за потреби)
        import re
        cleaned_text = re.sub(r',(\s*[}\]])', r'\1', cleaned_text)

        data = json.loads(cleaned_text)
        # Перевірка на наявність потрібних ключів
        if "function_calls" in data and "text" in data:
            return {
                "function_calls": data["function_calls"],
                "text": data["text"]
            }
        else:
            print("У відповіді відсутні ключі 'function_calls' або 'text'")
            return None
    except json.JSONDecodeError as e:
        print(f"Помилка парсингу JSON: {e}")
        return None
def create_mock_api_response(function_call):
    func_name = function_call['name']
    arguments = function_call['arguments']

    mock_responses = {
        'get_issue_by_date': f"Знайдено 3 завдання на {arguments.get('value_1', 'вказану дату')}",
        'get_issue_by_id': f"Завдання #{arguments.get('value_1', 'ID')}: Розробка нової функції (Статус: В роботі)",
        'get_issue_by_name': f"Завдання '{arguments.get('value_1', 'назва')}': Активне завдання",
        'get_issue_status': f"Статус завдання: В роботі",
        'get_issue_hours': f"Витрачено годин: 12.5",
        'fill_issue_hours': f"Години успішно заповнені: {arguments.get('value_2', 'N')} год.",
        'get_user_status': "Ваш статус: На роботі",
        'set_user_status': f"Статус змінено на: {arguments.get('value_1', 'новий статус')}",
        'create_issue': f"Створено завдання: '{arguments.get('value_1', 'назва')}'",
        'assign_issue': f"Завдання призначено користувачу: {arguments.get('value_2', 'користувач')}",
        'get_wiki_info': f"Wiki інформація знайдена про: {arguments.get('value_1', 'тему')}"
    }
    return mock_responses.get(func_name, "Функція виконана успішно")

# --- Streamlit UI ---
st.set_page_config(page_title="HR Assistant Demo", page_icon="🤖", layout="wide")

st.title("🤖 HR Assistant для Redmine")

tokenizer, model = load_model()
if tokenizer is None or model is None:
    st.stop()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Ваш запит")
    input_text = st.text_area("Введіть запит українською:", height=120)
    max_tokens = st.slider("Макс. токенів відповіді:", 50, 512, MAX_NEW_TOKENS)
    temperature = st.slider("Температура генерації:", 0.1, 1.0, TEMPERATURE, 0.1)
    top_k = st.slider("Top-K", 1, 100, TOP_K)
    top_p = st.slider("Top-P", 0.0, 1.0, TOP_P)
    repetition_penalty = st.slider("Repetition Penalty", 1.0, 2.0, REPETITION_PENALTY)
    generate_btn = st.button("Згенерувати відповідь")

with col2:
    st.subheader("Відповідь асистента")
    if generate_btn:
        if not input_text.strip():
            st.warning("Введіть текст запиту.")
        else:
            with st.spinner("Генеруємо відповідь..."):
                prompt = build_prompt(input_text)
                response = generate_response(
                    tokenizer, 
                    model,
                    prompt,
                    max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )
                parsed = parse_response(response)
                if parsed:
                    func_call = parsed["function_calls"]
                    text = parsed["text"]
                    print(f"Response: {parsed}")

                    st.markdown("### Текстова відповідь:")
                    st.write(text)

                    st.markdown("### Виклик функції:")
                    st.json(func_call)

                    # Мокова відповідь API
                    mock_api_response = create_mock_api_response(func_call[0])
                    st.success(f"📡 Мокова відповідь API: {mock_api_response}")

                else:
                    st.warning("Не вдалося розпізнати JSON у відповіді.")
                st.text_area("Відповідь моделі:", value=response, height=200)
                