import json
from peft import PeftModel
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

base_model_name = "EleutherAI/gpt-neo-1.3B"
lora_path = "lora_gpt_neo_1_3b_adapter"  # має бути шлях до каталогу з adapter_config.json, adapter_model.bin

# Завантажуємо токенайзер від базової моделі
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Завантажуємо базову модель
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

# Підключаємо LoRA-адаптер
model = PeftModel.from_pretrained(base_model, lora_path)

# Визначаємо пристрій
device = -1#0 if torch.cuda.is_available() else -1

# Ініціалізуємо генератор
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

st.title("HR Assistant Demo")
st.write("ІнструкціЇ:")
instruction = st.checkbox("Проаналізуй цей запит", value=True)
input_text = st.text_area("Вхід", placeholder="Введіть текст запиту...")
def build_interface_prompt(input_text: str, instruction: bool = True) -> str:
    prompt = ""
    if instruction:
        prompt += (
            "### Instruction:\n"
            "You are an AI assistant. Given an input query, analyze it and generate a structured JSON response.\n"
            "The response must contain an object with the following keys:\n"
            " - 'text': a helpful answer to the query (in the same language as the input).\n"
            " - 'metadata': an object with:\n"
            "    - 'source': a valid URL starting with 'https://'\n"
            "    - 'language': the detected language of the input (ISO 639-1, like 'en' or 'uk')\n"
            " - 'function_method' (optional): if applicable, specify the internal function to call for this query.\n"
            "   For example: 'get_weather', 'convert_currency', 'search_wikipedia', or leave null.\n\n"
        )
    prompt += f"""### Input:
    {input_text}

    ### Output:
    """
    return prompt
if st.button("Отримати відповідь"):
    with st.spinner("Генеруємо відповідь..."):
        prompt = build_interface_prompt(input_text, instruction)  # Додаємо цю строку
        result = generator(
            prompt,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        print(f"{result}")
        
        generated = result[0]["generated_text"]
        output = generated[len(prompt):].strip()
        
        try:
            parsed = json.loads(output)
            answer_text = parsed.get("text", "").strip()
            metadata = parsed.get("metadata", {})
            source = metadata.get("source", "")
            language = metadata.get("language", "")
        
            st.markdown(f"**Відповідь:**\n\n{answer_text}")
        
            if source or language:
                st.markdown("**Метадані:**")
                if source:
                    st.markdown(f"- Джерело: `{source}`")
                if language:
                    st.markdown(f"- Мова: `{language}`")
        
            if st.checkbox("Показати повну відповідь (JSON)"):
                st.code(output, language="json")
        
        except json.JSONDecodeError:
            st.warning("⚠️ Не вдалося розпарсити JSON. Ось сирий результат:")
            st.code(output)