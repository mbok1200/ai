import json
from peft import PeftModel
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import ChatPromptTemplate

base_model_name = "google/gemma-2-2b"
lora_path = "bradi12/lora_model_perfs"

# Завантажуємо токенайзер від базової моделі
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Завантажуємо базову модель
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

# Підключаємо LoRA-адаптер
model = PeftModel.from_pretrained(base_model, lora_path)
model = model.merge_and_unload()  # Critical for materializing tensors
# Визначаємо пристрій
device = -1#0 if torch.cuda.is_available() else -1

# Ініціалізуємо генератор
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

st.title("HR Assistant Demo")
st.write("ІнструкціЇ:")
instruction = st.checkbox("Проаналізуй цей запит", value=True)
input_text = st.text_area("Вхід", placeholder="Введіть текст запиту...")
def build_interface_prompt(input_text: str, instruction: bool = True) -> str:
    prompt = """### Instruction:
        "You are an AI assistant. Given an input query, analyze it and generate a structured JSON response.\n"
            "The response must contain an object with the following keys:\n"
            " - 'text': a helpful answer to the query (in the same language as the input).\n"
            " - 'metadata': an object with:\n"
            "    - 'source': a valid URL starting with 'https://'\n"
            "    - 'language': the detected language of the input (ISO 639-1, like 'en' or 'uk')\n"
            " - 'function_method' (optional): if applicable, specify the internal function to call for this query.\n"
        ### Input:{input_text}
        ### Output:"""



    PROMPT = ChatPromptTemplate.from_template(prompt)
    return PROMPT.format_prompt(
            input_text=input_text
        ).to_string()
if st.button("Отримати відповідь"):
    with st.spinner("Генеруємо відповідь..."):
        prompt = build_interface_prompt(input_text, instruction)
        result = generator(
            prompt,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        generated = result[0]["generated_text"]
        # Витягуємо відповідь після "Assistant:"
        if "Assistant:" in generated:
            answer = generated.split("Assistant:", 1)[-1].strip()
        else:
            answer = generated.strip()
        st.markdown(f"**Відповідь:**\n\n{answer}")