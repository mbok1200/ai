import streamlit as st
import torch
from transformers import MT5Tokenizer, MT5ForConditionalGeneration

model_name = './results/checkpoint-final'
device = torch.device("cpu")
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name,  low_cpu_mem_usage=False, use_safetensors=True)
model.to(device)
def postprocess(text):
    # Видаляємо службові токени
    text = text.replace('<pad>', '')
    text = text.replace('<extra_id_0>', '')
    text = text.replace('</s>', '')

    # Видаляємо дублювання слів (якщо модель захардкодила їх)
    import re
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)

    # Видаляємо зайві пробіли
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)

    # Робимо першу літеру великою, якщо потрібно
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    return text
st.title("HR Assistant Text Generation")
st.write("Enter your question or prompt:")
input_text = st.text_area("Input Text")

if st.button("Generate Answer"):
    if input_text:
        prefix = "Find then generate answer use it text "  # Додайте task prefix, якщо використовували при тренуванні
        full_input = prefix + input_text
        inputs = tokenizer(
            full_input, 
            return_tensors='pt', 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()} 
       # inputs = tokenizer(full_input, return_tensors='pt', truncation=True, padding=True, max_length=128).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=256,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                temperature=0.9,
                repetition_penalty=1.5,
                num_return_sequences=1
                # max_length=128,
                # num_beams=4,
                # do_sample=True,
                # top_p=0.95,
                # temperature=0.7
            )
            raw_answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            answer = postprocess(raw_answer)
        st.write(f"Generated Answer: {answer}")
    else:
        st.write("Please enter some text to generate an answer.")