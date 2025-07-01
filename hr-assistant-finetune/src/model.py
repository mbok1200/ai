from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class HRAssistantModel:
    def __init__(self, model_name='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def generate_response(self, prompt, max_length=150):
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        outputs = self.model.generate(inputs, max_length=max_length, num_return_sequences=1)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def fine_tune(self, train_data, epochs=3, learning_rate=5e-5):
        # Fine-tuning logic will be implemented here
        pass

    def save_model(self, save_directory):
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    def load_model(self, load_directory):
        self.model = GPT2LMHeadModel.from_pretrained(load_directory)
        self.tokenizer = GPT2Tokenizer.from_pretrained(load_directory)