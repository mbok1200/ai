from transformers import Trainer

class CustomTrainer(Trainer):
    def to(self, device):
        # Ігноруємо виклик .to(), бо модель вже розподілена з offloading
        print(f"[CustomTrainer] Пропускаємо model.to({device})")
        return self.model  # просто повертаємо модель без переміщення
