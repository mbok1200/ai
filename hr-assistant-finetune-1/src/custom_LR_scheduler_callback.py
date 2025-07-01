from transformers import TrainerCallback


class CustomLRSchedulerCallback(TrainerCallback):
    def __init__(self, initial_lr=1e-6, warmup_steps=100, decay_factor=0.1):
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.decay_factor = decay_factor
        self.current_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        if self.current_step < self.warmup_steps:
            lr = self.initial_lr * (self.current_step + 1) / self.warmup_steps
        else:
            lr = self.initial_lr * (self.decay_factor ** ((self.current_step - self.warmup_steps) // 100))
        
        for param_group in kwargs['optimizer'].param_groups:
            param_group['lr'] = lr
        
        self.current_step += 1
        return control