import torch
from langchain_huggingface import HuggingFaceEmbeddings

class Embeddings:
    def __init__(self, config):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
    def __call__ (self):
        return self.embeddings
