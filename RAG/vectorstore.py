import os
from langchain_chroma import Chroma

class VectorStore:
    def __init__(self, config, embeddings):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.vectorstore = Chroma(
            persist_directory=config.PERSIST_DIR,
            embedding_function=embeddings.embeddings,
            collection_name="documents"
        )
    def __call__(self):
        return self.vectorstore
    def count(self):
        return len(self.vectorstore.get()['ids'])
