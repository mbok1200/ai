from langchain.memory import ConversationBufferMemory

class Memory:
    def __init__(self, config):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    def __call__(self):
        return self.memory
