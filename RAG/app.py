import streamlit as st
from configuration import Config
from embeddings import Embeddings
from vectorstore import VectorStore
from memory import Memory
from conversation_chain import ConversationChain
from ui import RAGUI

def main():
    config = Config()
    embeddings = Embeddings(config)    
    vectorstore = VectorStore(config, embeddings)
    memory = Memory(config)
    chain = ConversationChain(config, vectorstore.vectorstore, memory.memory)
    ui = RAGUI(config, vectorstore.vectorstore, memory, chain)
    ui.run()

if __name__ == "__main__":
    main()
