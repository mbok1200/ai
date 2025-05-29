

import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import pinecone

# Завантаження ключів з оточення
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "langchain-demo"

# Ініціалізація Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Створення індексу, якщо його ще немає
if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine"
    )

# Завантаження документів
loader = TextLoader("example.txt")  # Ваш текстовий файл
documents = loader.load()

# Розбиття на чанки
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Ініціалізація ембеддингів OpenAI
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Підключення до Pinecone через LangChain
vectorstore = Pinecone.from_documents(docs, embeddings, index_name=INDEX_NAME)

# Ініціалізація LLM
llm = OpenAI(openai_api_key=OPENAI_API_KEY)

# Створення ланцюга пошуку + генерації
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# Приклад запиту
query = "Що таке LangChain?"
result = qa.run(query)
print(result)
