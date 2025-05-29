import os
from pathlib import Path
# Now import everything else
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_aws import ChatBedrock
import webbrowser
from helpers.helpers_fn import extract_text_from_local_file, get_all_files, load_file_map
# Configuration
PERSIST_DIR = "db"  # ChromaDB persistence directory
CHUNK_SIZE = 1000   # Size of text chunks
CHUNK_OVERLAP = 100  # Overlap between chunks
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"  # Model for embeddings
NUM_CHUNKS = 2  # Number of relevant chunks to retrieve
BEDROCK_MODEL = "anthropic.claude-3-sonnet-20240229-v1:0"  # Claude model version
AWS_REGION = "us-east-1"  # AWS region for Bedrock
TEMPERATURE = 0.7  # Temperature for response generation
MAX_HISTORY = 2  # Maximum number of conversation turns to remember

# Initialize ChromaDB with persistence
os.makedirs(PERSIST_DIR, exist_ok=True)

# Initialize LangChain chat model
@st.cache_resource(show_spinner=False)
def init_chat_model():
    return ChatBedrock(
        model_id=BEDROCK_MODEL,
        model_kwargs={"temperature": TEMPERATURE},
        region_name=AWS_REGION
    )

# Initialize embeddings
@st.cache_resource(show_spinner=False)
def init_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

# Initialize ChromaDB and create retriever
@st.cache_resource(show_spinner=False)
def init_chromadb():
    """Initialize ChromaDB with persistence and transformer embeddings"""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Initialize embeddings
    embeddings = init_embeddings()
    
    # Create or load Chroma vector store
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name="documents"
    )
    
    return vectorstore

# Initialize conversation memory
@st.cache_resource(show_spinner=False)
def init_memory():
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

def handle_segment_change():
    selected = st.session_state["sources"] 
    webbrowser.open_new_tab(file_map[selected])
# Create the conversation chain
def create_conversation_chain(llm, retriever, memory):
    template = """You are a helpful and friendly AI assistant.
    Your tasks:
    1. Answer questions using the information provided in the documents below.
    2. If the information in the documents is insufficient, honestly state that you cannot find the answer.
    3. If the question is unrelated to the documents, feel free to engage in general friendly conversation.
    4. Remember the context of the conversation to provide better answers.
    5. Use the context from the documents to answer the question.
    6. Provide sources for your answers when possible.

    Current conversation:
    {chat_history}

    Context from documents:
    {context}

    Human: {question}
    Assistant:"""

    PROMPT = ChatPromptTemplate.from_template(template)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True,
        verbose=True
    )

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'vectorstore' not in st.session_state:
    vectorstore = init_chromadb()
    st.session_state.vectorstore = vectorstore
    st.session_state.llm = init_chat_model()
    st.session_state.memory = init_memory()
    st.session_state.chain = create_conversation_chain(
        st.session_state.llm,
        vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={'k': 6, 'lambda_mult': 0.25}
        ),
        st.session_state.memory
    )

if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()

# Streamlit UI
st.title("🤖 AI Assistant & Document Chat")
st.markdown("""
This AI assistant can:
- Answer questions about your uploaded documents
- Remember details from your conversation
- Help with general questions
- Maintain context across the chat
""")
file_map = load_file_map()  # Load file map from JSON

# Sidebar for file upload
with st.sidebar:
    st.header("📄 Document Upload")
    if st.sidebar.button("Індексувати локальну папку data/drive"):
        with st.spinner("Індексація файлів у /data/gdrive..."):
            root_folder = "/home/mikola/projects/ai/data/gdrive"  
            files = get_all_files(root_folder)
            processed_count = 0
            for file_path in files:
                filename = os.path.relpath(file_path, root_folder) 
                source_path = None
                filenamesplit = os.path.basename(filename)  # Normalize path for web
                if filename not in st.session_state.processed_files:
                    content = extract_text_from_local_file(file_path)
                    if not content.strip():
                        st.warning(f"Не вдалося витягти текст з файлу {filename} або файл порожній.")
                        continue
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=CHUNK_SIZE,
                        chunk_overlap=CHUNK_OVERLAP,
                        separators=["\n\n", "\n", ".", " "]
                    )
                    
                    if filenamesplit in file_map:
                        source_path = file_map.get(filenamesplit.strip())
                    chunks = text_splitter.split_text(content)
                    chunks = [chunk for chunk in chunks if chunk.strip()]
                    if not chunks:
                        st.warning(f"Не вдалося створити непорожні чанки з файлу {filename}")
                        continue
                    try:
                        st.session_state.vectorstore.add_texts(
                            texts=chunks,
                            metadatas=[{
                                "filename": filename, 
                                "type": Path(file_path).suffix,
                                "source": source_path if source_path else "Unknown"
                            }] * len(chunks)
                        )
                        print(f"Successfully added {len(chunks)} chunks from {filename}")
                    except Exception as e:
                        print(f"Error adding chunks from {filename}: {e}")

                    st.session_state.processed_files.add(filename)
                    processed_count += 1
            st.success(f"Індексація завершена. Оброблено файлів: {processed_count}")
    
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.memory.clear()
        st.rerun()
    if st.sidebar.button("Перевірити кількість документів у ChromaDB"):
        count = st.session_state.vectorstore._collection.count()
        st.sidebar.info(f"Кількість документів у ChromaDB: {count}")
    st.segmented_control(
        "Sources",
        options=file_map.keys(),
        on_change=handle_segment_change,
        key="sources",
        selection_mode="single"
    )
# Chat interface
chat_container = st.container()
with chat_container:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message("assistant" if message["is_assistant"] else "user"):
            st.write(message["content"])
            if message.get("sources"):
                with st.expander("View sources"):
                    if not message["sources"]:
                        st.info("🤔 Не знайдено релевантної інформації у документах. Спробуй переформулювати питання.")

                    for source in message["sources"]:
                        filenamesplit = os.path.basename(source['filename'])
                        st.markdown(f"""
                        **From**: {source['filename']}
                        **Text**: \n\n{source['text']}
                        **Source url**: \n\n{file_map.get(filenamesplit.strip())}
                        ---
                        """)

# User input
if question := st.chat_input("Ask a question about your documents or chat with me"):
    # Add user message to chat
    st.session_state.chat_history.append({"is_assistant": False, "content": question})
    
    # Get response from chain
    try:
        result = st.session_state.chain({"question": question})
        answer = result["answer"]
        sources = []
        
        if result.get("source_documents"):
            for doc in result["source_documents"]:
                sources.append({
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "text": doc.page_content
                })
        
        # Add assistant response to chat
        st.session_state.chat_history.append({
            "is_assistant": True,
            "content": answer,
            "sources": sources if sources else None
        })
        
        # Rerun to update chat display
        st.rerun()
        
    except Exception as e:
        st.error(f"Error: {str(e)}") 