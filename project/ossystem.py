import os
from pathlib import Path
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from helpers.docs_fn import extract_text_from_local_file
from helpers.helpers_fn import (
    google_cse_search,
    init_embeddings,
    init_pinecone,
    init_memory,
    init_chat_model
)
from helpers.configuration import Config
config = Config()
file_map = config.load_file_map()

# --- Custom Pinecone retriever ---
def pinecone_retrieve(query, embeddings, index, k=config.NUM_CHUNKS, namespace="default"):
    query_emb = embeddings.embed_query(query)
    res = index.query(
        vector=query_emb,
        top_k=k,
        include_metadata=True,
        namespace=namespace
    )
    docs = []
    for match in res.matches:
        docs.append({
            "page_content": match.metadata.get("text", ""),
            "metadata": match.metadata
        })
    return docs

# --- LangChain chain with custom retriever ---
def create_conversation_chain(llm, memory, embeddings, index):
    template = """You are a helpful and friendly AI assistant.
    Your tasks:
    1. Answer questions using the information provided in the documents below.
    2. If the information in the documents is insufficient, honestly state that you cannot find the answer.
    3. If the question is unrelated to the documents, feel free to engage in general friendly conversation.
    4. Remember the context of the conversation to provide better answers.
    5. Use the context from the documents to answer the question.
    6. Provide sources for your answers when possible.
    7. if the question is not related to the documents, you can use Google Search to find the answer.
    8. If you use Google Search, provide the answer in a friendly manner.
    9. If user ask give them answer in not Ukrainian, you can use the answer in Ukrainian.
    10. Always answer in Ukrainian, regardless of the language of the question.
    Current conversation:
    {chat_history}

    Context from documents:
    {context}
    Question: {question}
    Assistant:"""
    PROMPT = ChatPromptTemplate.from_template(template)

    def custom_chain(inputs):
        question = inputs["question"]
        docs = pinecone_retrieve(
            question,
            embeddings,
            index,
            k=config.NUM_CHUNKS
        )
        context = "\n\n".join([doc["page_content"] for doc in docs])
        prompt = PROMPT.format_prompt(
            chat_history="\n".join([
                m['content'] for m in st.session_state.chat_history
                if isinstance(m, dict) and not m.get("pending")
            ]),
            context=context,
            question=question
        ).to_string()
        answer_obj = llm.invoke(prompt)
        answer = answer_obj.content if hasattr(answer_obj, "content") else str(answer_obj)
        if not answer.strip():
            answer = config.t("no_info")
        return {
            "answer": answer,
            "source_documents": docs
        }
    return custom_chain

# --- Session state ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'pinecone_index' not in st.session_state:
    pinecone_index, pinecone_client = init_pinecone()
    st.session_state.pinecone_index = pinecone_index
    st.session_state.pinecone_client = pinecone_client
    st.session_state.embeddings = init_embeddings()
    st.session_state.llm = init_chat_model()
    st.session_state.memory = init_memory()
    st.session_state.chain = create_conversation_chain(
        st.session_state.llm,
        st.session_state.memory,
        st.session_state.embeddings,
        pinecone_index
    )

if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()

if 'lang' not in st.session_state:
    st.session_state.lang = "uk"


# --- Language selector ---
with st.sidebar:
    st.selectbox(
        "🌐 Мова / Language",
        options=["uk", "en"],
        format_func=lambda x: "Українська" if x == "uk" else "English",
        key="lang",
        index=0 if st.session_state.lang == "uk" else 1,
        on_change=lambda: st.rerun()
    )

# --- UI with multilanguage ---
st.title(config.t("title"))
st.markdown(config.t("description"))

with st.sidebar:
    st.header(config.t("sidebar_header"))
    if st.button(config.t("index_btn")):
        with st.spinner(config.t("indexing")):
            files = config.get_all_files(config.DATA_DIR)
            processed_count = 0
            embeddings = st.session_state.embeddings
            index = st.session_state.pinecone_index
            for file_path in files:
                filename = os.path.relpath(file_path, config.DATA_DIR)
                source_path = None
                filenamesplit = os.path.splitext(os.path.basename(filename))[0]
                if filename not in st.session_state.processed_files:
                    content = extract_text_from_local_file(file_path)
                    if not content.strip():
                        st.warning(f"{config.t('error', error='Не вдалося витягти текст з файлу ' + filename + ' або файл порожній.')}")
                        continue
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=config.CHUNK_SIZE,
                        chunk_overlap=config.CHUNK_OVERLAP,
                        separators=["\n\n", "\n", ".", " "]
                    )
                    if filenamesplit in file_map:
                        source_path = None
                        for key in file_map:
                            if filenamesplit.lower() in key.lower():
                                source_path = file_map[key]
                                break
                    chunks = [chunk for chunk in text_splitter.split_text(content) if chunk.strip()]
                    if not chunks:
                        st.warning(f"{config.t('error', error='Не вдалося створити непорожні чанки з файлу ' + filename)}")
                        continue
                    metadatas = [{
                        "filename": filename,
                        "type": Path(file_path).suffix,
                        "source": source_path if source_path else "Unknown",
                        "text": chunk
                    } for chunk in chunks]
                    try:
                        vectors = []
                        for i, chunk in enumerate(chunks):
                            vector = embeddings.embed_documents([chunk])[0]
                            vector_id = config.to_ascii_id(f"{filename}_{i}")
                            vectors.append((
                                vector_id,
                                vector,
                                metadatas[i]
                            ))
                        config.batch_upsert(index, vectors, batch_size=50, namespace="default")
                    except Exception as e:
                        st.warning(config.t("error", error=f"Error adding chunks from {filename}: {e}"))
                    st.session_state.processed_files.add(filename)
                    processed_count += 1
            st.success(config.t("index_done", count=processed_count))

    if st.button(config.t("clear_chat")):
        st.session_state.chat_history = []
        st.session_state.memory.clear()
        st.rerun()
    if st.button(config.t("check_docs")):
        try:
            index = st.session_state.pinecone_client.Index("streamlit")
            stats = index.describe_index_stats()
            count = stats.get("total_vector_count", 0)
        except Exception as e:
            count = f"Error: {e}"
        st.info(config.t("docs_count", count=count))
    st.segmented_control(
        config.t("sources"),
        options=list(file_map.keys()),
        on_change=config.handle_segment_change,
        key="sources",
        selection_mode="single"
    )

# --- Chat interface ---
chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history:
        with st.chat_message("assistant" if message["is_assistant"] else "user"):
            st.write(message["content"])
            if message.get("sources"):
                with st.expander("View sources"):
                    if not message["sources"]:
                        st.info(config.t("no_info"))
                    for source in message["sources"]:
                        filenamesplit = os.path.splitext(os.path.basename(source['filename']))[0]
                        st.markdown(config.t("from", filename=source['filename']))
                        file_url = file_map.get(filenamesplit.strip())
                        if file_url:
                            st.link_button(filenamesplit, file_url)
                        else:
                            st.info(config.t("error", error=f"No file path found for {filenamesplit}"))

if question := st.chat_input(config.t("ask")):
    st.session_state.chat_history.append({"is_assistant": False, "content": question})
    st.session_state.chat_history.append({"is_assistant": True, "content": config.t("thinking"), "pending": True})
    st.session_state.pending_question = question
    st.rerun()

if getattr(st.session_state, "pending_question", None):
    for idx, msg in enumerate(st.session_state.chat_history):
        if msg.get("pending"):
            try:
                result = st.session_state.chain({"question": st.session_state.pending_question})
                answer = result["answer"]
                sources = []
                if result.get("source_documents"):
                    for doc in result["source_documents"]:
                        sources.append({
                            "filename": doc["metadata"].get("filename", "Unknown"),
                            "text": doc["page_content"]
                        })
                if not sources:
                    google_answer = google_cse_search(st.session_state.pending_question)
                    answer += config.t("google", answer=google_answer)
                st.session_state.chat_history[idx] = {
                    "is_assistant": True,
                    "content": answer,
                    "sources": sources if sources else None
                }
            except Exception as e:
                st.session_state.chat_history[idx] = {
                    "is_assistant": True,
                    "content": config.t("error", error=str(e))
                }
            st.session_state.pending_question = None
            st.rerun()
            break