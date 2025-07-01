from helpers_fn import extract_text_from_local_file, get_all_files, google_cse_search, load_file_map
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
import os
import webbrowser
from pathlib import Path

class RAGUI:
    def __init__(self, config, vectorstore, memory, chain):
        self.config = config
        self.vectorstore = vectorstore
        self.memory = memory
        self.chain = chain.chain()  # Initialize the chain with the vectorstore and memory
        self.llm = chain.llm  # Get the LLM from the chain
        self.file_map = self.get_file_map()
        self.initialize_functions()
    def initialize_functions(self):
        if 'chain' not in st.session_state:
            st.session_state.chain = self.chain
        if 'memory' not in st.session_state:
            st.session_state.memory = self.memory
        if 'llm' not in st.session_state:
            st.session_state.llm = self.llm
        if 'lang' not in st.session_state:
            st.session_state.lang = "uk"
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'vectorstore' not in st.session_state:
            st.session_state.vectorstore = self.vectorstore
        if 'processed_files' not in st.session_state:
            st.session_state.processed_files = set()
    def get_file_map(self):
        return load_file_map()
    def handle_segment_change(self):
        selected = st.session_state["sources"] 
        webbrowser.open_new_tab(self.file_map[selected])
    def t(self, key, **kwargs):
        return self.config.LANGUAGES[st.session_state.lang][key].format(**kwargs)

    def run(self):
        # --- Language selector ---
        with st.sidebar:
            st.selectbox(
                "🌐 Мова / Language",
                options=["uk", "en"],
                format_func=lambda x: {"uk": "Українська", "en": "English"}[x],
                key="lang",
                index=0 if st.session_state.lang == "uk" else 1,
                on_change=lambda: st.rerun()
            )

        # --- UI with multilanguage ---
        st.title(self.t("title"))
        st.markdown(self.t("description"))

        with st.sidebar:
            st.header(self.t("sidebar_header"))
            # Додаємо чекбокс для очищення ChromaDB
            clear_db = st.checkbox("Очистити ChromaDB перед індексацією", value=False)
            if st.button(self.t("index_btn")):
                if clear_db:
                    # Видаляємо всі файли з папки db/
                    import shutil
                    if os.path.exists(self.config.PERSIST_DIR):
                        shutil.rmtree(self.config.PERSIST_DIR)
                        os.makedirs(self.config.PERSIST_DIR, exist_ok=True)
                    # Оновлюємо vectorstore після очищення
                    st.session_state.vectorstore = self.vectorstore
                    st.session_state.processed_files = set()
                    st.session_state.memory.clear()
                with st.spinner(self.t("indexing")):
                    files = get_all_files(self.config.DATA_DIR)
                    print(f"DEBUG: Found {len(files)} files in {self.config.DATA_DIR}")
                    processed_count = 0
                    for file_path in files:
                        filename = os.path.relpath(file_path, self.config.DATA_DIR)
                        source_path = None
                        filenamesplit = os.path.splitext(os.path.basename(filename))[0]
                        if filename not in st.session_state.processed_files:
                            content = extract_text_from_local_file(file_path)
                            if not content.strip():
                                st.warning(f"{self.t('error', error='Не вдалося витягти текст з файлу ' + filename + ' або файл порожній.')}")
                                continue
                            text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=self.config.CHUNK_SIZE,
                                chunk_overlap=self.config.CHUNK_OVERLAP,
                                separators=["\n\n", "\n", ".", " "]
                            )
                            if filenamesplit in self.file_map:
                                source_path = None
                                for key in self.file_map:
                                    if filenamesplit.lower() in key.lower():
                                        source_path = self.file_map[key]
                                        break
                            chunks = [chunk for chunk in text_splitter.split_text(content) if chunk.strip()]
                            if not chunks:
                                st.warning(f"{self.t('error', error='Не вдалося створити непорожні чанки з файлу ' + filename)}")
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
                            except Exception as e:
                                st.warning(self.t("error", error=f"Error adding chunks from {filename}: {e}"))
                            st.session_state.processed_files.add(filename)
                            processed_count += 1
                    st.success(self.t("index_done", count=processed_count))

            if st.button(self.t("clear_chat")):
                st.session_state.chat_history = []
                if hasattr(st.session_state.memory, "chat_memory"):
                    st.session_state.memory.chat_memory.clear()
                st.rerun()
            if st.button(self.t("check_docs")):
                try:
                    count = st.session_state.vectorstore.count()
                except Exception:
                    count = 0
                st.info(self.t("docs_count", count=count))
            st.segmented_control(
                self.t("sources"),
                options=self.file_map.keys(),
                on_change=self.handle_segment_change,
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
                                st.info(self.t("no_info"))
                            for source in message["sources"]:
                                filenamesplit = os.path.splitext(os.path.basename(source['filename']))[0]
                                st.markdown(self.t("from", filename=source['filename']))
                                file_url = self.file_map.get(filenamesplit.strip())
                                if file_url:
                                    st.link_button(filenamesplit, file_url)
                                else:
                                    st.info(self.t("error", error=f"No file path found for {filenamesplit}"))

        if question := st.chat_input(self.t("ask")):
            st.session_state.chat_history.append({"is_assistant": False, "content": question})
            st.session_state.chat_history.append({"is_assistant": True, "content": self.t("thinking"), "pending": True})
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
                                    "filename": doc.metadata.get("filename", "Unknown"),
                                    "text": doc.page_content
                                })
                        if not sources:
                            google_answer = google_cse_search(st.session_state.pending_question)
                            answer += self.t("google", answer=google_answer)
                        st.session_state.chat_history[idx] = {
                            "is_assistant": True,
                            "content": answer,
                            "sources": sources if sources else None
                        }
                    except Exception as e:
                        st.session_state.chat_history[idx] = {
                            "is_assistant": True,
                            "content": self.t("error", error=str(e))
                        }
                    st.session_state.pending_question = None
                    st.rerun()
                    break
