import os, re, json, webbrowser
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(".env")
class Config:
    PERSIST_DIR = "db"
    CHUNK_SIZE = 1300
    CHUNK_OVERLAP = 150
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID") 
    MAX_EXAMPLES = 9000  # скільки абзаців обробити
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base") 
    EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
    NUM_CHUNKS = 6
    BEDROCK_MODEL = "anthropic.claude-3-sonnet-20240229-v1:0"
    AWS_REGION = "us-east-1"
    TEMPERATURE = 0.3
    MAX_HISTORY = 3
    DATA_DIR = "data/structured"
    FILE_MAP_JSON = os.path.join(os.path.dirname(__file__), "gdrive_file_map.json")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST")
    OPENSEARCH_USER = os.getenv("OPENSEARCH_USER")
    OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD")
    OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", 443))
    TRANSLATION_JSON = os.path.join(os.path.dirname(__file__), "translation.json")
    # --- Multilanguage support ---
    LANGUAGES = {}

    def __init__(self):
        os.makedirs(self.PERSIST_DIR, exist_ok=True)
        os.chmod(self.PERSIST_DIR, 0o770)
        with open(self.TRANSLATION_JSON, "r", encoding="utf-8") as f:
            self.LANGUAGES = json.load(f)
    def batch_upsert(self, index, vectors, batch_size=50, namespace="default"):
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            index.upsert(
                vectors=batch,
                namespace=namespace
            )
    def to_ascii_id(self, s):
        # Залишаємо тільки латиницю, цифри, дефіс, підкреслення і крапку
        s = s.encode("ascii", "ignore").decode()
        s = re.sub(r"[^A-Za-z0-9_\-\.]", "_", s)
        return s
    @st.cache_resource(show_spinner=False)
    # Load file map from JSON
    def load_file_map(_self):
        try:
            with open(_self.FILE_MAP_JSON, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            return {}
    def handle_segment_change(self):
        file_map = self.get_file_map()
        selected = st.session_state["sources"] 
        webbrowser.open_new_tab(file_map[selected])
    def sanitize_filename(self, filename):
        return re.sub(r'[\\/:"*?<>|]+', '_', filename)
    def get_all_files(self, root_dir):
        return [str(p) for p in Path(root_dir).rglob("*") if p.is_file()]

    def t(self, key, **kwargs):
        return self.LANGUAGES[st.session_state.lang][key].format(**kwargs)

