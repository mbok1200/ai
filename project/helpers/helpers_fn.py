import torch, os
import streamlit as st
from pinecone import (
    Pinecone,
    ServerlessSpec,
    CloudProvider,
    AwsRegion,
    VectorType
)
from googleapiclient.discovery import build
from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy import OpenSearch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_aws import ChatBedrock
from helpers.configuration import Config
config = Config()

def google_cse_search(query, lang="lang_uk"):
    try:
        service = build("customsearch", "v1", developerKey=config.GOOGLE_API_KEY)
        res = service.cse().list(
            q=query,
            cx=config.GOOGLE_CSE_ID,
            lr=lang,
            num=3
        ).execute()
        results = res.get("items", [])
        if not results:
            return "No results found."
        snippets = []
        for item in results:
            title = item.get("title", "")
            link = item.get("link", "")
            snippet = item.get("snippet", "")
            snippets.append(f"**[{title}]({link})**\n{snippet}")
        return "\n\n".join(snippets)
    except Exception as e:
        return f"Google CSE error: {e}"


@st.cache_resource(show_spinner=False)
def init_embeddings():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource(show_spinner=False)
def init_pinecone():
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index_name = "streamlit"
    if index_name not in pc.list_indexes().names():
        # pc.delete_index(index_name)
        pc.create_index(
            name=index_name,
            dimension=768,
            spec=ServerlessSpec(
                cloud=CloudProvider.AWS,
                region=AwsRegion.US_EAST_1
            ),
            vector_type=VectorType.DENSE
        )
    idx = pc.Index(index_name)
    return idx, pc

@st.cache_resource(show_spinner=False)
def init_opensearch():
    client = OpenSearch(
        hosts=[{"host": config.OPENSEARCH_HOST, "port": config.OPENSEARCH_PORT}],
        http_auth=(config.OPENSEARCH_USER, config.OPENSEARCH_PASS),
        use_ssl=True,
        verify_certs=True,
    )
    embeddings = init_embeddings()
    index_name = "your-index"  # замініть на свій індекс
    vectorstore = OpenSearchVectorSearch(
        opensearch_client=client,
        index_name=index_name,
        embedding_function=embeddings,
    )
    return vectorstore

@st.cache_resource(show_spinner=False)
def init_memory():
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

@st.cache_resource(show_spinner=False)
def init_chat_model():
    return ChatBedrock(
        model_id=config.BEDROCK_MODEL,
        model_kwargs={"temperature": config.TEMPERATURE},
        region_name=config.AWS_REGION
    )