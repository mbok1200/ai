# AI Assistant & Document Chat

This project is a Streamlit-based AI assistant that allows you to chat with your own documents. It leverages LangChain, ChromaDB, HuggingFace embeddings, and AWS Bedrock (Claude 3 Sonnet) to provide conversational retrieval-augmented question answering.

## Features

- **Document Upload & Indexing:** Index and chat with documents from a local folder.
- **Conversational Memory:** Maintains context across chat turns.
- **Source Attribution:** Provides sources for answers when possible.
- **General Chat:** Can answer general questions beyond the document context.

## Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd ai
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure AWS credentials:**
   - Ensure your AWS credentials are set up for Bedrock access.

4. **Prepare your documents:**
   - Place your documents in `/data/gdrive` or update the path in the code.

## Running the App

```bash
streamlit run config/talk_to_your_file_v3.py
```

## Usage

- Use the sidebar to index documents and manage chat history.
- Ask questions about your documents or have a general conversation.
- Click on sources to view the original document context.

## File Structure

- `config/talk_to_your_file_v3.py` - Main Streamlit app.
- `helpers/helpers_fn.py` - Helper functions for file extraction and mapping.
- `data/gdrive/` - Directory for your documents.

## Requirements

- Python 3.8+
- Streamlit
- LangChain
- ChromaDB
- HuggingFace Transformers
- AWS Bedrock access

## License

MIT License

---
