from dotenv import load_dotenv
from langchain.llms import OpenAI
import os
load_dotenv()  # take environment variables from .env.
API_KEY = os.environ.get('OPENAI_API_KEY')

def get_llm():
    """
    Initialize and return the OpenAI LLM instance.
    """
    # Check if the API key is set
    if not API_KEY:
        raise ValueError("API key for OpenAI is not set. Please set the OPENAI_API_KEY environment variable.")
    
    # Initialize the LLM
    llm = OpenAI(model_name="text-davinci-003", openai_api_key=API_KEY)
    
    return llm
def get_response(prompt):
    """
    Generate a response from the LLM using the provided prompt.
    """
    llm = get_llm()
    response = llm(prompt)
    return response