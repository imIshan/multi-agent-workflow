import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()

def create_writer_agent():
    model_name = os.getenv('OLLAMA_MODEL', 'llama3')

    llm = ChatOllama(
        model=model_name,
        temperature=0.3
    )
    return llm