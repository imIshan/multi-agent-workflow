import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

load_dotenv()

class Plan(BaseModel):
    task_type: str = Field(
        description= 'The type of task. Example: explanation, comparison, research, coding, debugging.'
    )
    needs_retrieval: bool = Field(
        description= 'Whether this task needs searching documents or vector database retrieval.'
    )
    steps: list[str] = Field(
        description='A short list of steps needed to complete the task.'
    )

def create_planner_agent():
    model_name = os.getenv('OLLAMA_MODEL', 'llama3')
    llm = ChatOllama(
        model=model_name,
        temperature=0
    )
    structured_llm = llm.with_structured_output(Plan)
    return structured_llm
