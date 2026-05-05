import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

load_dotenv()

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "multi_agent_docs"

def load_text_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def create_documents() -> list[Document]:
    file_path = "data/ai_frameworks.txt"
    text = load_text_file(file_path)
    document = Document(
        page_content=text,
        metadata={"source": file_path}
    )
    return [document]

def ingest_documents():
    embedding_model_name = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    embeddings = OllamaEmbeddings(model=embedding_model_name)

    documents = create_documents()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )

    vector_store.add_documents(documents)

    print(f"Ingested {len(documents)} document(s) into Chroma.")
    print(f"Chroma path: {CHROMA_PATH}")
    print(f"Collection name: {COLLECTION_NAME}")

if __name__ == "__main__":
    ingest_documents()