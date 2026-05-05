import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "multi_agent_docs"

def get_vector_store() -> Chroma:
    embeddings_model_name = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    embeddings = OllamaEmbeddings(model=embeddings_model_name)

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )

    return vector_store

def retrieve_context(user_request: str) -> str:
    vector_store = get_vector_store()

    results = vector_store.similarity_search(query=user_request, k=2)

    if not results:
        return "No relevant context found"
    
    context_parts = []

    for document in results:
        source = document.metadata.get('source', 'unknown source')
        content = document.page_content

        context_parts.append(
            f"Source: {source}\n Content: {content}"
        )
    return "\n\n---\n\n".join(context_parts)