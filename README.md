# Multi-Agent Workflow with Ollama, LangGraph, and Chroma

A local multi-agent AI workflow built in Python.

It includes:
- Planner agent with structured Pydantic output
- Conditional LangGraph routing
- Chroma vector database ingestion
- Retriever agent for document search
- Writer agent for final answer generation
- Local Ollama models for chat and embeddings

Workflow:
User request → Planner → Router → Retriever or Writer → Final Answer