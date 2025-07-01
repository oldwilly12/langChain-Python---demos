"""
📚 Ejercicio Intermedio 1 - Chat con documentos locales (RAG simple)

🎯 Objetivo:
Crear un chatbot que responda preguntas sobre documentos locales utilizando el patrón RAG (Retrieval-Augmented Generation), combinando recuperación semántica con generación LLM.

🛠️ Herramientas utilizadas:
- `TextLoader` o `PyPDFLoader`: para cargar archivos locales.
- `RecursiveCharacterTextSplitter`: para dividir texto en fragmentos.
- `OpenAIEmbeddings`: para convertir fragmentos en vectores.
- `FAISS`: para hacer búsquedas semánticas eficientes.
- `ChatOpenAI`: para generar respuestas basadas en los fragmentos encontrados.
- `RetrievalQA`: para conectar todo el pipeline.

💡 Ideas de extensión:
- Usar múltiples archivos o carpetas.
- Guardar y reutilizar el índice vectorial FAISS.
- Mostrar los fragmentos usados en la respuesta.

📦 Requisitos:
- langchain
- langchain-openai
- faiss-cpu
- pypdf
- python-dotenv
"""

# import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

load_dotenv()

def load_documents(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )

    all_splits = text_splitter.split_documents(documents)
    add_to_vector_store(all_splits)


def add_to_vector_store(documents):
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings_model,
        url="http://localhost:6333",
        collection_name="mi_coleccion_ia",
    )

    query = "What is an agent AI?"
    results =vector_store.similarity_search(query, k=3)
    for i, res in enumerate(results, 1):
        print(f"\nResultado {i}:\n{res.page_content}\n")


load_documents("./a-practical-guide-to-building-agents.pdf")
    
    