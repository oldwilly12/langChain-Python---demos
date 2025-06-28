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
# from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_documents(file_path):
    file_path = "./a-practical-guide-to-building-agents.pdf"
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )

    all_splits = text_splitter.split_documents(documents)
    create_vector_store(all_splits)


def create_vector_store(documents):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    