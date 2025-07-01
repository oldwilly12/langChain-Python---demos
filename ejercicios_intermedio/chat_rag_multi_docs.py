"""
📚 Ejercicio Intermedio 2 - Chat con múltiples documentos

🎯 Objetivo:
Construir un asistente que pueda responder preguntas basadas en varios archivos (PDF o TXT) almacenados en una carpeta, integrando todo el conocimiento en una sola base vectorial.

🧠 ¿Qué aprenderás en este ejercicio?
- Cómo cargar múltiples documentos desde una carpeta usando `DirectoryLoader`
- Cómo dividir su contenido con `RecursiveCharacterTextSplitter`
- Cómo generar embeddings con `OpenAIEmbeddings`
- Cómo indexar todos los documentos juntos en una sola colección en Qdrant
- Cómo realizar búsquedas semánticas que abarquen varios textos

🛠️ Herramientas utilizadas:
- `DirectoryLoader`: carga masiva de documentos
- `RecursiveCharacterTextSplitter`: fragmentación de textos
- `OpenAIEmbeddings`: vectorización de fragmentos
- `QdrantVectorStore`: almacenamiento semántico escalable
- `QdrantClient`: creación y control de la colección

💡 ¿Por qué es importante este ejercicio?
Este ejercicio te prepara para:
- Construir asistentes que consulten **bases documentales reales** (manuales, reportes, políticas)
- Escalar de uno a muchos archivos sin cambiar tu lógica
- Preparar bases de conocimiento más completas para agentes o apps empresariales

🚀 Este paso es clave antes de construir un agente inteligente que tome decisiones (Ejercicio Avanzado 1) o un asistente profesional como los que usarías en medicina, soporte o ventas.
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

docs_path = os.path.join(os.path.dirname(__file__), "..", "documents")

def load_documents():
    loader = DirectoryLoader(
        path=docs_path,
        glob="**/*.pdf", # Carga todos los archivos en subdirectorios
        show_progress=True
    )
    docs = loader.load()
    len(docs)
    print(docs[0].page_content[:100])


load_documents()