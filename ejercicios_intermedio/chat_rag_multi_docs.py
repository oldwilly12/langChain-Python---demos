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
from langchain import hub
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_qdrant import QdrantVectorStore
from langsmith import Client
from qdrant_client import QdrantClient
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

load_dotenv()

docs_path = os.path.join(os.path.dirname(__file__), "..", "documents")

def load_documents():
    loader = DirectoryLoader(
        path=docs_path,
        glob="**/*.pdf", # Carga todos los archivos en subdirectorios
        show_progress=True,
        loader_cls=PyPDFLoader,  # Carga archivos PDF
    )
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )

    all_splits = text_splitter.split_documents(docs)
    
    add_to_vector_store(all_splits)
    

def add_to_vector_store(docs):
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embeddings_model,
        url="http://localhost:6333",
        collection_name="mi_coleccion_ia",
    )

    print("Este es un Asistente IA con rag multi-documentos. \n Puedes hacer preguntas sobre los documentos cargados.")
    while True:
        query = "What is an agent AI?"
        results =vector_store.similarity_search(query, k=3)
        for i, res in enumerate(results, 1):
            print(f"\nResultado {i}:\n{res.page_content}\n")

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)


 
def consult_qdrant():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # client_smith = Client(
    #     api_key=os.getenv("SMITH_API_KKEY")
    # )
    prompt = hub.pull("rlm/rag-prompt")

    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name="langchain-testing",
        url="https://0b1daa23-c45c-4976-9441-5fcd5a788579.europe-west3-0.gcp.cloud.qdrant.io:6333",
        api_key=os.getenv("QDRANT_API_KEY")
    )

    retriver = qdrant.as_retriever(search_kwargs={"k": 3})

    model_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.5
    )


    print("Puedes hacer preguntas sobre los documentos almacenados.")
    while True:
        query = input("Tu pregunta: ")
        if query.lower() in ["salir", "exit", "quit"]:
            break
        
        format_docs = RunnableLambda(lambda docs: "\n\n".join(doc.page_content for doc in docs))

        qa_chain = (
            {
                "context": retriver | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | model_llm 
            | StrOutputParser()
        )

        respuesta = qa_chain.invoke({"question": query})
        print(f"\n Asistente: \n {respuesta}")

# load_documents()
consult_qdrant()





