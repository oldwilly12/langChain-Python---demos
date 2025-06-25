"""
📚 Ejercicio 2 - Chat con documentos locales (RAG básico)

🎯 Objetivo:
Construir un asistente que responda preguntas basadas en documentos locales utilizando el patrón RAG (Retrieval-Augmented Generation). 

🛠️ Herramientas utilizadas:
- `DirectoryLoader` o `PyPDFLoader` para cargar archivos locales (PDF o TXT).
- `RecursiveCharacterTextSplitter` para dividir texto en fragmentos pequeños.
- `OpenAIEmbeddings` para transformar los fragmentos en vectores numéricos.
- `FAISS` como base vectorial para búsqueda semántica.
- `ChatOpenAI` como modelo generativo.
- `RetrievalQA` para combinar recuperación y generación de respuestas.

💡 Ideas de extensión:
- Cambiar FAISS por Chroma o Qdrant.
- Guardar y recargar la base vectorial desde disco.
- Añadir LangGraph para flujo más personalizado y trazable.

📦 Requisitos:
- langchain
- openai
- faiss-cpu
- python-dotenv
- pypdf (si se usan PDFs)
"""







