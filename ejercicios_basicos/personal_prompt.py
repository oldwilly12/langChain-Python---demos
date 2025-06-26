"""
📘 Ejercicio Básico 2 - Chatbot con prompt personalizado

🎯 Objetivo:
Crear un asistente controlado con un prompt estructurado, capaz de responder según un estilo o rol definido (por ejemplo, como profesor, médico o soporte técnico).

🛠️ Herramientas utilizadas:
- `PromptTemplate`: Para definir cómo se estructura la entrada.
- `LLMChain`: Para unir el modelo con el prompt.
- `ChatOpenAI`: Modelo de lenguaje para generar las respuestas.

💡 Ideas de extensión:
- Agregar múltiples roles (médico, abogado, etc.)
- Cambiar el formato de salida (respuestas en lista, JSON, etc.)
- Incluir instrucciones de comportamiento en el prompt

📦 Requisitos:
- langchain
- openai
- python-dotenv
"""

import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5
)

prompt_template = ChatPromptTemplate([
    ("system", """Eres un asistente experto en explicar conceptos técnicos de manera sencilla.
Tu tarea es responder preguntas como si le hablaras a un estudiante de secundaria."""),
    ("user", "{pregunta}")
])

question = input("Ingresa una pregunta: ")
messages = prompt_template.invoke({"pregunta": question})

response = llm.invoke(messages)
print(response.content)
















