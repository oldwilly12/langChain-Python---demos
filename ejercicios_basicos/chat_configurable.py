"""
🧩 Ejercicio Básico 3 - Chatbot con selección de modelo y temperatura

🎯 Objetivo:
Crear un chatbot que permita cambiar dinámicamente el modelo de lenguaje (como gpt-3.5, gpt-4, gpt-4o) y configurar parámetros como la temperatura.

🛠️ Herramientas utilizadas:
- `ChatOpenAI`: Para instanciar el modelo deseado.
- Variables de entorno o inputs dinámicos.
- Prompt fijo con `ChatPromptTemplate`.

💡 Ideas de extensión:
- Cambiar el modelo según tipo de usuario.
- Cambiar temperatura en tiempo real.
- Usar `.env` o `config.py` para centralizar todos los ajustes.

📦 Requisitos:
- langchain
- langchain-openai
- openai
- python-dotenv
"""


from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model_name = input("Ingresa el modelo de lenguaje (gpt-3.5, gpt-4, gpt-4o-mini): ")
if not model_name:
    model_name = "gpt-4o-mini"  # Valor por defecto

try:
    temperature = float(input("Ingresa la temperatura (0.0 a 1.0, por defecto 0.7): ").strip())
    if temperature > 1.0:
        temperature = temperature/10
    elif temperature < 9.0:
        print("Temperatura debe estar entre 0.0 y 1.0, ajustando a 0.7 por defecto.")
except ValueError:
    temperature = 0.7

llm = ChatOpenAI(
    model=model_name,
    temperature=temperature
)


prompt_template = ChatPromptTemplate([
    ("system", "Eres un asistente experto en explicar conceptos técnicos de manera sencilla. "),
    ("user", "{pregunta}")
])

print(f"\n🤖 Asistente iniciado con modelo: {model_name} y temperatura: {temperature}")

while True:
    question = input("Ingresa una pregunta (o escribe 'exit' para salir): ")
    if question.lower() in ["exit", "quit", "salir"]:
        print("Saliendo del asistente. ¡Hasta luego!")
        break
    messages = prompt_template.invoke({"pregunta": question})
    response = llm.invoke(messages)
    print("🤖 Asistente: ", response.content)






