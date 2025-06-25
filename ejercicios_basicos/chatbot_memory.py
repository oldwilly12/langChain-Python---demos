"""
🔍 Ejercicio 1 - Chatbot básico con memoria

🎯 Objetivo:
Crear un chatbot conversacional utilizando LangChain en Python, que recuerde el contexto de la conversación utilizando memoria de tipo buffer.

🛠️ Herramientas clave:
- `ChatOpenAI`: Modelo de lenguaje de OpenAI para generar respuestas.
- `ConversationChain`: Cadena conversacional que une el modelo con la memoria.
- `ConversationBufferMemory`: Tipo de memoria que guarda el historial de mensajes completo en orden cronológico.

💡 Tips útiles:
- Puedes usar `memory.save_context()` para guardar manualmente una conversación y `memory.load_memory_variables()` para cargarla más tarde.
- Ideal para asistentes simples o chatbots que necesiten recordar lo que el usuario dijo anteriormente.
- Para extender este ejemplo, podrías conectar con archivos, bases de datos, o cambiar la memoria a tipos como `ConversationSummaryMemory`.

📦 Requisitos:
- LangChain
- OpenAI
- python-dotenv (para leer API keys desde .env)
"""


import os
import sys
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import OPENAI_API_KEY

config = {"configurable": {"thread_id": "abc123"}}
model = init_chat_model("gpt-4o-mini", model_provider="openai")

# Define a new graph
workflow = StateGraph( state_schema=MessagesState)

# Define the function that calls the model
def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}

# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add a memory saver to the graph
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config)
    ai_message = output["messages"][-1]
    print(memory)
    print(f"AI: {ai_message.content}")
