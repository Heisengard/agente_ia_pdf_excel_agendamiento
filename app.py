import streamlit as st
import os
import tempfile
from typing import Annotated, List
from typing_extensions import TypedDict

# --- INICIALIZACIÓN DE SESIÓN GLOBAL ---
# Inicializa vector_store antes de cualquier acceso
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Importaciones de LangChain (El "Pegamento" de las capas)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tracers.stdout import ConsoleCallbackHandler



# Importaciones de LangGraph (La lógica de la Capa 2)
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

 # Cambia a True para ver logs detallados

# --- CONFIGURACIÓN DE PÁGINA (Capa 1: Input/Output) ---
st.set_page_config(page_title="Chat con tu PDF", page_icon="📄")
st.title("📄 Agente de Análisis de Documentos")
st.markdown("Sube un PDF y haz preguntas sobre su contenido usando RAG (Retrieval-Augmented Generation).")

# --- GESTIÓN DE CLAVES ---
with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password")
    if not api_key:
        st.warning("Ingresa tu API Key para continuar.")
        st.stop()
    os.environ["OPENAI_API_KEY"] = api_key

# --- ESTADO DE SESIÓN GLOBAL ---
# Aquí guardamos el "Vector Store" para no reprocesar el PDF con cada pregunta
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- FUNCIÓN DE PROCESAMIENTO (Capa 3: PDF Parser + Capa 4: Vector Store) ---
def process_pdf(uploaded_file):
    """
    Toma el archivo crudo, extrae texto y crea el índice vectorial.
    """
    # 1. Guardar archivo temporalmente (PyPDFLoader necesita ruta física)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # 2. Cargar y dividir el texto (Chunking)
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    
    # Dividimos el texto en pedazos de 1000 caracteres con solapamiento
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 3. Crear Vector Store (Embeddings)
    # Esto convierte texto en números para búsqueda semántica
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(splits, embeddings)
    
    # Limpieza
    os.remove(tmp_path)
    return vector_store

# --- INTERFAZ DE CARGA (Conexión Usuario -> Sistema) ---
uploaded_file = st.file_uploader("Sube tu archivo PDF aquí", type="pdf")

if uploaded_file and st.session_state.vector_store is None:
    with st.spinner("⏳ Procesando documento (Parseando e Indexando)..."):
        st.session_state.vector_store = process_pdf(uploaded_file)
    st.success("✅ Documento procesado y memorizado en la Capa 4.")

# --- DEFINICIÓN DE HERRAMIENTAS RAG (Capa 3: Tool Layer) ---
if st.session_state.vector_store:
    
    # 1. Crear el objeto buscador (Retriever) desde la memoria
    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
    
    # 2. Crear una función "wrapper" local
    # Esta función YA TIENE el retriever guardado en su memoria local (Closure).
    # No necesita llamar a st.session_state, evitando el error de hilos.
    def search_function(query: str):
        docs = retriever.invoke(query)
        return "\n\n".join([d.page_content for d in docs])

    # 3. Empaquetar manualmente la herramienta
    retrieve_tool = Tool(
        name="pdf_search",
        func=search_function,
        description="Usa esta herramienta para buscar información relevante dentro del PDF subido. Úsala cuando te pregunten algo específico del documento."
    )

    tools = [retrieve_tool]

    # --- DEFINICIÓN DEL AGENTE (Capa 2: Agent Modules) ---
    class AgentState(TypedDict):
        messages: Annotated[List, add_messages]

    # Usamos gpt-3.5-turbo (o gpt-4o-mini si tienes saldo)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    # Construcción del Grafo
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    
    # Creamos el nodo de herramientas con nuestra herramienta manual
    workflow.add_node("tools", ToolNode(tools)) 
    
    workflow.set_entry_point("agent")
    
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", "__end__": END}
    )
    workflow.add_edge("tools", "agent")
    
    app = workflow.compile()

    # --- CHAT INTERFACE (Capa 1: Input/Output) ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        elif isinstance(msg, AIMessage) and msg.content:
            st.chat_message("assistant").write(msg.content)

    user_input = st.chat_input("Pregunta algo sobre el PDF...")

    if user_input:
        st.chat_message("user").write(user_input)
        st.session_state.messages.append(HumanMessage(content=user_input))

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("🔍 *Consultando la base de conocimiento...*")
            
            # Ejecutar el grafo
            inputs = {"messages": st.session_state.messages}
            
            result = app.invoke(
                inputs,
                config={"callbacks": [ConsoleCallbackHandler()]}            
            )
            final_msg = result["messages"][-1].content
            placeholder.markdown(final_msg)
            
            st.session_state.messages = result["messages"]

else:
    st.info("👆 Por favor sube un PDF para activar el cerebro del Agente.")