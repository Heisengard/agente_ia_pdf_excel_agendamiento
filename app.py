import streamlit as st
import os
import tempfile
from typing import Annotated, List
from typing_extensions import TypedDict

# --- INICIALIZACIÓN DE SESIÓN GLOBAL ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Importaciones de LangChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tracers.stdout import ConsoleCallbackHandler

# Importaciones de LangGraph
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Chat con Múltiples PDFs", page_icon="📚") # Icono cambiado a libros
st.title("📚 Agente Multi-Documento")
st.markdown("Sube **varios PDFs** y haz preguntas sobre su contenido combinado.")

# --- GESTIÓN DE CLAVES ---
with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password")
    if not api_key:
        st.warning("Ingresa tu API Key para continuar.")
        st.stop()
    os.environ["OPENAI_API_KEY"] = api_key
    
    # Botón para reiniciar la memoria si quieres subir otros archivos
    if st.button("Borrar Memoria y Subir Nuevos"):
        st.session_state.vector_store = None
        st.session_state.messages = []
        st.rerun()

# --- FUNCIÓN DE PROCESAMIENTO (Modificada para múltiples archivos) ---
def process_pdf_list(uploaded_files):
    """
    Toma una LISTA de archivos, extrae texto de todos y crea un único índice.
    """
    all_docs = [] # Lista para acumular el texto de TODOS los PDFs
    
    # Barra de progreso visual
    progress_text = "Procesando archivos..."
    my_bar = st.progress(0, text=progress_text)
    
    total_files = len(uploaded_files)
    
    # ### CAMBIO: Bucle para procesar cada archivo subido ###
    for i, file in enumerate(uploaded_files):
        # 1. Guardar temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name
            
        # 2. Cargar texto
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        
        # Agregamos los documentos de este PDF a la lista general
        all_docs.extend(docs) 
        
        # Limpieza del archivo temporal
        os.remove(tmp_path)
        
        # Actualizar barra de progreso
        my_bar.progress((i + 1) / total_files, text=f"Leyendo archivo {i+1} de {total_files}...")

    # 3. Dividir TODO el texto acumulado
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)

    # 4. Crear Vector Store único con toda la información
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(splits, embeddings)
    
    my_bar.empty() # Limpiar barra de progreso
    return vector_store

# --- INTERFAZ DE CARGA (Modificada) ---
# ### CAMBIO: accept_multiple_files=True ###
uploaded_files = st.file_uploader("Sube tus archivos PDF aquí", type="pdf", accept_multiple_files=True)

if uploaded_files and st.session_state.vector_store is None:
    # Solo procesamos si hay archivos y la memoria está vacía
    with st.spinner("⏳ Procesando biblioteca de documentos..."):
        st.session_state.vector_store = process_pdf_list(uploaded_files)
    st.success(f"✅ {len(uploaded_files)} documentos procesados y memorizados.")

# --- DEFINICIÓN DE HERRAMIENTAS RAG ---
if st.session_state.vector_store:
    
    # El retriever buscará en la mezcla de todos los documentos
    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5}) # Aumenté k a 5 para tener más contexto
    
    def search_function(query: str):
        docs = retriever.invoke(query)
        # Agregamos la fuente (nombre del archivo) a la respuesta para saber de qué PDF vino
        results = []
        for d in docs:
            source = d.metadata.get('source', 'Desconocido')
            content = d.page_content
            results.append(f"[Fuente: {source}]\nContenido: {content}")
            
        return "\n\n".join(results)

    retrieve_tool = Tool(
        name="pdf_search",
        func=search_function,
        description="Usa esta herramienta para buscar información en los documentos PDF subidos."
    )

    tools = [retrieve_tool]

    # --- DEFINICIÓN DEL AGENTE ---
    class AgentState(TypedDict):
        messages: Annotated[List, add_messages]

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools)) 
    
    workflow.set_entry_point("agent")
    
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", "__end__": END}
    )
    workflow.add_edge("tools", "agent")
    
    app = workflow.compile()

    # --- CHAT INTERFACE ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        elif isinstance(msg, AIMessage) and msg.content:
            st.chat_message("assistant").write(msg.content)

    user_input = st.chat_input("Pregunta algo sobre los documentos...")

    if user_input:
        st.chat_message("user").write(user_input)
        st.session_state.messages.append(HumanMessage(content=user_input))

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("🔍 *Consultando la biblioteca...*")
            
            inputs = {"messages": st.session_state.messages}
            
            # Usando el callback para debug en consola
            result = app.invoke(
                inputs,
                config={"callbacks": [ConsoleCallbackHandler()]}            
            )
            final_msg = result["messages"][-1].content
            placeholder.markdown(final_msg)
            
            st.session_state.messages = result["messages"]

else:
    st.info("👆 Sube uno o varios PDFs para comenzar.")