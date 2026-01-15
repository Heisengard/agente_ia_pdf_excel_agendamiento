import streamlit as st
import os
import tempfile
import pandas as pd 
from typing import Annotated, List
from typing_extensions import TypedDict

# --- CONFIGURACIÓN INICIAL ---
# CAMBIA ESTO si tu archivo está en otra carpeta. 
# Si está junto al script, déjalo así.
RUTA_DIRECTORIO = "data/excel/dataset_medicos_ficticio.xlsx" 

# --- INICIALIZACIÓN DE SESIÓN GLOBAL ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Importaciones de LangChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import Tool, tool # <--- Importamos 'tool' decorador
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tracers.stdout import ConsoleCallbackHandler

# Importaciones de LangGraph
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Agente Médico Pro", page_icon="👨‍⚕️") 
st.title("👨‍⚕️ Agente: Protocolos (PDF) + Directorio Local")
st.markdown(f"**Modo:** PDFs en Memoria Vectorial + Directorio (`{RUTA_DIRECTORIO}`) en Disco Local.")

# --- GESTIÓN DE CLAVES ---
with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password")
    if not api_key:
        st.warning("Ingresa tu API Key para continuar.")
        st.stop()
    os.environ["OPENAI_API_KEY"] = api_key
    
    if st.button("Borrar Memoria PDFs"):
        st.session_state.vector_store = None
        st.session_state.messages = []
        st.rerun()

# --- 1. HERRAMIENTA DE DIRECTORIO LOCAL (La novedad) ---
@tool
def consultar_directorio_local(busqueda: str):
    """
    Consulta el directorio médico (Excel local) para encontrar especialistas, 
    sedes, teléfonos o extensiones.
    Input: Una palabra clave (ej: 'Cardiología', 'Norte', 'Juan').
    """
    if not os.path.exists(RUTA_DIRECTORIO):
        return f"Error: No encuentro el archivo '{RUTA_DIRECTORIO}' en el sistema."
    
    try:
        # Leemos el archivo en tiempo real
        df = pd.read_excel(RUTA_DIRECTORIO, engine='openpyxl')
        
        # Filtramos buscando la palabra clave en varias columnas a la vez
        # (Nombre, Especialidad o Sede)
        mask = (
            df['Nombre Médico'].astype(str).str.contains(busqueda, case=False, na=False) |
            df['Especialidad'].astype(str).str.contains(busqueda, case=False, na=False) |
            df['Sede'].astype(str).str.contains(busqueda, case=False, na=False)
        )
        
        resultados = df[mask]
        
        if resultados.empty:
            return f"No encontré coincidencias para '{busqueda}' en el directorio."
        
        # Devolvemos los datos en formato texto
        return f"Resultados del Directorio:\n{resultados.to_string(index=False)}"
        
    except Exception as e:
        return f"Error al leer el directorio: {str(e)}"

# --- 2. PROCESAMIENTO DE PDFs (Solo PDFs) ---
def process_pdfs(uploaded_files):
    """
    Procesa SOLO PDFs para crear la base de conocimiento de protocolos.
    """
    all_docs = [] 
    
    progress_text = "Leyendo manuales y protocolos..."
    my_bar = st.progress(0, text=progress_text)
    total_files = len(uploaded_files)
    
    for i, file in enumerate(uploaded_files):
        # Procesamos solo si es PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name
        
        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            for d in docs:
                d.metadata["source"] = file.name
            all_docs.extend(docs)
        except Exception as e:
            st.error(f"Error en {file.name}: {e}")
        finally:
            os.remove(tmp_path)
        
        my_bar.progress((i + 1) / total_files)

    if not all_docs:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(splits, embeddings)
    
    my_bar.empty()
    return vector_store

# --- INTERFAZ DE CARGA (Solo PDFs) ---
uploaded_files = st.file_uploader(
    "Sube tus MANUALES o PROTOCOLOS (PDF)", 
    type=["pdf"], 
    accept_multiple_files=True
)

if uploaded_files and st.session_state.vector_store is None:
    with st.spinner("⏳ Estudiando protocolos..."):
        st.session_state.vector_store = process_pdfs(uploaded_files)
    
    if st.session_state.vector_store:
        st.success(f"✅ Protocolos memorizados. Listo para consultas mixtas.")

# --- 3. DEFINICIÓN DEL AGENTE Y HERRAMIENTAS ---

# Herramienta 1: Búsqueda en PDFs (Si existen)
tools = [consultar_directorio_local] # Empezamos con la herramienta local

if st.session_state.vector_store:
    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5})
    
    def search_pdfs(query: str):
        docs = retriever.invoke(query)
        return "\n\n".join([f"[Fuente: {d.metadata.get('source')}]\n{d.page_content}" for d in docs])

    retrieve_tool = Tool(
        name="buscar_protocolos_pdf",
        func=search_pdfs,
        description="Usa esto para buscar información CLÍNICA, protocolos o guías en los PDFs subidos."
    )
    tools.append(retrieve_tool)

# --- CONFIGURACIÓN DEL CEREBRO (LangGraph) ---
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
workflow.add_conditional_edges("agent", tools_condition, {"tools": "tools", "__end__": END})
workflow.add_edge("tools", "agent")

app = workflow.compile()

# --- CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    tipo = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(tipo).write(msg.content)

user_input = st.chat_input("Pregunta algo (ej: 'Necesito un cardiólogo en Norte y el protocolo de dolor de pecho')")

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("🤔 *Pensando...*")
        
        # Invocamos al agente
        result = app.invoke(
            {"messages": st.session_state.messages},
            config={"callbacks": [ConsoleCallbackHandler()]}            
        )
        
        final_msg = result["messages"][-1].content
        placeholder.markdown(final_msg)
        st.session_state.messages = result["messages"]