import streamlit as st
import os
import tempfile
import pandas as pd 
from typing import Annotated, List
from typing_extensions import TypedDict

# --- CONFIGURACIÓN DE RUTAS ---
# Asegúrate de que estos archivos existan en la misma carpeta que este script
RUTA_DIRECTORIO = "data\excel\dataset_medicos_ficticio.xlsx" 
RUTA_OPORTUNIDAD = "dataset_medicos_ficticio_oportunidad.xlsx" # Asegúrate que sea .xlsx o cambia a .csv si es necesario

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import Tool, tool 
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tracers.stdout import ConsoleCallbackHandler

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

st.set_page_config(page_title="Agente Multi-Base", page_icon="🏥") 
st.title("🏥 Agente: PDFs + Directorio + Oportunidad")
st.markdown(f"Consulta **Protocolos**, busca en **Directorio** y revisa **Oportunidad** (Días de espera).")

with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password")
    if not api_key:
        st.warning("API Key requerida.")
        st.stop()
    os.environ["OPENAI_API_KEY"] = api_key
    
    if st.button("Reiniciar Cerebro"):
        st.session_state.vector_store = None
        st.session_state.messages = []
        st.rerun()

# --- HERRAMIENTA 1: DIRECTORIO ---
@tool
def consultar_directorio_local(busqueda: str):
    """
    Usa esto para buscar DATOS DE CONTACTO (Especialidad, Sede, Extensión).
    Input: Nombre, especialidad o sede.
    """
    if not os.path.exists(RUTA_DIRECTORIO):
        return "Error: Archivo de directorio no encontrado."
    try:
        df = pd.read_excel(RUTA_DIRECTORIO, engine='openpyxl')
        mask = (
            df['Nombre Médico'].astype(str).str.contains(busqueda, case=False, na=False) |
            df['Especialidad'].astype(str).str.contains(busqueda, case=False, na=False) |
            df['Sede'].astype(str).str.contains(busqueda, case=False, na=False)
        )
        res = df[mask]
        if res.empty: return "No encontrado en directorio."
        return f"Datos de Contacto:\n{res.to_string(index=False)}"
    except Exception as e: return str(e)

# --- HERRAMIENTA 2: OPORTUNIDAD (CORREGIDA) ---
@tool
def consultar_oportunidad_agenda(nombre_medico: str):
    """
    Usa esto SOLO para saber la disponibilidad o días de espera de un médico.
    Input: Nombre del médico exacto o parcial.
    """
    if not os.path.exists(RUTA_OPORTUNIDAD):
        return f"Error: Archivo {RUTA_OPORTUNIDAD} no encontrado."
    
    try:
        # Soporte para CSV o Excel según tu archivo local
        if RUTA_OPORTUNIDAD.endswith('.csv'):
            df = pd.read_csv(RUTA_OPORTUNIDAD)
        else:
            df = pd.read_excel(RUTA_OPORTUNIDAD, engine='openpyxl')
        
        # Filtramos por nombre
        mask = df['Nombre Médico'].astype(str).str.contains(nombre_medico, case=False, na=False)
        res = df[mask]
        
        if res.empty: 
            return f"No tengo datos de oportunidad para '{nombre_medico}'."
        
        # Construimos respuesta interpretada (Días de espera)
        respuesta_texto = ""
        for index, row in res.iterrows():
            dias = row.get('Oportunidad', 'Sin dato')
            nombre = row.get('Nombre Médico', 'Desconocido')
            respuesta_texto += f"- Dr(a). {nombre}: Oportunidad de cita en {dias} días.\n"
            
        return f"Datos de Oportunidad encontrados:\n{respuesta_texto}"

    except Exception as e: return f"Error leyendo archivo oportunidad: {str(e)}"

# --- PROCESAMIENTO PDFS ---
def process_pdfs(uploaded_files):
    all_docs = [] 
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            path = tmp.name
        try:
            loader = PyPDFLoader(path)
            docs = loader.load()
            for d in docs: d.metadata["source"] = file.name
            all_docs.extend(docs)
        except: pass
        finally: os.remove(path)
    
    if not all_docs: return None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)
    return FAISS.from_documents(splits, OpenAIEmbeddings())

# --- INTERFAZ CARGA ---
uploaded_files = st.file_uploader("Sube Protocolos (PDF)", type=["pdf"], accept_multiple_files=True)

if uploaded_files and st.session_state.vector_store is None:
    with st.spinner("Leyendo PDFs..."):
        st.session_state.vector_store = process_pdfs(uploaded_files)
    st.success("PDFs listos.")

# --- DEFINICIÓN DE HERRAMIENTAS ---
tools = [consultar_directorio_local, consultar_oportunidad_agenda]

if st.session_state.vector_store:
    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
    def search_pdfs(query: str):
        docs = retriever.invoke(query)
        return "\n".join([d.page_content for d in docs])
    
    retrieve_tool = Tool(
        name="buscar_protocolos_pdf",
        func=search_pdfs,
        description="Busca información clínica en manuales PDF."
    )
    tools.append(retrieve_tool)

# --- LANGGRAPH ---
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
if "messages" not in st.session_state: st.session_state.messages = []

for msg in st.session_state.messages:
    tipo = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(tipo).write(msg.content)

user_input = st.chat_input("Ej: ¿Quién es el cardiólogo de la sede Sur y cuántos días de espera tiene?")

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("🔍 *Consultando bases de datos...*")
        
        try:
            res = app.invoke(
                {"messages": st.session_state.messages},
                config={"callbacks": [ConsoleCallbackHandler()]}
            )
            ans = res["messages"][-1].content
            placeholder.markdown(ans)
            st.session_state.messages = res["messages"]
        except Exception as e:
            placeholder.error(f"Error en el agente: {e}")