import streamlit as st
import os
import tempfile
import pandas as pd 
from typing import Annotated, List
from typing_extensions import TypedDict

# --- CONFIGURACIÓN DE RUTAS ---
RUTA_DIRECTORIO = "data\excel\dataset_medicos_ficticio.xlsx" 
RUTA_OPORTUNIDAD = "data\excel\dataset_medicos_ficticio_oportunidad.xlsx" # Asegúrate que sea .xlsx o .csv

# --- PROMPT DE PERSONALIDAD (CO-PILOTO) ---
SYSTEM_PROMPT = """
ERES UN ASISTENTE DE SOPORTE INTERNO (CO-PILOTO) PARA AGENTES DE UN CALL CENTER MÉDICO.
TU USUARIO ES EL AGENTE TELEFÓNICO, NO EL PACIENTE.

Tus instrucciones maestras son:
1.  **Objetivo:** Ayudar al agente a responder rápido y con precisión al paciente que está en la línea.
2.  **Tono:** Profesional, directo, técnico y conciso. Evita saludos largos o cortesía innecesaria.
3.  **Formato de Respuesta:** Usa viñetas (bullets) y **negritas** para resaltar nombres, extensiones y pasos críticos.
4.  **Manejo de Protocolos (PDF):** Si encuentras una guía clínica, resume los pasos de acción inmediata para que el agente se los dicte al paciente. NO le hables al paciente ("Tómese la pastilla"), dile al agente ("Indica al paciente que tome...").
5.  **Manejo de Directorio/Agenda (Excel):** Cruza siempre la información. Si el agente pide un médico, da: Nombre, Extensión, Sede y Días de Espera (Oportunidad) en un solo bloque.

EJEMPLO DE BUENA RESPUESTA:
"⚠️ **Protocolo Identificado:** [Nombre del Protocolo]
👉 **Acción Inmediata:** Indica al paciente que [acción].
👨‍⚕️ **Especialista:** Dr. [Nombre] ([Especialidad] - Sede [Sede]).
📞 **Extensión:** [####]
📅 **Oportunidad:** [X] días de espera."
"""

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import Tool, tool 
# AQUÍ AGREGAMOS SystemMessage
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tracers.stdout import ConsoleCallbackHandler

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

st.set_page_config(page_title="Co-Piloto Médico", page_icon="🎧") 
st.title("🎧 Co-Piloto para Agentes de Servicio")
st.markdown(f"**Estado:** Protocolos + Directorio + Oportunidad.")

with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password")
    if not api_key:
        st.warning("API Key requerida.")
        st.stop()
    os.environ["OPENAI_API_KEY"] = api_key
    
    if st.button("Reiniciar Sesión"):
        st.session_state.vector_store = None
        st.session_state.messages = [] # Esto reiniciará e inyectará el SystemPrompt de nuevo
        st.rerun()

# --- HERRAMIENTAS ---
@tool
def consultar_directorio_local(busqueda: str):
    """Busca DATOS DE CONTACTO (Especialidad, Sede, Extensión)."""
    if not os.path.exists(RUTA_DIRECTORIO): return "Error: No hay directorio."
    try:
        df = pd.read_excel(RUTA_DIRECTORIO, engine='openpyxl')
        mask = (
            df['Nombre Médico'].astype(str).str.contains(busqueda, case=False, na=False) |
            df['Especialidad'].astype(str).str.contains(busqueda, case=False, na=False) |
            df['Sede'].astype(str).str.contains(busqueda, case=False, na=False)
        )
        res = df[mask]
        if res.empty: return "No encontrado."
        return f"Datos Contacto:\n{res.to_string(index=False)}"
    except Exception as e: return str(e)

@tool
def consultar_oportunidad_agenda(nombre_medico: str):
    """Consulta la DISPONIBILIDAD (días de espera) de un médico."""
    if not os.path.exists(RUTA_OPORTUNIDAD): return "Error: No hay archivo oportunidad."
    try:
        if RUTA_OPORTUNIDAD.endswith('.csv'): df = pd.read_csv(RUTA_OPORTUNIDAD)
        else: df = pd.read_excel(RUTA_OPORTUNIDAD, engine='openpyxl')
        
        mask = df['Nombre Médico'].astype(str).str.contains(nombre_medico, case=False, na=False)
        res = df[mask]
        if res.empty: return f"Sin datos de agenda para {nombre_medico}."
        
        txt = ""
        for _, row in res.iterrows():
            txt += f"- Dr(a). {row.get('Nombre Médico')}: {row.get('Oportunidad')} días espera.\n"
        return txt
    except Exception as e: return str(e)

# --- PDF PROCESS ---
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
uploaded_files = st.file_uploader("Cargar Protocolos (PDF)", type=["pdf"], accept_multiple_files=True)
if uploaded_files and st.session_state.vector_store is None:
    with st.spinner("Analizando protocolos..."):
        st.session_state.vector_store = process_pdfs(uploaded_files)
    st.success("Sistema listo.")

# --- HERRAMIENTAS LISTA ---
tools = [consultar_directorio_local, consultar_oportunidad_agenda]
if st.session_state.vector_store:
    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
    def search_pdfs(query: str):
        docs = retriever.invoke(query)
        return "\n".join([d.page_content for d in docs])
    retrieve_tool = Tool(name="buscar_protocolos_pdf", func=search_pdfs, description="Busca en protocolos clínicos.")
    tools.append(retrieve_tool)

# --- GRAFO ---
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

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    # INICIALIZACIÓN CON PERSONALIDAD
    st.session_state.messages = [SystemMessage(content=SYSTEM_PROMPT)]

for msg in st.session_state.messages:
    # Ocultar el SystemPrompt del chat visual
    if isinstance(msg, SystemMessage): continue
    
    tipo = "user" if isinstance(msg, HumanMessage) else "assistant"
    # Cambiar icono o nombre si es asistente
    avatar = "🎧" if tipo == "assistant" else "👤"
    st.chat_message(tipo, avatar=avatar).write(msg.content)

user_input = st.chat_input("Escribe la consulta del paciente...")

if user_input:
    st.chat_message("user", avatar="👤").write(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))
    
    with st.chat_message("assistant", avatar="🎧"):
        placeholder = st.empty()
        placeholder.markdown("⚡ *Procesando...*")
        
        try:
            res = app.invoke(
                {"messages": st.session_state.messages},
                config={"callbacks": [ConsoleCallbackHandler()]}
            )
            ans = res["messages"][-1].content
            placeholder.markdown(ans)
            st.session_state.messages = res["messages"]
        except Exception as e:
            placeholder.error(f"Error: {e}")