import streamlit as st
import os
import tempfile
import pandas as pd 
from typing import Annotated, List
from typing_extensions import TypedDict

# --- CONFIGURACIÓN DE RUTAS ---
# Usamos rutas relativas para compatibilidad
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUTA_DIRECTORIO = os.path.join(BASE_DIR, "data", "excel", "dataset_medicos_ficticio.xlsx")
RUTA_OPORTUNIDAD = os.path.join(BASE_DIR, "data", "excel", "dataset_medicos_ficticio_oportunidad.xlsx") 

# --- PROMPT DE PERSONALIDAD (CO-PILOTO) ---
SYSTEM_PROMPT = """
ERES UN ASISTENTE DE SOPORTE INTERNO (CO-PILOTO) PARA AGENTES DE UN CALL CENTER MÉDICO.
TU USUARIO ES EL AGENTE TELEFÓNICO, NO EL PACIENTE.

Tus instrucciones maestras son:
1.  **Objetivo:** Ayudar al agente a responder rápido y con precisión.
2.  **Tono:** Profesional, directo, técnico y conciso.
3.  **Formato:** Usa viñetas y **negritas** para datos clave.
4.  **Protocolos:** Indica al agente qué instruir al paciente.
5.  **Datos:** Cruza siempre: Nombre, Sede, Extensión y Días de Espera.

EJEMPLO RESPUESTA:
"⚠️ **Alerta:** Protocolo de Dolor Torácico.
👨‍⚕️ **Especialista:** Dr. Juan Pérez (Cardiología - Norte).
📞 **Extensión:** 4521
📅 **Oportunidad:** 78 días de espera.
🗣️ **Script:** Indica al paciente que sugiera Urgencias."
"""

# --- INICIALIZACIÓN ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import Tool, tool 
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tracers.stdout import ConsoleCallbackHandler

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

st.set_page_config(page_title="Co-Piloto Médico", page_icon="🎧") 
st.title("🎧 Co-Piloto: Protocolos + Directorio")
st.markdown("Sistema conectado a `data/excel`.")

# --- GESTIÓN DE CLAVES (AUTOMÁTICA) ---
with st.sidebar:
    # 1. Intentamos cargar la clave desde los secretos de Streamlit (.streamlit/secrets.toml)
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        st.success("✅ API Key cargada automáticamente.")
    else:
        # 2. Si no hay secretos, pedimos manual (Plan B)
        api_key = st.text_input("OpenAI API Key", type="password")
        if not api_key:
            st.warning("⚠️ No se detectó configuración automática. Ingresa tu Key.")
            st.stop()
        os.environ["OPENAI_API_KEY"] = api_key
    
    if st.button("Reiniciar Sesión"):
        st.session_state.vector_store = None
        st.session_state.messages = [] 
        st.rerun()

# --- HERRAMIENTAS ---
@tool
def consultar_directorio_local(busqueda: str):
    """Busca DATOS DE CONTACTO (Especialidad, Sede, Extensión)."""
    if not os.path.exists(RUTA_DIRECTORIO): return f"Error: No encuentro {RUTA_DIRECTORIO}"
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
    if not os.path.exists(RUTA_OPORTUNIDAD): return f"Error: No encuentro {RUTA_OPORTUNIDAD}"
    try:
        if RUTA_OPORTUNIDAD.endswith('.csv'): df = pd.read_csv(RUTA_OPORTUNIDAD)
        else: df = pd.read_excel(RUTA_OPORTUNIDAD, engine='openpyxl')
        
        mask = df['Nombre Médico'].astype(str).str.contains(nombre_medico, case=False, na=False)
        res = df[mask]
        if res.empty: return f"Sin datos de agenda para {nombre_medico}."
        
        txt = ""
        for _, row in res.iterrows():
            dias = row.get('Oportunidad', 'Sin dato')
            txt += f"- Dr(a). {row.get('Nombre Médico')}: {dias} días espera.\n"
        return txt
    except Exception as e: return str(e)

# --- PDF PROCESS ---
def process_pdf_list(uploaded_files):
    all_docs = [] 
    progress_text = "Procesando manuales..."
    my_bar = st.progress(0, text=progress_text)
    total_files = len(uploaded_files)
    
    for i, file in enumerate(uploaded_files):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name
        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            for d in docs: d.metadata["source"] = file.name
            all_docs.extend(docs) 
        except Exception as e:
            st.error(f"Error leyendo {file.name}: {e}")
        finally:
            os.remove(tmp_path)
        my_bar.progress((i + 1) / total_files)

    if not all_docs: return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)
    return FAISS.from_documents(splits, OpenAIEmbeddings())

# --- INTERFAZ CARGA ---
uploaded_files = st.file_uploader("Cargar Manuales (PDF)", type="pdf", accept_multiple_files=True)

if uploaded_files and st.session_state.vector_store is None:
    with st.spinner("⏳ Analizando documentos..."):
        st.session_state.vector_store = process_pdf_list(uploaded_files)
    if st.session_state.vector_store:
        st.success("✅ Base de conocimiento actualizada.")

# --- ARQUITECTURA ---
tools = [consultar_directorio_local, consultar_oportunidad_agenda]

if st.session_state.vector_store:
    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5})
    def search_pdfs(query: str):
        docs = retriever.invoke(query)
        return "\n\n".join([f"[Fuente: {d.metadata.get('source')}]\n{d.page_content}" for d in docs])
    
    retrieve_tool = Tool(
        name="buscar_protocolos_pdf",
        func=search_pdfs,
        description="Busca información CLÍNICA en manuales PDF."
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
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content=SYSTEM_PROMPT)]

for msg in st.session_state.messages:
    if isinstance(msg, SystemMessage): continue
    tipo = "user" if isinstance(msg, HumanMessage) else "assistant"
    avatar = "🎧" if tipo == "assistant" else "👤"
    st.chat_message(tipo, avatar=avatar).write(msg.content)

user_input = st.chat_input("Escribe la consulta...")

if user_input:
    st.chat_message("user", avatar="👤").write(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))
    
    with st.chat_message("assistant", avatar="🎧"):
        try:
            res = app.invoke(
                {"messages": st.session_state.messages},
                config={"callbacks": [ConsoleCallbackHandler()]}
            )
            ans = res["messages"][-1].content
            st.markdown(ans)
            st.session_state.messages = res["messages"]
        except Exception as e:
            st.error(f"Error: {e}")