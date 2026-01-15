import streamlit as st
import os
import tempfile
import pandas as pd # <--- NUEVO: Para leer el Excel
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
from langchain_core.documents import Document # <--- NUEVO: Para crear docs desde Excel

# Importaciones de LangGraph
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Cerebro Médico Híbrido", page_icon="🧬") 
st.title("🧬 Agente Híbrido: Protocolos + Directorio")
st.markdown("Sube **PDFs (Manuales)** y tu **Excel (Directorio)**. El agente relacionará ambos.")

# --- GESTIÓN DE CLAVES ---
with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password")
    if not api_key:
        st.warning("Ingresa tu API Key para continuar.")
        st.stop()
    os.environ["OPENAI_API_KEY"] = api_key
    
    if st.button("Borrar Memoria y Reiniciar"):
        st.session_state.vector_store = None
        st.session_state.messages = []
        st.rerun()

# --- FUNCIÓN DE PROCESAMIENTO HÍBRIDA ---
def process_files(uploaded_files):
    """
    Procesa tanto PDFs como EXCEL y los fusiona en un solo cerebro.
    """
    all_docs = [] # Aquí acumularemos todo: texto de PDFs y filas de Excel
    
    progress_text = "Procesando archivos y fusionando conocimientos..."
    my_bar = st.progress(0, text=progress_text)
    total_files = len(uploaded_files)
    
    for i, file in enumerate(uploaded_files):
        filename = file.name
        
        # --- CASO 1: Es un EXCEL (.xlsx) ---
        if filename.endswith(('.xlsx', '.xls')):
            try:
                df = pd.read_excel(file)
                # Iteramos por cada médico para convertirlo en texto comprensible
                for index, row in df.iterrows():
                    # Creamos una "historia" para cada fila
                    texto_medico = (
                        f"DIRECTORIO MÉDICO - DETALLE DEL ESPECIALISTA:\n"
                        f"Nombre: {row.get('Nombre Médico', 'N/A')}\n"
                        f"Especialidad: {row.get('Especialidad', 'N/A')}\n"
                        f"Sede: {row.get('Sede', 'N/A')}\n"
                        f"Extensión telefónica: {row.get('Numero de extensión', 'N/A')}"
                    )
                    
                    # Creamos el documento LangChain
                    doc = Document(
                        page_content=texto_medico,
                        metadata={"source": filename, "type": "directorio_excel", "row": index}
                    )
                    all_docs.append(doc)
            except Exception as e:
                st.error(f"Error procesando Excel {filename}: {e}")

        # --- CASO 2: Es un PDF ---
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file.read())
                tmp_path = tmp_file.name
            
            try:
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                # Añadimos metadatos extra para identificar que viene de PDF
                for d in docs:
                    d.metadata["type"] = "manual_pdf"
                    d.metadata["source"] = filename
                
                all_docs.extend(docs)
            except Exception as e:
                st.error(f"Error procesando PDF {filename}: {e}")
            finally:
                os.remove(tmp_path)
        
        # Actualizar barra
        my_bar.progress((i + 1) / total_files, text=f"Integrando archivo {i+1} de {total_files}...")

    # 3. Dividir texto (Importante para los PDFs, irrelevante pero inocuo para el Excel)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)

    # 4. Crear Vector Store ÚNICO
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(splits, embeddings)
    
    my_bar.empty()
    return vector_store

# --- INTERFAZ DE CARGA ---
# Aceptamos PDF y Excel
uploaded_files = st.file_uploader(
    "Sube tus documentos (PDFs y Excel)", 
    type=["pdf", "xlsx", "xls"], 
    accept_multiple_files=True
)

if uploaded_files and st.session_state.vector_store is None:
    with st.spinner("⏳ Fusionando el directorio médico con los protocolos..."):
        st.session_state.vector_store = process_files(uploaded_files)
    st.success(f"✅ Cerebro actualizado con {len(uploaded_files)} archivos.")

# --- DEFINICIÓN DE HERRAMIENTAS RAG ---
if st.session_state.vector_store:
    
    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 6}) 
    
    def search_function(query: str):
        docs = retriever.invoke(query)
        results = []
        for d in docs:
            # Mostramos si la info viene del Excel o del PDF
            tipo = d.metadata.get('type', 'desconocido')
            source = d.metadata.get('source', 'Desconocido')
            content = d.page_content
            results.append(f"[Fuente: {source} ({tipo})]\n{content}")
            
        return "\n\n".join(results)

    retrieve_tool = Tool(
        name="knowledge_search",
        func=search_function,
        description="Usa esto para buscar información tanto en protocolos (PDF) como en el directorio de médicos (Excel)."
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

    user_input = st.chat_input("Ej: 'Tengo un paciente con dolor de pecho, ¿qué protocolo sigo y a quién llamo en la sede Norte?'")

    if user_input:
        st.chat_message("user").write(user_input)
        st.session_state.messages.append(HumanMessage(content=user_input))

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("🔍 *Consultando cerebro híbrido...*")
            
            inputs = {"messages": st.session_state.messages}
            
            result = app.invoke(
                inputs,
                config={"callbacks": [ConsoleCallbackHandler()]}            
            )
            final_msg = result["messages"][-1].content
            placeholder.markdown(final_msg)
            
            st.session_state.messages = result["messages"]

else:
    st.info("👆 Por favor sube el archivo Excel generado y algún PDF de prueba.")