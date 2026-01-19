# 🏥 Medical AI Co-Pilot: Agente de Soporte Operativo

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green?logo=langchain&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?logo=openai&logoColor=white)

> **Un Asistente Inteligente (RAG + Tools) diseñado para potenciar a los agentes de Call Center Médico.**

## 💡 Sobre el Proyecto

Este proyecto implementa un **Agente de IA Híbrido** diseñado para actuar como "Co-piloto" en tiempo real para operadores de salud. A diferencia de un chatbot para pacientes, este sistema está optimizado para la eficiencia operativa, cruzando información clínica con datos administrativos en segundos.

El sistema resuelve el problema de la **fragmentación de información**:
1.  Lee **Protocolos Clínicos** desde manuales PDF (Memoria Vectorial).
2.  Consulta **Directorios Médicos** desde Excel (Base de Datos Local).
3.  Verifica **Disponibilidad de Agenda** (Días de espera) en tiempo real.

---

## ✨ Funcionalidades Clave

* **🧠 Cerebro Híbrido (RAG + Structured Data):** Combina la búsqueda semántica en documentos no estructurados (PDFs) con la precisión de bases de datos estructuradas (Excel).
* **⚡ Lectura de "Oportunidad":** Capacidad única de interpretar columnas numéricas en Excel para informar sobre tiempos de espera (ej: "78 días para Cardiología").
* **🎧 Personalidad "Co-Piloto":** Prompt del sistema (System Prompt) ajustado para respuestas técnicas, directas y formateadas para lectura rápida (viñetas, negritas, scripts de guion).
* **🛡️ Gestión Segura de Credenciales:** Implementación de `secrets.toml` para manejo seguro de API Keys en entornos locales y de producción.

---

## 🛠️ Arquitectura Técnica

El proyecto utiliza **LangGraph** para orquestar el flujo de decisiones del agente:

1.  **Input:** Consulta del agente humano (ej: *"Paciente con dolor torácico, necesito cardiólogo"*).
2.  **Routing/Reasoning:** El modelo decide si necesita buscar en los PDFs (síntomas/protocolos) o en los Excels (médicos/agenda).
3.  **Tools Execution:**
    * `consultar_directorio_local`: Busca especialidades y sedes en `.xlsx`.
    * `consultar_oportunidad_agenda`: Verifica días de espera en `.xlsx`.
    * `buscar_protocolos_pdf`: RAG sobre documentos vectorizados con FAISS.
4.  **Output:** Respuesta consolidada con formato de soporte operativo.

---

## 🚀 Instalación y Uso

Sigue estos pasos para ejecutar el proyecto en tu entorno local.

### 1. Clonar el repositorio
```bash
git clone [https://github.com/TU_USUARIO/nombre-repo.git](https://github.com/TU_USUARIO/nombre-repo.git)
cd nombre-repo
