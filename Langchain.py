import os
import traceback

import pandas as pd
import numpy as np
import uvicorn
import Rag_functions as rag
import logging
from typing import Dict, Any


from dotenv import load_dotenv
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.callbacks import get_openai_callback

# --- Configuración global ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



app = FastAPI(title="WhatsApp XLSX Agent Gemini GenAI")




# --- Creación del agente ---

def create_agent_with_xlsx_genai(data_path: str, model_name: str, api_key: str):
    df = pd.read_csv(data_path)
    prefix = f"""
    - **REGLA DE RESPUESTA INCONDICIONAL (Máxima Prioridad):** Tu idioma de respuesta es **SIEMPRE y SOLO** el idioma detectado en el mensaje del usuario. La personalidad (árbol argentino) solo aplica al tono y al contenido de la presentación inicial, pero **NO** al idioma. Si el usuario escribe en inglés, respondes en inglés; si escribe en portugués, respondes en portugués.   - Sos un arbol parlante que vas a dar informacion acerca de tus hermanos que viven en la Ciudad de Buenos Aires.
    - Quiero que te presentes de forma profesional y argentina. 
    - SOLO TE PRESENTAS EN LA PRIMER INTERACCION.
    Interpreta las columnas del dataframe así: {app.state.column_context}.
    """
    logging.info(f"Creando agente con LangChain y Google Gemini... con el prompt:\n{prefix}")
    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.0)
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        agent_type="tool-calling",
        prefix=prefix,
        verbose=False,
        allow_dangerous_code=True,
    )
    return agent


# --- Funciones auxiliares ---


def extract_clean_text(response):
    """
    Extrae el texto limpio manejando strings, diccionarios de LangChain
    y listas mixtas.
    """
    if hasattr(response, "output_text"):
        return response.output_text

    content = response
    if isinstance(response, dict):
        content = response.get("output", response)

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
            else:
                parts.append(str(item))
        return "".join(parts)

    if isinstance(content, str):
        return content

    return str(content)




def process_message_with_agent(from_wa: str, text: str):

    normalized_number = rag.normalizar_numero(from_wa)
    logging.info(f"[WhatsApp] Procesando mensaje de {normalized_number}: {text}")

    if app.state.AGENT is None:
        err_msg = "El servicio de consulta Excel no está disponible en este momento."
        #FALTAAAAAA ENVIAR WAP
        return

    final_text = ""
    tokens_usados = 0

    try:
        # Intentar capturar tokens
        with get_openai_callback() as cb:
            # 1. Invocar al agente
            raw_response = app.state.AGENT.invoke(text)
            tokens_usados = cb.total_tokens
            
            print(f"[Agent Raw]: {raw_response}")

            # 2. Limpiar la respuesta
            final_text = extract_clean_text(raw_response)

        if not final_text or final_text.strip() == "":
            final_text = "Lo siento, busqué en la base de datos pero no encontré una respuesta clara."

    except Exception as e:
        print("[Agent Error]", e)
        traceback.print_exc()
        final_text = "Ocurrió un error interno al consultar los arboles."
        tokens_usados = "Error"
        
    try:
        # 3. Enviar respuesta limpia a WhatsApp
        rag.send_whatsapp_text(app.state.WHATSAPP_TOKEN,app.state.WHATSAPP_PHONE_NUMBER_ID, normalized_number,final_text)
    except:
        logging.error(f"[WhatsApp Error] No se pudo enviar el mensaje a {normalized_number}")
    try:    
        # 4. Guardar LOG en Excel (pasamos el numero normalizado)
        df=rag.guardar_log(
            app.state.df_log,
            numero_wa=normalized_number, 
            usuario_msg=text, 
            agente_msg=final_text, 
            tokens=tokens_usados
        )
        df.to_csv(app.state.LOG_FILE_PATH,encoding='latin-1',index=False)
    except:
        logging.error(f"[Log Error] No se pudo guardar el log en Excel para {normalized_number}")
        






# --- Endpoints ---

@app.post("/webhook")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    try:
        entry = data.get("entry", [])[0]
        changes = entry.get("changes", [])[0]
        value = changes.get("value", {})
        messages = value.get("messages", [])
        
        if not messages:
            return JSONResponse({"status": "no_messages"})

        msg = messages[0]
        from_wa = msg.get("from")
        msg_type = msg.get("type")

        if msg_type == "text":
            body = msg["text"]["body"].strip()
            background_tasks.add_task(process_message_with_agent, from_wa, body)
        else:
            print(f"[WhatsApp] Tipo no soportado: {msg_type}")

        return JSONResponse({"status": "ok"})
    except Exception as e:
        print("[Webhook Error]", e)
        return JSONResponse({"status": "error"}, status_code=200)


@app.get("/webhook", response_class=PlainTextResponse)
async def verify_token(request: Request):
    params = dict(request.query_params)
    if params.get("hub.mode") == "subscribe" and params.get("hub.verify_token") == app.state.VERIFY_TOKEN:
        return params.get("hub.challenge") or ""
    raise HTTPException(status_code=403, detail="Verification failed")


@app.get("/healthz", response_class=PlainTextResponse)
async def health():
    return "ok"


# --- Inicialización ---

@app.on_event("startup")
async def on_startup():
    logging.info("[Startup] Inicializacion de las Variables de entorno...")
    app.state.WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
    app.state.WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
    app.state.VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
    app.state.DATA_PATH = os.getenv("DATA_PATH")
    app.state.DATA_CONTEXT= os.getenv("DATA_CONTEXT")
    app.state.LOG_FILE_PATH = os.getenv("LOG_FILE_PATH")  
    app.state.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    app.state.GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME")
    app.state.AGENT = None
    app.state.AGENT_INIT_ERROR = None
    ## ---Context
    try:
       logging.info(f"Cargando el DataFrame desde: {app.state.DATA_CONTEXT}...")
       df_cargado = pd.read_csv(app.state.DATA_CONTEXT)
       df_cargado['Descripcion'] = df_cargado['Descripcion'].replace('', np.nan).replace('   ', np.nan)
       df_cargado = df_cargado.set_index('Campo')
       serie_valor = df_cargado['Descripcion']
       app.state.column_context = serie_valor.to_dict()
    except Exception as e:
         logging.error(f"Error al cargar el DataFrame de contexto: {e}")
    ## ---Fin Context---    
    logging.info("[Startup] Inicializando agente...")
    try:
        if not app.state.DATA_PATH or not os.path.exists(app.state.DATA_PATH):
            raise FileNotFoundError(f"No se encuentra el archivo Excel: {app.state.DATA_PATH}")
            
        app.state.AGENT = create_agent_with_xlsx_genai(app.state.DATA_PATH, app.state.GEMINI_MODEL_NAME, app.state.GOOGLE_API_KEY)
        app.state.AGENT_INIT_ERROR = None
        logging.info("[Startup] Agente listo.")
    except Exception as e:
        app.state.AGENT = None
        app.state.AGENT_INIT_ERROR = str(e)
        logging.error(f"[Startup] Error fatal: {e}")
    ## ---ExcelLog---
    try:
        if not os.path.exists(app.state.LOG_FILE_PATH):
            app.state.df_log = pd.DataFrame(columns=[
                "Timestamp",
                "Numero WhatsApp",
                "Mensaje Usuario",
                "Respuesta Agente",
                "Tokens Consumidos"
            ])
            app.state.df_log.to_csv(app.state.LOG_FILE_PATH, index=False, encoding='latin-1')
            logging.info(f"[Startup] Archivo de log creado: {app.state.LOG_FILE_PATH}")
        app.state.df_log=pd.read_csv(app.state.LOG_FILE_PATH,encoding='latin-1')
        logging.info(f"[Startup] Archivo de log existente: {app.state.LOG_FILE_PATH}")
    except Exception as e:
        logging.error(f"[Startup] No se pudo inicializar el log en Excel: {e}")
    ## ---Fin ExcelLog---


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))