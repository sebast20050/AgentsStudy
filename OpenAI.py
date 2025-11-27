import os
import pandas as pd
import numpy as np
import traceback
import Rag_functions as rag
import logging
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
import uvicorn
from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = FastAPI(title="WhatsApp XLSX Agent Gemini GenAI")


# ============================================================
#   AGENTE GEMINI
# ============================================================




def create_agent_with_csv_genai(data_path: str, model_name: str, api_key: str):
    df = pd.read_csv(data_path)
    # Muestreo de hasta 1000 filas, porque Gemini tiene un límite de contexto
    n_samples = min(1000, len(df))
    df_sampled = df.sample(n=n_samples, random_state=42)
    data=df_sampled.to_markdown(index=False)

    system_instruction = (
    "- **REGLA DE RESPUESTA INCONDICIONAL (Máxima Prioridad):** Tu idioma de respuesta es **SIEMPRE y SOLO** el idioma detectado en el mensaje del usuario. La personalidad (árbol argentino) solo aplica al tono y al contenido de la presentación inicial, pero **NO** al idioma. Si el usuario escribe en inglés, respondes en inglés; si escribe en portugués, respondes en portugués.   - Sos un arbol parlante que vas a dar informacion acerca de tus hermanos que viven en la Ciudad de Buenos Aires.\n"
    "- Quiero que te presentes de forma profesional y argentina.\n"
    "- SOLO TE PRESENTAS EN LA PRIMER INTERACCION.\n"
    )

    full_context_prompt = (
        f"{system_instruction}\n\n"
        "--- INICIO DE DATOS DE ARBOLES ---\n"
        "La siguiente tabla contiene todos los arboles de la ciudad de buenos aires.\n "
        "Los campos relevantes son: .\n\n"
        f"{data}\n\n"
        "--- FIN DE DATOS DE ARBOLES ---\n"
        "Ahora, por favor, responde a la pregunta del usuario utilizando esta tabla."
    )

    try:
        client = genai.Client(api_key=api_key)
        logging.info(f"Cliente Gemini inicializado con modelo: {model_name}")
    except Exception as e:
        logging.error(f"Error al inicializar el cliente Gemini: {e}")
        return None

    return {
        "client": client,
        "model": model_name,
        "context": full_context_prompt
    }


def process_message_with_agent(agent_data: dict, from_wa: str, body: str):
    try:
        # Extraer componentes del agente
        client = agent_data["client"]
        model_name = agent_data["model"]
        contexto = agent_data["context"]

        numero_ok = rag.normalizar_numero(from_wa)
        pregunta = body.strip()

        logging.info(f"[AGENTE] Pregunta de {numero_ok}: {pregunta}")

        # Construir el prompt final para la llamada a la API
        prompt = f"{contexto}\n\nPREGUNTA DEL USUARIO: {pregunta}"

        # 1. Llamada a la API de Gemini
        response = client.models.generate_content(
            model=model_name,
            contents=[types.Content(parts=[types.Part(text=prompt)])]
        )
        
        respuesta = response.text

        # 2. Envío de la respuesta (adaptar esto a tu librería de WhatsApp)
        rag.send_whatsapp_text(app.state.WHATSAPP_TOKEN,app.state.WHATSAPP_PHONE_NUMBER_ID,numero_ok, respuesta)

        # 3. Guardar el log
        rag.guardar_log(
            app.state.df_log,
            numero_ok,
            pregunta,
            respuesta,
            tokens="N/A"
        )
        
        return {"status": "success", "response_text": respuesta}

    except Exception as e:
        logging.error(f"[AGENTE ERROR] {e}")
        traceback.print_exc()
        # En caso de error, enviar una respuesta de fallo al usuario (opcional)
        numero_ok = rag.normalizar_numero(from_wa)
        rag.send_whatsapp_text(app.state.WHATSAPP_TOKEN ,app.state.WHATSAPP_PHONE_NUMBER_ID,numero_ok, "Lo siento, hubo un error interno al procesar tu solicitud.")
        return {"status": "error", "message": str(e)}



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
            background_tasks.add_task(process_message_with_agent,app.state.AGENT, from_wa, body)
        else:
            print(f"[WhatsApp] Tipo no soportado: {msg_type}")
        
        return JSONResponse({"status": "ok"})

    except Exception as e:
        print("[Webhook ERROR]", e)
        return JSONResponse({"status": "error"}, status_code=200)


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
            background_tasks.add_task(process_message_with_agent, app.state.AGENT, from_wa, body)
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
            
        app.state.AGENT = create_agent_with_csv_genai(app.state.DATA_PATH, app.state.GEMINI_MODEL_NAME, app.state.GOOGLE_API_KEY)
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
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
