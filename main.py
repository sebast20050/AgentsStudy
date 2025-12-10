import os
import pandas as pd
import numpy as np
import traceback
import Rag_functions as rag
import logging
import json
from datetime import datetime
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title="WhatsApp XLSX Agent Gemini GenAI + Human Interface")

# --- Configuración CORS y Templates ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración de templates para servir el frontend
templates = Jinja2Templates(directory="templates")

# ============================================================
#   AGENTE GEMINI
# ============================================================

def create_agent_with_csv_genai(data_path: str, model_name: str, api_key: str):
    try:
        if not os.path.exists(data_path):
            logging.error(f"El archivo de datos no existe: {data_path}")
            return None

        df = pd.read_csv(data_path)
        # Muestreo de hasta 1000 filas
        n_samples = min(1000, len(df))
        df_sampled = df.sample(n=n_samples, random_state=42)
        data = df_sampled.to_markdown(index=False)

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

        client = genai.Client(api_key=api_key)
        logging.info(f"Cliente Gemini inicializado con modelo: {model_name}")

        return {
            "client": client,
            "model": model_name,
            "context": full_context_prompt
        }
    except Exception as e:
        logging.error(f"Error al crear el agente: {e}")
        return None


def process_message_with_agent(agent_data: dict, from_wa: str, body: str):
    # CORRECCIÓN: Verificar si el agente existe antes de intentar usarlo
    if agent_data is None:
        logging.error("[AGENTE ERROR] El agente no está inicializado (agent_data is None). Revisa logs de inicio.")
        # Opcional: Avisar al usuario que hay un error técnico
        # numero_ok = rag.normalizar_numero(from_wa)
        # rag.send_whatsapp_text(app.state.WHATSAPP_TOKEN, app.state.WHATSAPP_PHONE_NUMBER_ID, numero_ok, "⚠️ Error técnico: El agente no está disponible.")
        return {"status": "error", "message": "Agente no inicializado"}

    try:
        client = agent_data.get("client")
        model_name = agent_data.get("model")
        contexto = agent_data.get("context")

        if not client:
             logging.error("[AGENTE ERROR] Cliente de Gemini no encontrado en agent_data.")
             return

        numero_ok = rag.normalizar_numero(from_wa)
        
        # VALIDACIÓN DOBLE: Si por alguna razón entró aquí y está pausado, abortar.
        if hasattr(app.state, 'PAUSED_USERS') and numero_ok in app.state.PAUSED_USERS:
            logging.info(f"[AGENTE OMITIDO] El usuario {numero_ok} está pausado. No se responde.")
            return

        pregunta = body.strip()
        logging.info(f"[AGENTE] Pregunta de {numero_ok}: {pregunta}")

        # Construir el prompt final
        prompt = f"{contexto}\n\nPREGUNTA DEL USUARIO: {pregunta}"

        # 1. Llamada a la API de Gemini
        response = client.models.generate_content(
            model=model_name,
            contents=[types.Content(parts=[types.Part(text=prompt)])]
        )
        
        respuesta = response.text

        # 2. Envío de la respuesta
        rag.send_whatsapp_text(app.state.WHATSAPP_TOKEN, app.state.WHATSAPP_PHONE_NUMBER_ID, numero_ok, respuesta)

        # 3. Guardar el log y actualizar estado
        app.state.df_log = rag.guardar_log(
            app.state.df_log,
            numero_ok,
            pregunta,
            respuesta,
            tokens="N/A"
        )
        
        # Guardar cambios en disco inmediatamente
        if app.state.LOG_FILE_PATH:
            app.state.df_log.to_csv(app.state.LOG_FILE_PATH, index=False, encoding='latin-1')
        
        return {"status": "success", "response_text": respuesta}

    except Exception as e:
        logging.error(f"[AGENTE ERROR] {e}")
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

# ============================================================
#   ENDPOINTS FRONTEND
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Renderiza el Frontend"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/contacts")
async def get_contacts():
    """Devuelve la lista de números únicos que han interactuado"""
    # CORRECCIÓN: Verificar atributo antes de acceder
    if not hasattr(app.state, 'df_log') or app.state.df_log is None or app.state.df_log.empty:
        return []
    
    try:
        unique_numbers = app.state.df_log['Numero WhatsApp'].unique().tolist()
        contacts = []
        for num in unique_numbers:
            status = "PAUSED" if str(num) in app.state.PAUSED_USERS else "ACTIVE"
            contacts.append({"phone": str(num), "status": status})
        return contacts
    except Exception as e:
        logging.error(f"Error en /api/contacts: {e}")
        return []

@app.get("/api/history/{phone}")
async def get_history(phone: str):
    """Devuelve el historial de chat para un número"""
    if not hasattr(app.state, 'df_log') or app.state.df_log is None:
        return []

    try:
        df = app.state.df_log
        # Convertir a string para asegurar match
        df['Numero WhatsApp'] = df['Numero WhatsApp'].astype(str)
        chat_df = df[df['Numero WhatsApp'] == str(phone)].copy()
        
        chat_df = chat_df.fillna("")
        
        history = []
        for _, row in chat_df.iterrows():
            if row['Mensaje Usuario']:
                history.append({
                    "sender": "user",
                    "text": str(row['Mensaje Usuario']),
                    "time": str(row['Timestamp'])
                })
            if row['Respuesta Agente']:
                history.append({
                    "sender": "agent",
                    "text": str(row['Respuesta Agente']),
                    "time": str(row['Timestamp'])
                })
        return history
    except Exception as e:
        logging.error(f"Error fetching history: {e}")
        return []

@app.post("/api/send_manual")
async def send_manual_message(request: Request):
    """Envía mensaje manual, PAUSA al agente y loguea la interacción"""
    data = await request.json()
    phone = data.get("phone")
    message = data.get("message")
    
    if not phone or not message:
        return JSONResponse({"status": "error", "message": "Faltan datos"})

    try:
        rag.send_whatsapp_text(app.state.WHATSAPP_TOKEN, app.state.WHATSAPP_PHONE_NUMBER_ID, phone, message)
        
        app.state.PAUSED_USERS.add(phone)
        
        fecha_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        nuevo_registro = {
            "Timestamp": [fecha_hora],
            "Numero WhatsApp": [phone],
            "Mensaje Usuario": [np.nan], 
            "Respuesta Agente": [f"[MANUAL] {message}"], 
            "Tokens Consumidos": ["MANUAL"]
        }
        df_nuevo = pd.DataFrame(nuevo_registro)
        app.state.df_log = pd.concat([app.state.df_log, df_nuevo], ignore_index=True)
        
        if app.state.LOG_FILE_PATH:
            app.state.df_log.to_csv(app.state.LOG_FILE_PATH, index=False, encoding='latin-1')

        return {"status": "success", "agent_status": "PAUSED"}
    
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})

@app.post("/api/restore_agent")
async def restore_agent(request: Request):
    """Reactiva al agente para un usuario"""
    data = await request.json()
    phone = data.get("phone")
    
    if phone in app.state.PAUSED_USERS:
        app.state.PAUSED_USERS.remove(phone)
        return {"status": "success", "agent_status": "ACTIVE"}
    
    return {"status": "no_change", "agent_status": "ACTIVE"}

# ============================================================
#   WEBHOOK
# ============================================================

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
        from_wa_raw = msg.get("from")
        msg_type = msg.get("type")

        if msg_type == "text":
            body = msg["text"]["body"].strip()
            numero_ok = rag.normalizar_numero(from_wa_raw)

            # Verifica si el agente funciona antes de procesar nada
            if app.state.AGENT is None:
                logging.warning(f"Mensaje recibido de {numero_ok}, pero el AGENTE no está inicializado.")
                # Aquí podrías decidir si loguearlo igual o no.

            # --- LÓGICA DE INTERRUPCIÓN ---
            if numero_ok in app.state.PAUSED_USERS:
                logging.info(f"[WEBHOOK] Usuario {numero_ok} está PAUSADO. Solo logueamos entrada.")
                
                fecha_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                nuevo_registro = {
                    "Timestamp": [fecha_hora],
                    "Numero WhatsApp": [numero_ok],
                    "Mensaje Usuario": [body],
                    "Respuesta Agente": [np.nan], 
                    "Tokens Consumidos": ["SILENCED"]
                }
                df_temp = pd.DataFrame(nuevo_registro)
                app.state.df_log = pd.concat([app.state.df_log, df_temp], ignore_index=True)
                
                if app.state.LOG_FILE_PATH:
                    app.state.df_log.to_csv(app.state.LOG_FILE_PATH, index=False, encoding='latin-1')
            else:
                # Flujo normal: Agente responde
                background_tasks.add_task(process_message_with_agent, app.state.AGENT, from_wa_raw, body)
        else:
            print(f"[WhatsApp] Tipo no soportado: {msg_type}")
        
        return JSONResponse({"status": "ok"})

    except Exception as e:
        print("[Webhook ERROR]", e)
        traceback.print_exc()
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

# ============================================================
#   INICIALIZACION (STARTUP)
# ============================================================

@app.on_event("startup")
async def on_startup():
    logging.info("[Startup] Iniciando servidor...")
    
    # 1. Definir variables por defecto (Evita AttributeError 'State' object has no attribute...)
    app.state.AGENT = None
    app.state.PAUSED_USERS = set()
    
    # Estructura básica del Log por defecto
    app.state.df_log = pd.DataFrame(columns=[
        "Timestamp", "Numero WhatsApp", "Mensaje Usuario", "Respuesta Agente", "Tokens Consumidos"
    ])

    # 2. Cargar variables de entorno
    app.state.WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
    app.state.WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
    app.state.VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
    app.state.DATA_PATH = os.getenv("DATA_PATH")
    app.state.DATA_CONTEXT= os.getenv("DATA_CONTEXT")
    app.state.LOG_FILE_PATH = os.getenv("LOG_FILE_PATH")  
    app.state.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    app.state.GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME")

    # 3. Inicializar Log desde archivo (si existe)
    if app.state.LOG_FILE_PATH:
        try:
            if os.path.exists(app.state.LOG_FILE_PATH):
                app.state.df_log = pd.read_csv(app.state.LOG_FILE_PATH, encoding='latin-1')
                logging.info(f"[Startup] Log cargado: {len(app.state.df_log)} registros.")
            else:
                # Si no existe, creamos el archivo vacío con cabeceras
                app.state.df_log.to_csv(app.state.LOG_FILE_PATH, index=False, encoding='latin-1')
                logging.info(f"[Startup] Nuevo archivo de log creado: {app.state.LOG_FILE_PATH}")
        except Exception as e:
            logging.error(f"[Startup] Error crítico al cargar Log: {e}. Se usará DataFrame en memoria.")
            # app.state.df_log ya tiene un DataFrame vacío por la línea inicial, así que el server no muere.

    # 4. Inicializar Agente
    logging.info("[Startup] Inicializando agente Gemini...")
    try:
        if not app.state.GOOGLE_API_KEY:
            logging.error("[Startup] FALTA GOOGLE_API_KEY en .env")
        elif not app.state.DATA_PATH:
            logging.error("[Startup] FALTA DATA_PATH en .env")
        else:
            app.state.AGENT = create_agent_with_csv_genai(
                app.state.DATA_PATH, 
                app.state.GEMINI_MODEL_NAME, 
                app.state.GOOGLE_API_KEY
            )
            
        if app.state.AGENT:
            logging.info("[Startup] Agente listo y operativo.")
        else:
            logging.warning("[Startup] El agente NO se pudo inicializar (revisa rutas o API Key).")
            
    except Exception as e:
        logging.error(f"[Startup] Error fatal inicializando Agente: {e}")
        app.state.AGENT = None

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))