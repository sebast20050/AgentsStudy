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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    if agent_data is None:
        logging.error("[AGENTE ERROR] El agente no está inicializado.")
        return

    try:
        client = agent_data.get("client")
        model_name = agent_data.get("model")
        contexto = agent_data.get("context")

        if not client: return

        numero_ok = rag.normalizar_numero(from_wa)
        
        # Validar pausa una vez más por si acaso
        if hasattr(app.state, 'PAUSED_USERS') and numero_ok in app.state.PAUSED_USERS:
            return

        pregunta = body.strip()
        
        # Construir prompt
        prompt = f"{contexto}\n\nPREGUNTA DEL USUARIO: {pregunta}"

        # 1. Llamada a la API
        response = client.models.generate_content(
            model=model_name,
            contents=[types.Content(parts=[types.Part(text=prompt)])]
        )
        respuesta = response.text

        # 2. Envío de la respuesta
        rag.send_whatsapp_text(app.state.WHATSAPP_TOKEN, app.state.WHATSAPP_PHONE_NUMBER_ID, numero_ok, respuesta)

        # 3. Guardar SOLO la respuesta del agente (la pregunta ya se guardó en el webhook)
        # Pasamos np.nan en el mensaje de usuario para no duplicarlo en el chat visual
        app.state.df_log = rag.guardar_log(
            app.state.df_log,
            numero_ok,
            np.nan, 
            respuesta,
            tokens="N/A"
        )
        
        if app.state.LOG_FILE_PATH:
            app.state.df_log.to_csv(app.state.LOG_FILE_PATH, index=False, encoding='latin-1')
        
    except Exception as e:
        logging.error(f"[AGENTE ERROR] {e}")
        traceback.print_exc()


# ============================================================
#   ENDPOINTS FRONTEND
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/contacts")
async def get_contacts():
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
    if not hasattr(app.state, 'df_log') or app.state.df_log is None:
        return []

    try:
        df = app.state.df_log
        df['Numero WhatsApp'] = df['Numero WhatsApp'].astype(str)
        # Filtramos y hacemos copia
        chat_df = df[df['Numero WhatsApp'] == str(phone)].copy()
        
        # Reemplazar NaN con "" para manejar filas parciales (solo user o solo agent)
        chat_df = chat_df.fillna("")
        
        history = []
        for _, row in chat_df.iterrows():
            # Si hay mensaje de usuario en esta fila
            if row['Mensaje Usuario']:
                history.append({
                    "sender": "user",
                    "text": str(row['Mensaje Usuario']),
                    "time": str(row['Timestamp'])
                })
            # Si hay respuesta de agente en esta fila (puede ser la misma fila o distinta)
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
    data = await request.json()
    phone = data.get("phone")
    message = data.get("message")
    
    if not phone or not message:
        return JSONResponse({"status": "error", "message": "Faltan datos"})

    try:
        # Enviar
        rag.send_whatsapp_text(app.state.WHATSAPP_TOKEN, app.state.WHATSAPP_PHONE_NUMBER_ID, phone, message)
        
        # Pausar
        app.state.PAUSED_USERS.add(phone)
        
        # Guardar en Log (Solo respuesta agente, usuario nan)
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
    data = await request.json()
    phone = data.get("phone")
    
    # 1. Si no está pausado, no hay nada que restaurar
    if phone not in app.state.PAUSED_USERS:
         return {"status": "no_change", "agent_status": "ACTIVE"}

    # 2. Verificar condición: Debe haber una respuesta previa MANUAL
    try:
        df = app.state.df_log
        # Filtrar mensajes de este usuario
        user_msgs = df[df['Numero WhatsApp'].astype(str) == str(phone)]
        
        # Obtener la columna de respuestas del agente, eliminar vacíos
        agent_responses = user_msgs['Respuesta Agente'].dropna()
        
        if agent_responses.empty:
             return JSONResponse({"status": "error", "message": "Debe dar una respuesta previa antes de restaurar."}, status_code=400)

        last_response = agent_responses.iloc[-1]
        
        if "[MANUAL]" not in str(last_response):
             return JSONResponse({"status": "error", "message": "Debe dar una respuesta previa (Manual) para poder restaurar."}, status_code=400)
             
    except Exception as e:
        logging.error(f"Error verificando historial para restore: {e}")
        # En caso de error de lectura, por seguridad no restauramos o forzamos? Mejor prevenimos.
        return JSONResponse({"status": "error", "message": "Error leyendo historial."}, status_code=500)

    # 3. Restaurar si pasó la validación
    app.state.PAUSED_USERS.remove(phone)
    return {"status": "success", "agent_status": "ACTIVE"}

# ============================================================
#   WEBHOOK (MODIFICADO PARA GUARDADO INMEDIATO)
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

            # --- CAMBIO CRITICO: Guardar mensaje del USUARIO inmediatamente ---
            # Esto permite que aparezca en el chat frontend sin esperar a la IA
            fecha_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Guardamos Usuario con Agente vacio
            registro_entrada = {
                "Timestamp": [fecha_hora],
                "Numero WhatsApp": [numero_ok],
                "Mensaje Usuario": [body],
                "Respuesta Agente": [np.nan], 
                "Tokens Consumidos": ["PENDING"]
            }
            df_temp = pd.DataFrame(registro_entrada)
            
            # Aseguramos que app.state.df_log no sea None
            if not hasattr(app.state, 'df_log') or app.state.df_log is None:
                 app.state.df_log = df_temp
            else:
                 app.state.df_log = pd.concat([app.state.df_log, df_temp], ignore_index=True)

            if app.state.LOG_FILE_PATH:
                 app.state.df_log.to_csv(app.state.LOG_FILE_PATH, index=False, encoding='latin-1')

            # --- LOGICA DE PROCESAMIENTO ---
            if numero_ok in app.state.PAUSED_USERS:
                logging.info(f"[WEBHOOK] Usuario {numero_ok} PAUSADO. Mensaje guardado, no se activa IA.")
            else:
                if app.state.AGENT:
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

@app.on_event("startup")
async def on_startup():
    logging.info("[Startup] Iniciando servidor...")
    app.state.AGENT = None
    app.state.PAUSED_USERS = set()
    app.state.df_log = pd.DataFrame(columns=["Timestamp", "Numero WhatsApp", "Mensaje Usuario", "Respuesta Agente", "Tokens Consumidos"])
    
    app.state.WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
    app.state.WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
    app.state.VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
    app.state.DATA_PATH = os.getenv("DATA_PATH")
    app.state.DATA_CONTEXT= os.getenv("DATA_CONTEXT")
    app.state.LOG_FILE_PATH = os.getenv("LOG_FILE_PATH")  
    app.state.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    app.state.GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME")

    if app.state.LOG_FILE_PATH and os.path.exists(app.state.LOG_FILE_PATH):
        try:
            app.state.df_log = pd.read_csv(app.state.LOG_FILE_PATH, encoding='latin-1')
            logging.info(f"[Startup] Log cargado: {len(app.state.df_log)} registros.")
        except: pass
    
    if app.state.GOOGLE_API_KEY and app.state.DATA_PATH:
         app.state.AGENT = create_agent_with_csv_genai(app.state.DATA_PATH, app.state.GEMINI_MODEL_NAME, app.state.GOOGLE_API_KEY)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))