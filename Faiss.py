import os
import pandas as pd
import numpy as np
import traceback
import Rag_functions as rag # Asume que este módulo existe y contiene las funciones necesarias
import logging
import faiss
from tqdm import tqdm

from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
import uvicorn
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()

# ============================================================
#               CONFIGURACIÓN GENERAL
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI(title="WhatsApp RAG Agent Gemini + FAISS")

# Rutas de persistencia del RAG:


# ============================================================
#               FUNCIONES DE CREACIÓN/PERSISTENCIA
# ============================================================

def create_embeddings_and_index(df: pd.DataFrame, client: genai.Client, embedding_model: str):
    """
    1️⃣ Genera embeddings a partir del DataFrame y crea el objeto de índice FAISS.
    NO guarda los archivos de persistencia.
    """
    logging.info("Creando textos para embedding...")
    
    # 1. Preparar el corpus
    cols_to_embed = [col for col in df.columns if col not in ['id', 'ID']]
    # Genera el 'text_chunk' uniendo valores de columnas
    df["text_chunk"] = df[cols_to_embed].astype(str).agg(" | ".join, axis=1)
    corpus = df["text_chunk"].tolist()
    
    logging.info(f"Generando embeddings para {len(corpus)} fragmentos...")

    # 2. Generar embeddings por lotes
    embeddings_list = []
    batch_size = 100

    for i in tqdm(range(0, len(corpus), batch_size)):
        batch = corpus[i:i + batch_size]
        try:
            response = client.models.embed_content(
                model=embedding_model,
                contents=batch
            )
            # Asegurarse de que el objeto Value sea convertido a float
            for emb in response.embeddings:
                embeddings_list.append(emb.values) 
        except Exception as e:
            logging.error(f"Error en embedding batch {i}: {e}")
            continue

    if not embeddings_list:
        logging.error("No se pudieron generar embeddings.")
        return None, None

    # 3. Crear índice FAISS
    embeddings_array = np.array(embeddings_list).astype("float32")
    dimension = embeddings_array.shape[1]

    logging.info(f"Creando índice FAISS con dimensión {dimension}...")
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
    
    return index, corpus


def load_or_create_faiss_artifacts(df: pd.DataFrame, client: genai.Client, embedding_model: str):
    """
    2️⃣ Lógica de Persistencia: Intenta cargar. Si falla/no existe, crea y guarda.
    Esto previene la inyección de datos (regeneración) en cada startup.
    """
    faiss_index = None
    corpus = None
    
    # 1. INTENTAR CARGAR ÍNDICE
    if os.path.exists(app.state.FAISS_INDEX_PATH) and os.path.exists(app.state.CORPUS_PATH):
        try:
            logging.info("Archivos FAISS y Corpus encontrados. Cargando desde disco...")
            faiss_index = faiss.read_index(app.state.FAISS_INDEX_PATH)
            with open(app.state.CORPUS_PATH, "r", encoding="utf-8") as f:
                corpus = f.read().split("\n")
            logging.info(f"Carga exitosa. Documentos: {len(corpus)}")
            return faiss_index, corpus # Éxito: retorna y evita la inyección
            
        except Exception as e:
            logging.error(f"Error al cargar FAISS o Corpus: {e}. Regenerando...")
            # Si falla la carga, pasamos a generar
    else:
        logging.info("Archivos FAISS/Corpus no encontrados. Generando nuevos embeddings...")
    
    # 2. GENERAR Y GUARDAR ÍNDICE (Solo si no se pudo cargar)
    faiss_index, corpus = create_embeddings_and_index(df, client, embedding_model)
    
    if faiss_index is None:
        return None, None
        
    try:
        faiss.write_index(faiss_index, app.state.FAISS_INDEX_PATH )
        with open(app.state.CORPUS_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(corpus))
        logging.info(f"Índice FAISS guardado en {app.state.FAISS_INDEX_PATH } y corpus en {app.state.CORPUS_PATH}")
    except Exception as e:
        logging.error(f"Error al guardar FAISS o Corpus: {e}")

    return faiss_index, corpus


def create_rag_agent(data_path: str, model_name: str, embedding_model: str, api_key: str):
    """
    3️⃣ Inicializa el cliente, carga el dataset y obtiene los artefactos FAISS.
    """
    try:
        client = genai.Client(api_key=api_key)
        logging.info(f"Cliente Gemini inicializado con modelo {model_name}")
    except Exception as e:
        logging.error(f"No se pudo inicializar cliente Gemini: {e}")
        return None

    try:
        # Carga del dataset COMPLETO
        df = pd.read_csv(data_path)
    except Exception as e:
        logging.error(f"Error cargando CSV: {e}")
        return None

    # Llama a la función que maneja Carga/Generación/Persistencia
    faiss_index, corpus = load_or_create_faiss_artifacts(df, client, embedding_model)

    if faiss_index is None:
        logging.error("No se pudo obtener el índice FAISS. Agente no inicializado.")
        return None

    system_instruction = (
        "- REGLA: RESPONDER SIEMPRE en el idioma detectado en el mensaje.\n"
        "- Sos un árbol parlante que da información sobre árboles de CABA.\n"
        "- Presentate profesionalmente solo en la primera interacción.\n"
    )

    return {
        "client": client,
        "model": model_name,
        "system_instruction": system_instruction,
        "faiss_index": faiss_index,
        "corpus": corpus
    }

# ============================================================
#                      PROCESAMIENTO RAG
# ============================================================

def process_message_with_agent(agent_data: dict, from_wa: str, body: str):
    """
    Procesa el mensaje del usuario utilizando el agente RAG (FAISS + Gemini).
    """
    wp_token = app.state.WHATSAPP_TOKEN
    wp_phone_number_id = app.state.WHATSAPP_PHONE_NUMBER_ID

    try:
        client = agent_data["client"]
        model_name = agent_data["model"]
        system_instruction = agent_data["system_instruction"]
        faiss_index = agent_data["faiss_index"]
        corpus = agent_data["corpus"]

        numero_ok = rag.normalizar_numero(from_wa)
        pregunta = body.strip()

        logging.info(f"[RAG] Pregunta de {numero_ok}: {pregunta}")

        # --------- 1. EMBEDDING DE CONSULTA ----------
        query_embedding_response = client.models.embed_content(
            model=app.state.EMBEDDING_MODEL,
            contents=[pregunta]
        )

        query_embedding = np.array(
            query_embedding_response.embeddings[0].values
        ).astype("float32").reshape(1, -1)

        # --------- 2. BUSQUEDA FAISS ----------
        k = 5
        distances, indices = faiss_index.search(query_embedding, k)

        retrieved_context = "\n---\n".join([corpus[i] for i in indices[0]])

        # --------- 3. GENERACIÓN DE RESPUESTA (Gemini) ----------
        full_context = (
            f"{system_instruction}\n\n"
            "--- INICIO DE DATOS RELEVANTES ---\n"
            f"{retrieved_context}\n"
            "--- FIN DE DATOS RELEVANTES ---\n"
        )

        final_prompt = f"{full_context}\n\nPREGUNTA DEL USUARIO: {pregunta}"

        response = client.models.generate_content(
            model=model_name,
            contents=[types.Content(parts=[types.Part(text=final_prompt)])]
        )

        respuesta = response.text

        # --------- 4. ENVÍO Y LOG ----------
        rag.send_whatsapp_text(wp_token, wp_phone_number_id, numero_ok, respuesta)

        rag.guardar_log(
            app.state.df_log,
            numero_ok,
            pregunta,
            respuesta,
            tokens="N/A"
        )

        return {"status": "success", "response_text": respuesta}

    except Exception as e:
        logging.error(f"[RAG ERROR] {e}")
        traceback.print_exc()

        numero_ok = rag.normalizar_numero(from_wa)
        rag.send_whatsapp_text(
            wp_token, wp_phone_number_id, numero_ok,
            "Lo siento, hubo un error interno al procesar tu solicitud."
        )

        return {"status": "error", "message": str(e)}

# ============================================================
#                      ENDPOINTS API
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
        from_wa = msg.get("from")
        msg_type = msg.get("type")

        if app.state.AGENT is None:
            return JSONResponse({"status": "error", "message": "Agente no inicializado"}, 503)

        if msg_type == "text":
            body = msg["text"]["body"].strip()
            # Delegar el procesamiento al thread de fondo para respuesta asíncrona de WhatsApp
            background_tasks.add_task(process_message_with_agent, app.state.AGENT, from_wa, body)

        return JSONResponse({"status": "ok"})

    except Exception as e:
        logging.error(f"[Webhook ERROR] {e}")
        return JSONResponse({"status": "error"})


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
#                      STARTUP
# ============================================================

@app.on_event("startup")
async def startup_event():
    logging.info("[Startup] Cargando variables...")

    # Cargar variables de entorno
    app.state.WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
    app.state.WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
    app.state.VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
    app.state.DATA_PATH = os.getenv("DATA_PATH")
    app.state.DATA_CONTEXT = os.getenv("DATA_CONTEXT")
    app.state.LOG_FILE_PATH = os.getenv("LOG_FILE_PATH")
    app.state.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    app.state.GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME")
    app.state.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    app.state.FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH")
    app.state.CORPUS_PATH = os.getenv("CORPUS_PATH")
    app.state.AGENT = None

    # --- CONTEXTO ADICIONAL (Opcional) ---
    try:
        df_ctx = pd.read_csv(app.state.DATA_CONTEXT)
        df_ctx["Descripcion"] = df_ctx["Descripcion"].replace('', np.nan).replace('   ', np.nan)
        df_ctx = df_ctx.set_index("Campo")
        app.state.column_context = df_ctx["Descripcion"].to_dict()
    except Exception as e:
        logging.error(f"Error cargando contexto: {e}. Se ignorará.")

    # --- INICIALIZACIÓN RAG (FAISS/Gemini) ---
    try:
        # Llama a la función que ahora maneja la carga/generación del índice FAISS
        app.state.AGENT = create_rag_agent(
            app.state.DATA_PATH,
            app.state.GEMINI_MODEL_NAME,
            app.state.EMBEDDING_MODEL, # Incluir el modelo de embedding
            app.state.GOOGLE_API_KEY
        )
    except Exception as e:
        logging.error(f"Error inicializando RAG: {e}")
        app.state.AGENT = None

    # --- LOG ---
    try:
        if not os.path.exists(app.state.LOG_FILE_PATH):
            df = pd.DataFrame(columns=[
                "Timestamp", "Numero WhatsApp", "Mensaje Usuario",
                "Respuesta Agente", "Tokens Consumidos"
            ])
            df.to_csv(app.state.LOG_FILE_PATH, index=False, encoding="latin-1")

        app.state.df_log = pd.read_csv(app.state.LOG_FILE_PATH, encoding="latin-1")

    except Exception as e:
        logging.error(f"No se pudo cargar log: {e}")


if __name__ == "__main__":
    # Asegúrate de tener la variable de entorno PORT definida o usará 8000 por defecto
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))