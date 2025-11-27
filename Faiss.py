import os
import pandas as pd
import numpy as np
import traceback
import Rag_functions as rag
import logging
import faiss # Librería para el índice vectorial
from tqdm import tqdm

from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
import uvicorn
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI(title="WhatsApp RAG Agent Gemini + FAISS")

# ============================================================
#               CONFIGURACIÓN DE PERSISTENCIA
# ============================================================

# Rutas de persistencia del RAG:
FAISS_INDEX_PATH = "faiss_index.bin"
CORPUS_PATH = "corpus.txt"

# ============================================================
#                     AGENTE RAG + FAISS
# ============================================================

def create_faiss_index(df: pd.DataFrame, client: genai.Client):
    """
    Genera embeddings, crea el índice FAISS y guarda el índice y el corpus.
    Solo se llama si los archivos persistentes no existen.
    """
    logging.info("Creando textos para embedding...")

    cols_to_embed = [col for col in df.columns if col not in ['id', 'ID']]
    # Genera el 'text_chunk' uniendo valores de columnas
    df["text_chunk"] = df[cols_to_embed].astype(str).agg(" | ".join, axis=1)

    corpus = df["text_chunk"].tolist()
    logging.info(f"Generando embeddings para {len(corpus)} fragmentos...")

    embeddings_list = []
    batch_size = 100

    for i in tqdm(range(0, len(corpus), batch_size)):
        batch = corpus[i:i + batch_size]

        try:
            response = client.models.embed_content(
                model=app.state.EMBEDDING_MODEL,
                contents=batch
            )
            # Accede a la lista de embeddings usando el atributo .embeddings
            for emb in response.embeddings:
                embeddings_list.append(emb.values)

        except Exception as e:
            logging.error(f"Error en embedding batch {i}: {e}")
            continue

    if not embeddings_list:
        logging.error("No se pudieron generar embeddings. FAISS fallará.")
        return None, None

    embeddings_array = np.array(embeddings_list).astype("float32")
    dimension = embeddings_array.shape[1]

    logging.info(f"Creando índice FAISS con dimensión {dimension}...")
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
    
    # 4. === PERSISTENCIA: GUARDAR ÍNDICE Y CORPUS ===
    try:
        faiss.write_index(index, FAISS_INDEX_PATH)
        with open(CORPUS_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(corpus))
        logging.info(f"Índice FAISS guardado en {FAISS_INDEX_PATH} y corpus en {CORPUS_PATH}")
    except Exception as e:
        logging.error(f"Error al guardar FAISS o Corpus: {e}")
    # ================================================

    return index, corpus


def create_rag_agent(data_path: str, model_name: str, api_key: str):
    try:
        client = genai.Client(api_key=api_key)
        logging.info(f"Cliente Gemini inicializado con modelo {model_name}")
    except Exception as e:
        logging.error(f"No se pudo inicializar cliente Gemini: {e}")
        return None

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        logging.error(f"Error cargando CSV: {e}")
        return None

    faiss_index = None
    corpus = None

    # === PERSISTENCIA: INTENTAR CARGAR ÍNDICE ===
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CORPUS_PATH):
        try:
            logging.info("Archivos FAISS y Corpus encontrados. Cargando desde disco...")
            faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            with open(CORPUS_PATH, "r", encoding="utf-8") as f:
                corpus = f.read().split("\n")
            logging.info(f"Carga exitosa. Dimensión: {faiss_index.d}, Documentos: {len(corpus)}")
            
        except Exception as e:
            logging.error(f"Error al cargar FAISS o Corpus: {e}. Regenerando...")
            # Si falla la carga, generamos de nuevo
            faiss_index, corpus = create_faiss_index(df, client) 
    else:
        logging.info("Archivos FAISS/Corpus no encontrados. Generando nuevos embeddings...")
        faiss_index, corpus = create_faiss_index(df, client)
    # ============================================

    if faiss_index is None:
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


def process_message_with_agent(agent_data: dict, from_wa: str, body: str):
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

        # --------- EMBEDDING DE CONSULTA ----------
        query_embedding_response = client.models.embed_content(
            model=app.state.EMBEDDING_MODEL,
            contents=[pregunta] # Uso de 'contents'
        )

        # Acceso al atributo .embeddings y luego al primer elemento de la lista
        query_embedding = np.array(
            query_embedding_response.embeddings[0].values
        ).astype("float32").reshape(1, -1)

        # --------- BUSQUEDA FAISS ----------
        k = 5
        distances, indices = faiss_index.search(query_embedding, k)

        retrieved_context = "\n---\n".join([corpus[i] for i in indices[0]])

        full_context = (
            f"{system_instruction}\n\n"
            "--- INICIO DE DATOS RELEVANTES ---\n"
            f"{retrieved_context}\n"
            "--- FIN DE DATOS RELEVANTES ---\n"
        )

        final_prompt = f"{full_context}\n\nPREGUNTA DEL USUARIO: {pregunta}"

        # --------- GENERACIÓN GEMINI ----------
        response = client.models.generate_content(
            model=model_name,
            contents=[types.Content(parts=[types.Part(text=final_prompt)])]
        )

        respuesta = response.text

        # --------- ENVÍO POR WHATSAPP ----------
        rag.send_whatsapp_text(wp_token, wp_phone_number_id, numero_ok, respuesta)

        # --------- LOG ----------
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

    app.state.WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
    app.state.WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
    app.state.VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
    app.state.DATA_PATH = os.getenv("DATA_PATH")
    app.state.DATA_CONTEXT = os.getenv("DATA_CONTEXT")
    app.state.LOG_FILE_PATH = os.getenv("LOG_FILE_PATH")

    app.state.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    app.state.GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME")
    app.state.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

    app.state.AGENT = None

    # --- CONTEXTO ---
    try:
        df_ctx = pd.read_csv(app.state.DATA_CONTEXT)
        df_ctx["Descripcion"] = df_ctx["Descripcion"].replace('', np.nan).replace('   ', np.nan)
        df_ctx = df_ctx.set_index("Campo")
        app.state.column_context = df_ctx["Descripcion"].to_dict()
    except Exception as e:
        logging.error(f"Error cargando contexto: {e}")

    # --- RAG ---
    try:
        # Llama a la función que ahora maneja la carga/generación del índice FAISS
        app.state.AGENT = create_rag_agent(
            app.state.DATA_PATH,
            app.state.GEMINI_MODEL_NAME,
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
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))