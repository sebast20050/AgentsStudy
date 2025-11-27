from datetime import datetime
import os
import pandas as pd
import requests
import logging

def guardar_log(dataframe, numero_wa, usuario_msg, agente_msg, tokens="N/A") -> pd.DataFrame:
    """
    El Objetivo de esta función es guardar un registro de las interacciones en un archivo Excel.
    Hago que reciba un dataframe y no el excel, para que por cada registro nuevo no tenga que leer todo el excel,
    sino que solo lo lea una vez por cada llamada a la función.
    """
    try:
        fecha_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        nuevo_registro = {
            "Timestamp": [fecha_hora],
            "Numero WhatsApp": [numero_wa],
            "Mensaje Usuario": [usuario_msg],
            "Respuesta Agente": [agente_msg],
            "Tokens Consumidos": [tokens]
        }
        df_nuevo = pd.DataFrame(nuevo_registro)
        df_nuevo=pd.concat([dataframe, df_nuevo], ignore_index=True)
        logging.info(f"[Log] Guardando interacción en el log: {df_nuevo.to_string()}")
        return df_nuevo
    except Exception as e:
        logging.error(f"[Log Error] No se pudo guardar en el Dataframe: {e}")



def send_whatsapp_text(wp_token: str, wp_phone_number_id: str, to: str, body) -> dict:
    """Envía texto por WhatsApp."""
    if not body:
        return {}
    
    body_str = str(body)[:4096]
    to_clean = str(to).replace("+", "").strip()

    url = f"https://graph.facebook.com/v21.0/{wp_phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {wp_token}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_clean,
        "type": "text",
        "text": {"body": body_str},
    }

    try:
        print(f"--- ENVIANDO A: {to_clean} ---")
        print(f"Token usado: '{wp_token}'")
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        respuesta_json = resp.json()

        if resp.status_code == 200:
            print(f"[WhatsApp] Enviado OK. ID: {respuesta_json.get('messages', [{}])[0].get('id')}")
            return respuesta_json
        else:
            print(f"[WhatsApp ERROR] Status {resp.status_code}")
            print(f"DETALLE: {respuesta_json}")
            return respuesta_json

    except Exception as e:
        print(f"[WhatsApp EXCEPTION] {e}")
        return {}


def normalizar_numero(numero_recibido):
    """
    Convierte el formato internacional 549... al formato local 54...15...
    """
    if not numero_recibido:
        return None

    numero = str(numero_recibido).strip()

    if numero.startswith("549"):
        sin_nueve = numero.replace("549", "54", 1)

        # 11 (CABA)
        if sin_nueve.startswith("5411"):
            return "541115" + sin_nueve[4:]

        codigos_3 = ["299", "221", "351", "223", "341"]
        for cod in codigos_3:
            if sin_nueve.startswith("54" + cod):
                return f"54{cod}15" + sin_nueve[5:]

        codigos_4 = ["2972", "2944", "2324"]
        for cod in codigos_4:
            if sin_nueve.startswith("54" + cod):
                return f"54{cod}15" + sin_nueve[6:]

        return numero

    return numero

