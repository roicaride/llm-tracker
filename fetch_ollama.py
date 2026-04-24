"""
Descarga la lista de modelos oficiales de ollama.com/library
y la guarda en data/ollama_models.json.

Estrategia: búsqueda alfabética (a-z, 0-9) para cubrir todos los modelos.
Solo se guardan modelos sin prefijo de usuario (modelos del repositorio oficial).
"""

import requests
import json
import os
import re
import time
from datetime import datetime, timezone

OUTPUT_PATH = "data/ollama_models.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
}


def fetch_models_for_query(session: requests.Session, q: str) -> list[str]:
    resp = session.get(
        "https://ollama.com/search",
        params={"q": q},
        timeout=15,
    )
    resp.raise_for_status()
    return re.findall(r'x-test-search-response-title>([^<]+)<', resp.text)


def fetch_all() -> list[str]:
    session = requests.Session()
    session.headers.update(HEADERS)

    all_models: set[str] = set()

    # Búsqueda por cada carácter inicial para cubrir toda la librería
    queries = list("abcdefghijklmnopqrstuvwxyz0123456789")

    for q in queries:
        try:
            models = fetch_models_for_query(session, q)
            # Solo modelos oficiales: sin prefijo de usuario (sin "/")
            official = [m for m in models if "/" not in m]
            all_models.update(official)
            time.sleep(0.3)
        except requests.RequestException as e:
            print(f"  Aviso '{q}': {e}")

    return sorted(all_models)


if __name__ == "__main__":
    print("Descargando modelos oficiales de Ollama...")
    models = fetch_all()

    payload = {
        "_fetched_at": datetime.now(timezone.utc).isoformat(),
        "models": models,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"OK — {len(models)} modelos oficiales en {OUTPUT_PATH}")
    print("Muestra:", models[:20])
