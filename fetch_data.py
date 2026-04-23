import requests
import json
import os
from datetime import datetime, timezone

AA_API_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"
OUTPUT_PATH = "data/models.json"


def fetch_and_save(api_key: str):
    resp = requests.get(AA_API_URL, headers={"x-api-key": api_key}, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    payload["_fetched_at"] = datetime.now(timezone.utc).isoformat()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"OK — {len(payload.get('data', []))} modelos guardados en {OUTPUT_PATH}")


if __name__ == "__main__":
    key = os.environ.get("AA_API_KEY")
    if not key:
        raise SystemExit("Falta la variable de entorno AA_API_KEY")
    fetch_and_save(key)
