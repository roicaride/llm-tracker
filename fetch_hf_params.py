"""
Descarga el conteo exacto de parámetros de modelos open weight
desde la API pública de HuggingFace y guarda el resultado en
data/hf_params.json.

Solo se incluyen modelos cuyo nombre en Artificial Analysis NO
contiene el tamaño (p. ej. "Mistral Large 2 (Jul '24)"), porque
para los demás ("Llama 3.1 70B") el regex de params_lookup.py
ya funciona perfectamente.

Para modelos MoE se especifica active_b manualmente, ya que
safetensors.total solo da el total de parámetros.
"""

import json
import os
import sys
import time
import requests
from datetime import datetime, timezone

OUTPUT_PATH = "data/hf_params.json"

# ─────────────────────────────────────────────────────────────
# MAPA: nombre AA → (hf_id, active_b)
# active_b: parámetros activos (MoE); None si denso o sin dato
# ─────────────────────────────────────────────────────────────
HF_ID_MAP: dict[str, tuple[str, float | None]] = {

    # ── Snowflake Arctic ──────────────────────────────────────
    "Arctic Instruct": ("Snowflake/snowflake-arctic-instruct", 17.0),

    # ── AI21 Jamba ────────────────────────────────────────────
    "Jamba 1.5 Large": ("ai21labs/Jamba-1.5-Large", 94.0),
    "Jamba 1.5 Mini":  ("ai21labs/Jamba-1.5-Mini",  12.0),
    "Jamba 1.6 Large": ("ai21labs/Jamba-1.5-Large", 94.0),
    "Jamba 1.6 Mini":  ("ai21labs/Jamba-1.5-Mini",  12.0),

    # ── Cohere ────────────────────────────────────────────────
    "Command A":          ("CohereForAI/c4ai-command-a-03-2025", None),
    "Command-R (Mar '24)": ("CohereForAI/c4ai-command-r-v01",   None),
    "Command-R+ (Apr '24)": ("CohereForAI/c4ai-command-r-plus", None),
    "Tiny Aya Global":    ("CohereForAI/aya-expanse-8b",         None),

    # ── Databricks DBRX ───────────────────────────────────────
    "DBRX Instruct": ("databricks/dbrx-instruct", 36.0),

    # ── DeepSeek ──────────────────────────────────────────────
    "DeepSeek R1 (Jan '25)":         ("deepseek-ai/DeepSeek-R1",              37.0),
    "DeepSeek R1 0528 (May '25)":    ("deepseek-ai/DeepSeek-R1-0528",         37.0),
    "DeepSeek V3 (Dec '24)":         ("deepseek-ai/DeepSeek-V3",              37.0),
    "DeepSeek V3 0324":              ("deepseek-ai/DeepSeek-V3-0324",         37.0),
    "DeepSeek-V2-Chat":              ("deepseek-ai/DeepSeek-V2-Chat",         21.0),
    "DeepSeek-V2.5":                 ("deepseek-ai/DeepSeek-V2.5",            21.0),
    "DeepSeek-V2.5 (Dec '24)":       ("deepseek-ai/DeepSeek-V2.5-1210",       21.0),
    "DeepSeek-Coder-V2":             ("deepseek-ai/DeepSeek-Coder-V2-Instruct",     21.0),
    "DeepSeek Coder V2 Lite Instruct": ("deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", 2.4),

    # ── Meta Llama 4 (active=17B, total mucho mayor) ──────────
    "Llama 4 Maverick": ("meta-llama/Llama-4-Maverick-17B-128E-Instruct", 17.0),
    "Llama 4 Scout":    ("meta-llama/Llama-4-Scout-17B-16E-Instruct",     17.0),

    # ── Microsoft Phi ─────────────────────────────────────────
    "Phi-4":                    ("microsoft/phi-4",                       None),
    "Phi-4 Mini Instruct":      ("microsoft/Phi-4-mini-instruct",         None),
    "Phi-4 Multimodal Instruct": ("microsoft/Phi-4-multimodal-instruct",  None),

    # ── Mistral ───────────────────────────────────────────────
    "Mistral Large (Feb '24)":    ("mistralai/Mistral-Large-Instruct-2402",       None),
    "Mistral Large 2 (Jul '24)":  ("mistralai/Mistral-Large-Instruct-2407",       None),
    "Mistral Large 2 (Nov '24)":  ("mistralai/Mistral-Large-Instruct-2411",       None),
    "Mistral Large 3":            ("mistralai/Mistral-Large-3-Instruct",           None),
    "Mistral Medium 3":           ("mistralai/Mistral-Medium-3-Instruct-2025-05",  None),
    "Mistral Medium 3.1":         ("mistralai/Mistral-Medium-3-Instruct-2025-05",  None),
    "Mistral Saba":               ("mistralai/Mistral-Saba-24B-Instruct-2502",     None),
    "Mistral Small (Feb '24)":    ("mistralai/Mistral-7B-Instruct-v0.2",           None),
    "Mistral Small (Sep '24)":    ("mistralai/Mistral-Small-Instruct-2409",        None),
    "Mistral Small 3":            ("mistralai/Mistral-Small-3.1-24B-Instruct-2503", None),
    "Mistral Small 3.1":          ("mistralai/Mistral-Small-3.1-24B-Instruct-2503", None),
    "Pixtral Large":              ("mistralai/Pixtral-Large-Instruct-2411",         None),
    "Devstral Small (May '25)":   ("mistralai/Devstral-Small-2505",                None),
    "Magistral Medium 1":         ("mistralai/Magistral-Medium-2506",              None),
    "Magistral Medium 1.2":       ("mistralai/Magistral-Medium-2506",              None),
    "Magistral Small 1":          ("mistralai/Magistral-Small-2506",               None),
    "Magistral Small 1.2":        ("mistralai/Magistral-Small-2506",               None),

    # ── Moonshot / Kimi ───────────────────────────────────────
    "Kimi K2": ("moonshotai/Kimi-K2-Instruct", None),

    # ── OpenChat ──────────────────────────────────────────────
    "OpenChat 3.5 (1210)": ("openchat/openchat-3.5-1210", None),

    # ── Perplexity ────────────────────────────────────────────
    "R1 1776": ("perplexity-ai/r1-1776", 37.0),

    # ── Reka ──────────────────────────────────────────────────
    "Reka Flash (Sep '24)": ("RekaAI/reka-flash-3", None),
    "Reka Flash 3":         ("RekaAI/reka-flash-3", None),

    # ── Upstage Solar ─────────────────────────────────────────
    "Solar Mini": ("upstage/solar-1-mini-chat", None),

    # ── xAI Grok-1 ────────────────────────────────────────────
    "Grok-1": ("xai-org/grok-1", 86.0),

    # ── Deep Cogito ───────────────────────────────────────────
    "Cogito v2.1 (Reasoning)": ("deepcogito/cogito-v1-preview-llama-70B", None),
}

HEADERS = {
    "User-Agent": "llm-tracker/1.0 (github.com/roicaride/llm-tracker)",
    "Accept": "application/json",
}

# Si hay un token HF disponible lo usamos para desbloquear modelos gated
_hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if _hf_token:
    HEADERS["Authorization"] = f"Bearer {_hf_token}"


def get_params_from_hf(session: requests.Session, hf_id: str) -> float | None:
    """
    Devuelve el total de parámetros en miles de millones usando
    la API pública de HuggingFace, o None si no hay datos.
    """
    url = f"https://huggingface.co/api/models/{hf_id}"
    try:
        resp = session.get(url, timeout=20)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()

        # safetensors.total es la fuente más fiable
        st = data.get("safetensors") or {}
        if "total" in st:
            return st["total"] / 1e9

        return None
    except requests.RequestException as e:
        print(f"    Error de red ({hf_id}): {e}")
        return None


if __name__ == "__main__":
    print("Descargando parámetros desde HuggingFace API...")
    session = requests.Session()
    session.headers.update(HEADERS)

    results: dict = {}
    ok = 0

    for aa_name, (hf_id, active_b) in HF_ID_MAP.items():
        total_b = get_params_from_hf(session, hf_id)
        if total_b is not None:
            results[aa_name] = {
                "total_b":  round(total_b, 3),
                "active_b": active_b,
                "hf_id":    hf_id,
            }
            active_str = f" ({active_b}B activos)" if active_b else ""
            print(f"  OK {aa_name}: {total_b:.1f}B{active_str}")
            ok += 1
        else:
            print(f"  -- {aa_name}: sin safetensors en HF ({hf_id})")
        time.sleep(0.3)

    payload = {
        "_fetched_at": datetime.now(timezone.utc).isoformat(),
        "params": results,
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nOK - {ok}/{len(HF_ID_MAP)} modelos con datos en {OUTPUT_PATH}")
