"""
Comprueba si un modelo de AA está disponible en la librería oficial de Ollama.

El matching es fuzzy porque los nombres son distintos:
  Ollama:  "llama3.3"         AA: "Llama 3.3 Instruct 70B"
  Ollama:  "deepseek-r1"      AA: "DeepSeek R1 (Jan '25)"
  Ollama:  "qwen2.5"          AA: "Qwen2.5 Instruct 72B"
  Ollama:  "mistral-small"    AA: "Mistral Small 3.1"

Estrategia: normalizar ambos nombres a palabras y buscar secuencia contigua.
"""

import re
import json
import os


def _normalize(s: str) -> list[str]:
    """Convierte un nombre en lista de tokens normalizados."""
    # Quitar texto entre paréntesis: "(Jan '25)", "(Reasoning)", etc.
    s = re.sub(r'\s*\([^)]*\)', '', s)
    # Quitar texto tras guión que sea solo descriptor: "Non-reasoning" → ""
    # (no quitamos guiones que forman parte del nombre)
    s = s.lower()
    # Insertar espacio en fronteras letra-dígito: "llama3" → "llama 3", "r1" → "r 1"
    s = re.sub(r'([a-z])(\d)', r'\1 \2', s)
    s = re.sub(r'(\d)([a-z])', r'\1 \2', s)
    # Reemplazar separadores y eliminar cualquier char no alfanumérico restante
    s = re.sub(r'[-_./+]', ' ', s)
    s = re.sub(r'[^a-z0-9\s]', '', s)
    # Quitar tokens que no aportan al matching
    noise = {"instruct", "chat", "preview", "latest", "it", "v", "hf", "gguf"}
    tokens = [t for t in s.split() if t and t not in noise]
    return tokens


def check_ollama(aa_name: str, ollama_set: set[str]) -> bool:
    """True si el modelo de AA tiene un equivalente en la librería Ollama."""
    aa_tokens = _normalize(aa_name)

    for ollama_name in ollama_set:
        om_tokens = _normalize(ollama_name)
        if not om_tokens:
            continue
        n = len(om_tokens)
        # Buscar secuencia contigua om_tokens dentro de aa_tokens
        for i in range(len(aa_tokens) - n + 1):
            if aa_tokens[i:i + n] == om_tokens:
                return True

    return False


def load_ollama_set(path: str = "data/ollama_models.json") -> set[str]:
    """Carga el JSON de modelos Ollama guardado por fetch_ollama.py."""
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return set(json.load(f).get("models", []))
