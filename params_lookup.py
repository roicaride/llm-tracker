"""
Extracción de parámetros (billones) para cada modelo LLM.

Estrategia en 3 capas:
  1. Regex sobre el nombre  →  "70B", "235B A22B", "8x7B", "270M"
  2. Lookup exacto          →  modelos cuyo nombre no lleva el tamaño
  3. Lookup por prefijo     →  familias enteras (DeepSeek V3.x, etc.)

Fuentes: papers oficiales, anuncios de las empresas, HuggingFace model cards.
Valores no confirmados públicamente → None (muestra "—" en la app).
"""

import re

# ─────────────────────────────────────────────────────────────
# LOOKUP EXACTO  (nombre completo del modelo → (total_B, active_B))
# active_B: solo para MoE; None si es denso o desconocido
# ─────────────────────────────────────────────────────────────
PARAMS_EXACT: dict[str, tuple[float | None, float | None]] = {

    # ── Snowflake Arctic ──────────────────────────────────────
    "Arctic Instruct":                          (480,   17),   # 480B total, 17B active (MoE)

    # ── AI21 Jamba (Hybrid SSM-Transformer MoE) ───────────────
    "Jamba 1.5 Large":                          (398,   94),
    "Jamba 1.5 Mini":                           (52,    12),
    "Jamba 1.6 Large":                          (398,   94),
    "Jamba 1.6 Mini":                           (52,    12),
    "Jamba 1.7 Large":                          (398,   94),
    "Jamba 1.7 Mini":                           (52,    12),
    "Jamba Reasoning 3B":                       (3,     None),

    # ── Cohere ────────────────────────────────────────────────
    "Command A":                                (111,   None),
    "Command-R (Mar '24)":                      (35,    None),
    "Command-R+ (Apr '24)":                     (104,   None),
    "Tiny Aya Global":                          (8,     None),

    # ── Databricks DBRX ───────────────────────────────────────
    "DBRX Instruct":                            (132,   36),   # 132B total MoE, 36B active

    # ── DeepSeek (sin B en nombre) ────────────────────────────
    "DeepSeek-V2-Chat":                         (236,   21),
    "DeepSeek-V2.5":                            (236,   21),
    "DeepSeek-V2.5 (Dec '24)":                  (236,   21),
    "DeepSeek-Coder-V2":                        (236,   21),
    "DeepSeek Coder V2 Lite Instruct":          (16,    2.4),  # Lite = 16B total, 2.4B active
    "DeepSeek R1 (Jan '25)":                    (671,   37),   # 671B total MoE, 37B active
    "DeepSeek R1 0528 (May '25)":               (671,   37),
    "DeepSeek V3 (Dec '24)":                    (671,   37),
    "DeepSeek V3 0324":                         (671,   37),
    "DeepSeek V3.1 (Non-reasoning)":            (671,   37),
    "DeepSeek V3.1 (Reasoning)":                (671,   37),
    "DeepSeek V3.1 Terminus (Non-reasoning)":   (671,   37),
    "DeepSeek V3.1 Terminus (Reasoning)":       (671,   37),
    "DeepSeek V3.2 (Non-reasoning)":            (671,   37),
    "DeepSeek V3.2 (Reasoning)":                (671,   37),
    "DeepSeek V3.2 Exp (Non-reasoning)":        (671,   37),
    "DeepSeek V3.2 Exp (Reasoning)":            (671,   37),
    "DeepSeek V3.2 Speciale":                   (671,   37),

    # ── Devstral (Mistral Small base) ────────────────────────
    "Devstral 2":                               (24,    None),
    "Devstral Medium":                          (22,    None),
    "Devstral Small (May '25)":                 (24,    None),
    "Devstral Small (Jul '25)":                 (24,    None),
    "Devstral Small 2":                         (24,    None),

    # ── xAI Grok ──────────────────────────────────────────────
    "Grok-1":                                   (314,   86),   # open-source, 314B MoE

    # ── INTELLECT ─────────────────────────────────────────────
    "INTELLECT-3":                              (32,    None),

    # ── Kimi / K2 (Moonshot AI, ~1T) ─────────────────────────
    "Kimi K2":                                  (1000,  None),
    "Kimi K2 0905":                             (1000,  None),
    "Kimi K2 Thinking":                         (1000,  None),
    "Kimi K2.5 (Non-reasoning)":                (1000,  None),
    "Kimi K2.5 (Reasoning)":                    (1000,  None),
    "Kimi K2.6":                                (1000,  None),
    "K2 Think V2":                              (1000,  None),
    "K2-V2 (high)":                             (1000,  None),
    "K2-V2 (low)":                              (1000,  None),
    "K2-V2 (medium)":                           (1000,  None),

    # ── InclusionAI Ling ──────────────────────────────────────
    "Ling-1T":                                  (1000,  None),
    "Ring-1T":                                  (1000,  None),

    # ── Llama 4 (Meta MoE) ────────────────────────────────────
    "Llama 4 Maverick":                         (400,   17),   # 400B total, 17B active
    "Llama 4 Scout":                            (109,   17),   # 109B total, 17B active

    # ── Magistral (Mistral Large base) ───────────────────────
    "Magistral Medium 1":                       (123,   None),
    "Magistral Medium 1.2":                     (123,   None),
    "Magistral Small 1":                        (24,    None),
    "Magistral Small 1.2":                      (24,    None),

    # ── MiniMax ───────────────────────────────────────────────
    "MiniMax M1 40k":                           (456,   None),
    "MiniMax M1 80k":                           (456,   None),

    # ── Microsoft Phi ─────────────────────────────────────────
    "Phi-4":                                    (14,    None),
    "Phi-4 Mini Instruct":                      (3.8,   None),
    "Phi-4 Multimodal Instruct":                (5.6,   None),

    # ── Mistral (sin B en nombre) ─────────────────────────────
    "Mistral Large (Feb '24)":                  (123,   None),
    "Mistral Large 2 (Jul '24)":                (123,   None),
    "Mistral Large 2 (Nov '24)":                (123,   None),
    "Mistral Large 3":                          (123,   None),
    "Mistral Medium":                           (56,    None),  # original = 8x7B≈56B
    "Mistral Medium 3":                         (22,    None),
    "Mistral Medium 3.1":                       (22,    None),
    "Mistral Saba":                             (24,    None),
    "Mistral Small (Feb '24)":                  (7,     None),
    "Mistral Small (Sep '24)":                  (22,    None),
    "Mistral Small 3":                          (24,    None),
    "Mistral Small 3.1":                        (24,    None),
    "Mistral Small 3.2":                        (24,    None),
    "Mistral Small 4 (Non-reasoning)":          (24,    None),
    "Mistral Small 4 (Reasoning)":              (24,    None),
    "Pixtral Large":                            (123,   None),

    # ── Nous Research (fine-tunes Llama) ─────────────────────
    "Cogito v2.1 (Reasoning)":                  (70,    None),  # Llama 3.1 70B base

    # ── OpenChat ──────────────────────────────────────────────
    "OpenChat 3.5 (1210)":                      (7,     None),  # Mistral 7B base

    # ── Perplexity ────────────────────────────────────────────
    "R1 1776":                                  (671,   37),   # DeepSeek R1 base

    # ── Reka ──────────────────────────────────────────────────
    "Reka Flash (Sep '24)":                     (21,    None),
    "Reka Flash 3":                             (21,    None),

    # ── Upstage Solar ─────────────────────────────────────────
    "Solar Mini":                               (10.7,  None),

    # ── Gemma 3n / Gemma 4 "E" (Equivalent notation) ─────────
    # "E4B" = equivalent 4B compute; params reales son menores
    "Gemma 3n E2B Instruct":                    (0.5,   None),  # ~500M real
    "Gemma 3n E4B Instruct":                    (2.0,   None),  # ~2B real
    "Gemma 3n E4B Instruct Preview (May '25)":  (2.0,   None),
    "Gemma 4 E2B (Non-reasoning)":              (0.5,   None),
    "Gemma 4 E2B (Reasoning)":                  (0.5,   None),
    "Gemma 4 E4B (Non-reasoning)":              (2.0,   None),
    "Gemma 4 E4B (Reasoning)":                  (2.0,   None),

    # ── Granite (IBM, sin B en nombre) ───────────────────────
    "Granite 4.0 Micro":                        (0.4,   None),  # ~400M
    "Granite 4.0 H Small":                      (7,     None),  # ~7B (estimado)
}

# ─────────────────────────────────────────────────────────────
# LOOKUP POR PREFIJO
# Para familias donde múltiples variantes comparten tamaño
# Orden: más específico primero
# ─────────────────────────────────────────────────────────────
PARAMS_PREFIX: list[tuple[str, tuple[float | None, float | None]]] = [
    # Kimi Linear (MoE)
    ("Kimi Linear",         (48,    3)),

    # Ling flash/mini (InclusionAI)
    ("Ling 2.6 Flash",      (72,    None)),
    ("Ling-flash",          (72,    None)),
    ("Ling-mini",           (16,    None)),
    ("Ring-flash",          (72,    None)),

    # MiMo (Xiaomi)
    ("MiMo-V2-Pro",         (7,     None)),
    ("MiMo-V2.5-Pro",       (7,     None)),
    ("MiMo-V2-Omni",        (7,     None)),
    ("MiMo-V2-Flash",       (1.5,   None)),
    ("MiMo-V2.5",           (7,     None)),

    # LFM2.5 (Liquid AI)
    ("LFM2.5-VL-1.6B",      (1.6,   None)),
    ("LFM2.5-1.2B",         (1.2,   None)),
]


# ─────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────
def extract_params(name: str) -> tuple[float | None, float | None]:
    """
    Devuelve (total_b, active_b) en miles de millones (B = billion).
    active_b solo se rellena en modelos MoE con dato conocido.
    Devuelve (None, None) cuando no hay información confirmada.
    """

    # 1 ── Arquitectura 8x7B / NxNB (MoE tipo Mixtral) ────────
    m = re.search(r'(\d+)x(\d+(?:\.\d+)?)B', name, re.IGNORECASE)
    if m:
        experts = int(m.group(1))
        per_expert = float(m.group(2))
        total = round(experts * per_expert, 1)
        # Mixtral usa top-2, pero no extraemos activos aquí para no confundir
        return total, None

    # 2 ── Patrón MoE "NB AXAB" (ej. "235B A22B") ────────────
    m = re.search(r'(?<![a-zA-Z])(\d+(?:\.\d+)?)\s*B\s+A(\d+(?:\.\d+)?)\s*B', name, re.IGNORECASE)
    if m:
        return float(m.group(1)), float(m.group(2))

    # 3 ── Patrón estándar "NB" ────────────────────────────────
    m = re.search(r'(?<![a-zA-Z])(\d+(?:\.\d+)?)\s*B(?!\w)', name, re.IGNORECASE)
    if m:
        return float(m.group(1)), None

    # 4 ── Patrón millones "NM" → convertir a B ───────────────
    m = re.search(r'(?<![a-zA-Z])(\d+(?:\.\d+)?)\s*M(?!\w)', name, re.IGNORECASE)
    if m:
        return round(float(m.group(1)) / 1000, 3), None

    # 5 ── "(NB)" entre paréntesis ─────────────────────────────
    m = re.search(r'\((\d+(?:\.\d+)?)\s*B\)', name, re.IGNORECASE)
    if m:
        return float(m.group(1)), None

    # 6 ── Lookup exacto ───────────────────────────────────────
    if name in PARAMS_EXACT:
        return PARAMS_EXACT[name]

    # 7 ── Lookup por prefijo ──────────────────────────────────
    for prefix, result in PARAMS_PREFIX:
        if name.startswith(prefix):
            return result

    return None, None


def format_params(total_b: float | None, active_b: float | None) -> str:
    """Devuelve string legible: '70B', '235B (22B)', '270M', '1T', '—'."""
    if total_b is None:
        return "—"

    def fmt_b(b: float) -> str:
        if b >= 1000:
            t = b / 1000
            return f"{t:.0f}T" if t == int(t) else f"{t:.1f}T"
        if b >= 1:
            return f"{b:.0f}B" if b == int(b) else f"{b:.1f}B"
        return f"{round(b * 1000):.0f}M"

    s = fmt_b(total_b)
    if active_b is not None:
        s += f" ({fmt_b(active_b)} activos)"
    return s
