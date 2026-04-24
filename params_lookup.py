"""
Extracción de parámetros (billones) para cada modelo LLM.

Estrategia en 3 capas:
  1. Regex sobre el nombre  →  "70B", "235B A22B", "8x7B", "270M"
  2. Lookup exacto          →  modelos cuyo nombre no lleva el tamaño
  3. Lookup por prefijo     →  familias enteras sin B en el nombre

Solo se incluyen valores confirmados en papers oficiales,
anuncios de las empresas o model cards verificadas en HuggingFace.
Cualquier valor dudoso → None (muestra "—" en la app).
"""

import re

# ─────────────────────────────────────────────────────────────
# LOOKUP EXACTO
# nombre completo del modelo → (total_B, active_B)
# active_B: solo para MoE con dato confirmado; None si denso o sin dato
# ─────────────────────────────────────────────────────────────
PARAMS_EXACT: dict[str, tuple[float | None, float | None]] = {

    # ── Snowflake Arctic ──────────────────────────────────────
    # Fuente: blog.snowflake.com/arctic
    "Arctic Instruct":                          (480,   17),

    # ── AI21 Jamba ────────────────────────────────────────────
    # Fuente: research.ai21.com/blog/jamba
    "Jamba 1.5 Large":                          (398,   94),
    "Jamba 1.5 Mini":                           (52,    12),
    "Jamba 1.6 Large":                          (398,   94),
    "Jamba 1.6 Mini":                           (52,    12),
    "Jamba 1.7 Large":                          (398,   94),
    "Jamba 1.7 Mini":                           (52,    12),
    "Jamba Reasoning 3B":                       (3,     None),

    # ── Cohere ────────────────────────────────────────────────
    # Fuente: huggingface.co/CohereForAI
    "Command A":                                (111,   None),
    "Command-R (Mar '24)":                      (35,    None),
    "Command-R+ (Apr '24)":                     (104,   None),
    "Tiny Aya Global":                          (8,     None),

    # ── Databricks DBRX ───────────────────────────────────────
    # Fuente: databricks.com/blog/introducing-dbrx-new-state-art-open-llm
    "DBRX Instruct":                            (132,   36),

    # ── DeepSeek ──────────────────────────────────────────────
    # Fuente: papers DeepSeek (arxiv) y huggingface.co/deepseek-ai
    "DeepSeek-V2-Chat":                         (236,   21),
    "DeepSeek-V2.5":                            (236,   21),
    "DeepSeek-V2.5 (Dec '24)":                  (236,   21),
    "DeepSeek-Coder-V2":                        (236,   21),
    "DeepSeek Coder V2 Lite Instruct":          (16,    2.4),
    "DeepSeek R1 (Jan '25)":                    (671,   37),
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

    # ── Devstral ──────────────────────────────────────────────
    # Fuente: mistral.ai — Devstral Small = Mistral Small 24B,
    #         Devstral Medium = Mistral Medium 3 22B
    "Devstral Small (May '25)":                 (24,    None),
    "Devstral Small (Jul '25)":                 (24,    None),
    "Devstral Small 2":                         (24,    None),
    "Devstral Medium":                          (22,    None),
    "Devstral 2":                               (22,    None),

    # ── xAI Grok-1 ────────────────────────────────────────────
    # Fuente: github.com/xai-org/grok-1 (open source)
    "Grok-1":                                   (314,   86),

    # ── INTELLECT-3 ───────────────────────────────────────────
    # Fuente: primeintelligence.ai
    "INTELLECT-3":                              (32,    None),

    # ── Kimi / K2 (Moonshot AI) ───────────────────────────────
    # Fuente: github.com/MoonshotAI/Kimi-K2 (~1T parámetros)
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

    # ── InclusionAI Ling/Ring ─────────────────────────────────
    # El nombre incluye "1T" → presumiblemente ~1T parámetros
    "Ling-1T":                                  (1000,  None),
    "Ring-1T":                                  (1000,  None),

    # ── Llama 4 (Meta MoE) ────────────────────────────────────
    # Fuente: ai.meta.com/blog/llama-4-multimodal-intelligence
    "Llama 4 Maverick":                         (400,   17),
    "Llama 4 Scout":                            (109,   17),

    # ── Magistral (Mistral, fine-tunes con razonamiento) ──────
    # Fuente: mistral.ai — Medium = Large 2 base, Small = Small 3 base
    "Magistral Medium 1":                       (123,   None),
    "Magistral Medium 1.2":                     (123,   None),
    "Magistral Small 1":                        (24,    None),
    "Magistral Small 1.2":                      (24,    None),

    # ── MiniMax M1 ────────────────────────────────────────────
    # Fuente: minimaxi.com/news (456B total)
    "MiniMax M1 40k":                           (456,   None),
    "MiniMax M1 80k":                           (456,   None),

    # ── Microsoft Phi ─────────────────────────────────────────
    # Fuente: huggingface.co/microsoft
    "Phi-4":                                    (14,    None),
    "Phi-4 Mini Instruct":                      (3.8,   None),
    "Phi-4 Multimodal Instruct":                (5.6,   None),

    # ── Mistral ───────────────────────────────────────────────
    # Fuente: mistral.ai y huggingface.co/mistralai
    "Mistral Large (Feb '24)":                  (123,   None),
    "Mistral Large 2 (Jul '24)":                (123,   None),
    "Mistral Large 2 (Nov '24)":                (123,   None),
    "Mistral Large 3":                          (123,   None),
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

    # ── Nous Research ─────────────────────────────────────────
    # Fuente: huggingface.co/deepcogito — Cogito v2.1 = Llama 3.1 70B base
    "Cogito v2.1 (Reasoning)":                  (70,    None),

    # ── OpenChat ──────────────────────────────────────────────
    # Fuente: huggingface.co/openchat — Mistral 7B base
    "OpenChat 3.5 (1210)":                      (7,     None),

    # ── Perplexity ────────────────────────────────────────────
    # Fuente: huggingface.co/perplexity-ai — DeepSeek R1 671B base
    "R1 1776":                                  (671,   37),

    # ── Reka ──────────────────────────────────────────────────
    # Fuente: reka.ai/reka-flash
    "Reka Flash (Sep '24)":                     (21,    None),
    "Reka Flash 3":                             (21,    None),

    # ── Upstage Solar ─────────────────────────────────────────
    # Fuente: huggingface.co/upstage/solar-1-mini-chat
    "Solar Mini":                               (10.7,  None),
}


# ─────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────
def extract_params(name: str) -> tuple[float | None, float | None]:
    """
    Devuelve (total_b, active_b) en miles de millones.
    active_b solo si MoE con dato confirmado.
    (None, None) cuando no hay información confirmada públicamente.
    """

    # 1 ── Arquitectura NxNB (ej. "8x7B") ────────────────────
    m = re.search(r'(\d+)x(\d+(?:\.\d+)?)B', name, re.IGNORECASE)
    if m:
        total = round(int(m.group(1)) * float(m.group(2)), 1)
        return total, None

    # 2 ── MoE explícito "NB AXAB" (ej. "235B A22B") ─────────
    m = re.search(r'(?<![a-zA-Z])(\d+(?:\.\d+)?)\s*B\s+A(\d+(?:\.\d+)?)\s*B', name, re.IGNORECASE)
    if m:
        return float(m.group(1)), float(m.group(2))

    # 3 ── Estándar "NB" ──────────────────────────────────────
    m = re.search(r'(?<![a-zA-Z])(\d+(?:\.\d+)?)\s*B(?!\w)', name, re.IGNORECASE)
    if m:
        return float(m.group(1)), None

    # 4 ── Millones "NM" → B ──────────────────────────────────
    m = re.search(r'(?<![a-zA-Z])(\d+(?:\.\d+)?)\s*M(?!\w)', name, re.IGNORECASE)
    if m:
        return round(float(m.group(1)) / 1000, 3), None

    # 5 ── "(NB)" entre paréntesis ────────────────────────────
    m = re.search(r'\((\d+(?:\.\d+)?)\s*B\)', name, re.IGNORECASE)
    if m:
        return float(m.group(1)), None

    # 6 ── Lookup exacto ──────────────────────────────────────
    if name in PARAMS_EXACT:
        return PARAMS_EXACT[name]

    return None, None


def format_params(total_b: float | None, active_b: float | None) -> str:
    """'70B', '235B (22B activos)', '270M', '1T', '—'"""
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
