# ============================================================
# CLASIFICACIÓN DE LICENCIAS POR CREATOR — AA API
# Actualizado: 23 abr 2026
# Criterio: open_weight = pesos públicos descargables
#           open_source = además código entrenamiento / OSI
#           closed      = propietario, sin pesos públicos
#
# Fuentes consultadas: HuggingFace, páginas oficiales, papers
# ============================================================

CREATOR_LICENSE = {

    # ── OPEN SOURCE / OPEN WEIGHT ────────────────────────────

    "Alibaba": {
        "status": "open_weight",
        "license": "Apache 2.0 / Qwen License",
        "notes": "Qwen2.5+ es Apache 2.0. Versiones antiguas usan Qwen Community License con restricciones comerciales para >100M usuarios.",
        "hf": "https://huggingface.co/Qwen"
    },
    "Allen Institute for AI": {
        "status": "open_source",
        "license": "Apache 2.0",
        "notes": "OLMo y Molmo son fully open source: pesos, datos de entrenamiento y código.",
        "hf": "https://huggingface.co/allenai"
    },
    "Arcee AI": {
        "status": "open_weight",
        "license": "Apache 2.0",
        "notes": "Trinity Large Thinking pesos públicos en HuggingFace.",
        "hf": "https://huggingface.co/arcee-ai"
    },
    "ByteDance Seed": {
        "status": "mixed",
        "license": "Seed-OSS-36B: Apache 2.0 / Doubao: cerrado",
        "notes": "Seed-OSS-36B-Instruct es open weight. Doubao es propietario.",
        "hf": "https://huggingface.co/ByteDance-Seed"
    },
    "Deep Cogito": {
        "status": "open_weight",
        "license": "Apache 2.0",
        "notes": "Cogito v2.1 disponible en HuggingFace.",
        "hf": "https://huggingface.co/deepcogito"
    },
    "DeepSeek": {
        "status": "open_weight",
        "license": "MIT / DeepSeek License",
        "notes": "V3, R1 y variantes son open weight (MIT). Licencia propia para uso comercial a escala.",
        "hf": "https://huggingface.co/deepseek-ai"
    },
    "Google": {
        "status": "mixed",
        "license": "Gemma: Gemma ToS / Gemini: cerrado",
        "notes": "IMPORTANTE: Gemma (1/2/3/4) son open weight con Gemma Terms of Service — uso comercial permitido pero con restricciones. Gemini (1.0/1.5/2.x/3.x) es completamente propietario y cerrado. En AA aparecen mezclados bajo 'Google'.",
        "hf": "https://huggingface.co/google"
    },
    "IBM": {
        "status": "open_source",
        "license": "Apache 2.0",
        "notes": "Granite 3.x y 4.x son fully open source con datos de entrenamiento publicados.",
        "hf": "https://huggingface.co/ibm-granite"
    },
    "InclusionAI": {
        "status": "open_weight",
        "license": "Apache 2.0",
        "notes": "Ling series pesos públicos.",
        "hf": "https://huggingface.co/InclusionAI"
    },
    "Kimi": {
        "status": "open_weight",
        "license": "Kimi License (Modified MIT)",
        "notes": "K2 y variantes son open weight. Licencia propia — leer antes de uso comercial.",
        "hf": "https://huggingface.co/moonshotai"
    },
    "KwaiKAT": {
        "status": "open_weight",
        "license": "Apache 2.0",
        "notes": "KAT-Coder series disponible en HuggingFace.",
        "hf": "https://huggingface.co/kwaikeg"
    },
    "LG AI Research": {
        "status": "open_weight",
        "license": "Apache 2.0",
        "notes": "EXAONE 4.0 y K-EXAONE con pesos públicos.",
        "hf": "https://huggingface.co/LGAI-EXAONE"
    },
    "Liquid AI": {
        "status": "open_weight",
        "license": "Liquid AI License",
        "notes": "LFM2 series open weight. Licencia propia no OSI.",
        "hf": "https://huggingface.co/liquid-tech"
    },
    "MBZUAI Institute of Foundation Models": {
        "status": "open_weight",
        "license": "Apache 2.0",
        "notes": "K2 Think series open weight.",
        "hf": "https://huggingface.co/MBZUAI"
    },
    "Meta": {
        "status": "open_weight",
        "license": "Llama License / Apache 2.0",
        "notes": "Llama 2/3/3.1/3.2/3.3/4 open weight. Llama License con restricción para >700M usuarios activos mensuales. Muse Spark en preview sin licencia definitiva aún.",
        "hf": "https://huggingface.co/meta-llama"
    },
    "Microsoft Azure": {
        "status": "open_weight",
        "license": "MIT",
        "notes": "Phi-3, Phi-4 son open weight MIT. Phi-4 Multimodal también.",
        "hf": "https://huggingface.co/microsoft"
    },
    "MiniMax": {
        "status": "open_weight",
        "license": "MiniMax License (Modified MIT)",
        "notes": "M1, M2, M2.1, M2.5, M2.7 open weight con licencia propia.",
        "hf": "https://huggingface.co/MiniMaxAI"
    },
    "Mistral": {
        "status": "mixed",
        "license": "Apache 2.0 / Mistral License / comercial",
        "notes": "Mistral 7B, Mixtral 8x7B, Mistral Small son Apache 2.0. Mistral Large, Pixtral Large y modelos premium son propietarios. Devstral y Magistral: licencias propias.",
        "hf": "https://huggingface.co/mistralai"
    },
    "Nanbeige": {
        "status": "open_weight",
        "license": "Apache 2.0",
        "notes": "Nanbeige4.1 pesos públicos.",
        "hf": "https://huggingface.co/Nanbeige"
    },
    "Nous Research": {
        "status": "open_weight",
        "license": "Apache 2.0 / Llama License",
        "notes": "Fine-tunes sobre Llama — hereda licencia base. Hermes series open weight.",
        "hf": "https://huggingface.co/NousResearch"
    },
    "NVIDIA": {
        "status": "open_weight",
        "license": "Apache 2.0 / NVIDIA Open Model License",
        "notes": "Nemotron series open weight. Llama-based Nemotron hereda Llama License.",
        "hf": "https://huggingface.co/nvidia"
    },
    "OpenChat": {
        "status": "open_weight",
        "license": "Apache 2.0",
        "notes": "OpenChat 3.5 open weight sobre Mistral.",
        "hf": "https://huggingface.co/openchat"
    },
    "Prime Intellect": {
        "status": "open_weight",
        "license": "Apache 2.0",
        "notes": "INTELLECT-3 open weight.",
        "hf": "https://huggingface.co/PrimeIntellect"
    },
    "Sarvam": {
        "status": "open_weight",
        "license": "Apache 2.0",
        "notes": "Sarvam 105B y 30B open weight, enfocados en idiomas indios.",
        "hf": "https://huggingface.co/sarvamai"
    },
    "ServiceNow": {
        "status": "open_weight",
        "license": "Apache 2.0",
        "notes": "Apriel series open weight.",
        "hf": "https://huggingface.co/ServiceNow"
    },
    "StepFun": {
        "status": "open_weight",
        "license": "StepFun License",
        "notes": "Step 3.5 Flash open weight con licencia propia.",
        "hf": "https://huggingface.co/stepfun-ai"
    },
    "Swiss AI Initiative": {
        "status": "open_source",
        "license": "Apache 2.0",
        "notes": "Apertus fully open source, entrenado con datos públicos europeos.",
        "hf": "https://huggingface.co/swiss-ai"
    },
    "TII UAE": {
        "status": "open_weight",
        "license": "TII Falcon License",
        "notes": "Falcon series open weight. Licencia propia permisiva.",
        "hf": "https://huggingface.co/tiiuae"
    },
    "Trillion Labs": {
        "status": "open_weight",
        "license": "Apache 2.0",
        "notes": "Tri-21B open weight.",
        "hf": "https://huggingface.co/TrillionAI"
    },
    "Upstage": {
        "status": "open_weight",
        "license": "Apache 2.0 / CC BY 4.0",
        "notes": "Solar series open weight. Solar Mini y Solar Pro pesos públicos.",
        "hf": "https://huggingface.co/upstage"
    },
    "Xiaomi": {
        "status": "open_weight",
        "license": "Apache 2.0",
        "notes": "MiMo-V2 series open weight Apache 2.0.",
        "hf": "https://huggingface.co/XiaomiMiMo"
    },
    "Z AI": {
        "status": "open_weight",
        "license": "MIT / Z AI License",
        "notes": "GLM-4.x y GLM-5 series open weight. Licencia MIT para la mayoría.",
        "hf": "https://huggingface.co/THUDM"
    },

    # ── CERRADOS / PROPIETARIOS ──────────────────────────────

    "AI21 Labs": {
        "status": "closed",
        "license": "Propietario",
        "notes": "Jamba series API-only, sin pesos públicos.",
        "hf": None
    },
    "Amazon": {
        "status": "closed",
        "license": "Propietario (AWS)",
        "notes": "Nova series completamente propietario, solo via AWS Bedrock.",
        "hf": None
    },
    "Anthropic": {
        "status": "closed",
        "license": "Propietario",
        "notes": "Claude series completamente propietario. Sin pesos públicos.",
        "hf": None
    },
    "Baidu": {
        "status": "closed",
        "license": "Propietario",
        "notes": "ERNIE series API-only.",
        "hf": None
    },
    "China Mobile": {
        "status": "closed",
        "license": "Propietario",
        "notes": "JT-MINI propietario.",
        "hf": None
    },
    "Cohere": {
        "status": "mixed",
        "license": "Command-R: CC BY-NC / Command A: propietario",
        "notes": "Command-R y Command-R+ tienen pesos en HF con CC BY-NC (no comercial). Command A es cerrado. Tiny Aya es Apache 2.0.",
        "hf": "https://huggingface.co/CohereForAI"
    },
    "Databricks": {
        "status": "open_weight",
        "license": "Apache 2.0",
        "notes": "DBRX open weight Apache 2.0.",
        "hf": "https://huggingface.co/databricks"
    },
    "Inception": {
        "status": "closed",
        "license": "Propietario",
        "notes": "Mercury 2 propietario.",
        "hf": None
    },
    "Korea Telecom": {
        "status": "closed",
        "license": "Propietario",
        "notes": "Mi:dm K series propietario.",
        "hf": None
    },
    "LongCat": {
        "status": "closed",
        "license": "Desconocido",
        "notes": "Sin información pública de licencia.",
        "hf": None
    },
    "Motif Technologies": {
        "status": "open_weight",
        "license": "Apache 2.0",
        "notes": "Motif-2 open weight.",
        "hf": "https://huggingface.co/MotifTech"
    },
    "Naver": {
        "status": "closed",
        "license": "Propietario",
        "notes": "HyperCLOVA X propietario.",
        "hf": None
    },
    "OpenAI": {
        "status": "mixed",
        "license": "gpt-oss-120B: MIT / resto: propietario",
        "notes": "gpt-oss-120b es el único modelo open weight de OpenAI. Todo lo demás (GPT-4.x, o1, o3, o4) es propietario.",
        "hf": "https://huggingface.co/openai-community"
    },
    "Perplexity": {
        "status": "mixed",
        "license": "R1 1776: Apache 2.0 / Sonar: propietario",
        "notes": "R1 1776 es open weight. Sonar y variantes son propietarios.",
        "hf": "https://huggingface.co/perplexity-ai"
    },
    "Reka AI": {
        "status": "closed",
        "license": "Propietario",
        "notes": "Reka Flash propietario.",
        "hf": None
    },
    "Snowflake": {
        "status": "open_weight",
        "license": "Apache 2.0",
        "notes": "Arctic Instruct open weight.",
        "hf": "https://huggingface.co/Snowflake"
    },
    "xAI": {
        "status": "closed",
        "license": "Propietario",
        "notes": "Grok series completamente propietario.",
        "hf": None
    },
}

# Función de clasificación para uso en app
def is_open_weight(creator_name: str) -> bool:
    info = CREATOR_LICENSE.get(creator_name, {})
    return info.get("status") in ("open_weight", "open_source")

def is_open_source(creator_name: str) -> bool:
    info = CREATOR_LICENSE.get(creator_name, {})
    return info.get("status") == "open_source"

def get_license_info(creator_name: str) -> dict:
    return CREATOR_LICENSE.get(creator_name, {
        "status": "unknown",
        "license": "Desconocido",
        "notes": "Creator no clasificado — revisar manualmente.",
        "hf": None
    })

# Casos especiales por nombre de modelo (override de creator)
MODEL_OVERRIDES = {
    # Google: separar Gemma (open) de Gemini (cerrado)
    "Gemma": "open_weight",     # cualquier modelo con "Gemma" en el nombre
    "Gemini": "closed",         # cualquier modelo con "Gemini" en el nombre
    # OpenAI
    "gpt-oss": "open_weight",
    "GPT-": "closed",
    "o1": "closed",
    "o3": "closed",
    "o4": "closed",
    # Mistral: los pequeños son open
    "Mistral 7B": "open_weight",
    "Mixtral": "open_weight",
    "Mistral Small": "open_weight",  # Small 2409 en adelante
    # Cohere
    "Command-R": "open_weight",  # CC BY-NC
    "Command A": "closed",
    "Tiny Aya": "open_weight",
    # Perplexity
    "R1 1776": "open_weight",
    "Sonar": "closed",
    # ByteDance
    "Seed-OSS": "open_weight",
    "Doubao": "closed",
}

def classify_model(model_name: str, creator_name: str) -> str:
    """
    Clasifica un modelo como open_weight, open_source, closed o unknown.
    Prioriza overrides por nombre sobre clasificación por creator.
    """
    for keyword, status in MODEL_OVERRIDES.items():
        if keyword.lower() in model_name.lower():
            return status
    return get_license_info(creator_name).get("status", "unknown")

