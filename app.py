import streamlit as st
import pandas as pd
import datetime
import json
import os
from license_map import classify_model, get_license_info, CREATOR_LICENSE
from params_lookup import extract_params, format_params

st.set_page_config(
    page_title="Open LLM Tracker",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.top-card {
    border-radius: 10px; padding: 22px 16px;
    text-align: center; border-top: 4px solid; margin-bottom: 4px;
}
.gold   { border-top-color: #F5C518; background: #fffef5; }
.silver { border-top-color: #A8A9AD; background: #f9f9f9; }
.bronze { border-top-color: #CD7F32; background: #fdf8f5; }
.badge  {
    display: inline-flex; align-items: center; justify-content: center;
    border-radius: 8px; color: white; font-weight: 700; line-height: 1;
}
.pill {
    display: inline-block; padding: 2px 9px; border-radius: 12px;
    font-size: 0.74rem; font-weight: 600; color: white; vertical-align: middle;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────
DATA_PATH = "data/models.json"
CACHE_TTL = 60 * 60

LIC_LABEL = {
    "open_source": "Open source",
    "open_weight": "Open weight",
    "mixed":       "Mixed",
    "closed":      "Cerrado",
}
LIC_COLOR = {
    "open_source": "#10B981",
    "open_weight": "#3B82F6",
    "mixed":       "#F59E0B",
    "closed":      "#EF4444",
}

LIC_OPTIONS = ["Open weight", "Open source", "Mixed", "Cerrado"]
LIC_MAP = {
    "Open weight": "open_weight",
    "Open source": "open_source",
    "Mixed":       "mixed",
    "Cerrado":     "closed",
}

BADGE_COLORS = ["#3B82F6","#8B5CF6","#10B981","#F59E0B","#EF4444","#6366F1","#EC4899","#14B8A6","#F97316","#84CC16"]

CREATOR_LOGOS = {
    "Meta":                   "https://github.com/meta-llama.png?size=64",
    "Google":                 "https://github.com/google.png?size=64",
    "Microsoft Azure":        "https://github.com/microsoft.png?size=64",
    "OpenAI":                 "https://github.com/openai.png?size=64",
    "Anthropic":              "https://github.com/anthropics.png?size=64",
    "Mistral":                "https://github.com/mistralai.png?size=64",
    "DeepSeek":               "https://github.com/deepseek-ai.png?size=64",
    "NVIDIA":                 "https://github.com/NVIDIA.png?size=64",
    "IBM":                    "https://github.com/ibm.png?size=64",
    "Cohere":                 "https://github.com/cohere-ai.png?size=64",
    "Alibaba":                "https://github.com/QwenLM.png?size=64",
    "Allen Institute for AI": "https://github.com/allenai.png?size=64",
    "Z AI":                   "https://github.com/THUDM.png?size=64",
    "Amazon":                 "https://github.com/aws.png?size=64",
    "xAI":                    "https://github.com/xai-org.png?size=64",
    "Nous Research":          "https://github.com/NousResearch.png?size=64",
    "Snowflake":              "https://github.com/snowflakedb.png?size=64",
    "Databricks":             "https://github.com/databricks.png?size=64",
    "LG AI Research":         "https://github.com/LG-AI-EXAONE.png?size=64",
    "MiniMax":                "https://github.com/MiniMaxAI.png?size=64",
    "ByteDance Seed":         "https://github.com/bytedance.png?size=64",
    "Kimi":                   "https://github.com/MoonshotAI.png?size=64",
    "TII UAE":                "https://github.com/tiiuae.png?size=64",
    "MBZUAI Institute of Foundation Models": "https://github.com/MBZUAI.png?size=64",
    "Upstage":                "https://github.com/UpstageAI.png?size=64",
    "ServiceNow":             "https://github.com/ServiceNow.png?size=64",
    "Xiaomi":                 "https://github.com/XiaomiMiMo.png?size=64",
    "Arcee AI":               "https://github.com/arcee-ai.png?size=64",
    "Sarvam":                 "https://github.com/sarvamai.png?size=64",
    "Swiss AI Initiative":    "https://github.com/swiss-ai.png?size=64",
    "StepFun":                "https://github.com/stepfun-ai.png?size=64",
    "Perplexity":             "https://github.com/perplexity-ai.png?size=64",
    "AI21 Labs":              "https://github.com/AI21Labs.png?size=64",
    "Naver":                  "https://github.com/naver.png?size=64",
    "Prime Intellect":        "https://github.com/PrimeIntellect-ai.png?size=64",
    "Liquid AI":              "https://github.com/liquid-tech.png?size=64",
    "Deep Cogito":            "https://github.com/deepcogito.png?size=64",
    "Baidu":                  "https://github.com/PaddlePaddle.png?size=64",
    "Reka AI":                "https://github.com/rekacorporation.png?size=64",
    "KwaiKAT":                "https://github.com/kwaikeg.png?size=64",
    "Trillion Labs":          "https://github.com/TrillionAI.png?size=64",
    "InclusionAI":            "https://github.com/InclusionAI.png?size=64",
    "Nanbeige":               "https://github.com/Nanbeige.png?size=64",
}

# Benchmarks que se expresan en % (0-1 en df interno → ×100 para display)
PCT_BENCH = ["ifbench","gpqa","hle","lcr","livecodebench","scicode","tau2","terminalbench","aime25"]


def scale_pct(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    df = df.copy()
    for c in (cols or PCT_BENCH):
        if c in df.columns:
            df[c] = df[c] * 100
    return df


def creator_badge(name: str, size: int = 32) -> str:
    color = BADGE_COLORS[abs(hash(name)) % len(BADGE_COLORS)]
    initials = "".join(w[0] for w in name.split()[:2]).upper()
    fs = round(size * 0.32, 1)
    return f'<span class="badge" style="background:{color};width:{size}px;height:{size}px;font-size:{fs}px">{initials}</span>'


def creator_img(name: str, size: int = 40) -> str:
    url = CREATOR_LOGOS.get(name)
    if url:
        return f'<img src="{url}" width="{size}" height="{size}" style="border-radius:8px;object-fit:cover;vertical-align:middle">'
    return creator_badge(name, size)


def lic_pill(status: str) -> str:
    color = LIC_COLOR.get(status, "#888")
    label = LIC_LABEL.get(status, status)
    return f'<span class="pill" style="background:{color}">{label}</span>'


def pct_col(label: str, max_v: float = 100.0, help: str = "") -> st.column_config.ProgressColumn:
    return st.column_config.ProgressColumn(label, min_value=0, max_value=max_v, format="%.1f%%", help=help)


# ─────────────────────────────────────────────
# GUÍA DE BENCHMARKS
# ─────────────────────────────────────────────
BENCH_GUIDE = [
    {"emoji":"🧠","nombre":"AA Intelligence Index","en_cristiano":"La nota global del modelo",
     "detalle":"Artificial Analysis combina múltiples benchmarks en una sola cifra de 0 a 100. Sirve para comparar rápido sin entrar en detalle. Un modelo con 70 es notablemente mejor que uno con 55 en casi todo.",
     "escala":"0 – 100 · más alto = mejor"},
    {"emoji":"📋","nombre":"IFBench — Instruction Following","en_cristiano":"¿Hace exactamente lo que le pides?",
     "detalle":"Le piden responder solo en JSON, solo con 3 bullets, sin usar ciertas palabras... y se mide si lo cumple al pie de la letra. Crítico cuando el output tiene que ser parseable por código.",
     "escala":"0 – 100% · más alto = más obediente"},
    {"emoji":"🔬","nombre":"GPQA◆ — Graduate-level Science QA","en_cristiano":"¿Sabe ciencia de verdad?",
     "detalle":"Preguntas creadas por doctorandos en física, química y biología que solo los propios expertos pueden responder. No vale memorizar datos, hay que razonar. Un humano de ciencias ronda el 65%.",
     "escala":"0 – 100% · experto humano ≈ 65%"},
    {"emoji":"🎓","nombre":"HLE — Humanity's Last Exam","en_cristiano":"El examen más difícil del mundo",
     "detalle":"Preguntas extremadamente difíciles creadas por académicos de las mejores universidades. Los mejores modelos actuales rondan el 20–35%. Un 5% ya era impresionante hace un año.",
     "escala":"0 – 45%+ · top modelos ≈ 20–35%"},
    {"emoji":"📄","nombre":"AA-LCR — Long Context Reasoning","en_cristiano":"¿Lee y razona sobre documentos largos?",
     "detalle":"El modelo tiene que leer textos de 10.000 a 100.000 tokens y responder preguntas que requieren razonar a través de todo el documento, no solo buscar una frase.",
     "escala":"0 – 100% · más alto = mejor"},
    {"emoji":"💻","nombre":"LiveCodeBench","en_cristiano":"¿Sabe programar de verdad?",
     "detalle":"Problemas reales de competición de código que aparecieron después del corte de entrenamiento, así que el modelo no puede haberlos memorizado. Solo disponible para modelos evaluados recientemente.",
     "escala":"0 – 100% · solo modelos evaluados en este benchmark"},
    {"emoji":"🧪","nombre":"SciCode","en_cristiano":"¿Escribe código científico?",
     "detalle":"Problemas de programación en 16 disciplinas científicas: mecánica cuántica, biología computacional, astrofísica... Mucho más específico que resolver algoritmos de entrevista.",
     "escala":"0 – 100% · más alto = mejor"},
    {"emoji":"🤖","nombre":"τ²-Bench","en_cristiano":"¿Actúa bien como agente autónomo?",
     "detalle":"Simula tickets de soporte de una empresa de telecomunicaciones. El modelo tiene que usar herramientas, leer el contexto y resolver el problema sin ayuda humana en múltiples pasos.",
     "escala":"0 – 100% · más alto = más autónomo"},
    {"emoji":"🖥️","nombre":"TerminalBench (Hard)","en_cristiano":"¿Maneja el terminal como un sysadmin?",
     "detalle":"Tareas reales en terminal Linux: gestión de ficheros, procesamiento de datos, scripts de automatización.",
     "escala":"0 – 100% · más alto = mejor"},
    {"emoji":"🧮","nombre":"AIME 2025","en_cristiano":"¿Hace matemáticas de olimpiada?",
     "detalle":"American Invitational Mathematics Examination 2025. Los problemas de matemáticas más difíciles a nivel preuniversitario. Los mejores modelos ya superan el 90%.",
     "escala":"0 – 100% · más alto = mejor"},
    {"emoji":"💰","nombre":"Precio por millón de tokens","en_cristiano":"¿Cuánto cuesta en producción?",
     "detalle":"Precio blend = media ponderada input/output (ratio 3:1). Un millón de tokens son ~750.000 palabras.",
     "escala":"$/1M tokens · menos = más barato"},
    {"emoji":"⚡","nombre":"Velocidad — Tokens por segundo","en_cristiano":"¿Qué tan rápido responde?",
     "detalle":"Tokens generados por segundo en condiciones reales. 50 tok/s es velocidad de lectura cómoda. Más de 150 tok/s se siente instantáneo.",
     "escala":"tok/s · más alto = más rápido"},
]

# ─────────────────────────────────────────────
# PERFILES
# ─────────────────────────────────────────────
PROFILES = {
    "⭐ Golden Dataset (RAG)": {
        "resumen": "Generar datasets de evaluación",
        "description": "Usas el modelo para crear preguntas, respuestas esperadas y ground-truth a partir de documentos. Lo más importante es que siga instrucciones de formato exactas y que razone bien para generar preguntas difíciles.",
        "use_cases": ["Q&A sobre documentación técnica","expected_answers fieles al contexto","Preguntas multi-hop y normativas","RAGAS testset generation"],
        "weights": {"ifbench":0.35,"gpqa":0.25,"hle":0.15,"lcr":0.15,"price":0.10,"speed":0.00},
        "require": ["ifbench","gpqa","hle"], "color": "#3B8BD4",
    },
    "🤖 Agentes": {
        "resumen": "Sistemas con herramientas y pasos múltiples",
        "description": "El modelo actúa solo: llama a herramientas, consulta APIs, toma decisiones en múltiples pasos. Lo más importante es que resuelva tareas autónomas (τ²-Bench) y siga instrucciones complejas.",
        "use_cases": ["Automatización de flujos","Agentes con herramientas","Planificación multi-paso","Customer support autónomo"],
        "weights": {"ifbench":0.25,"gpqa":0.15,"hle":0.05,"lcr":0.10,"tau2":0.30,"terminalbench":0.05,"price":0.05,"speed":0.05},
        "require": ["ifbench","tau2"], "color": "#7B2FBE",
    },
    "💻 Coding": {
        "resumen": "Generación y revisión de código",
        "description": "Generas código, revisas PRs o depuras bugs. Lo más importante es que resuelva problemas reales de programación (LiveCodeBench) y código científico (SciCode).",
        "use_cases": ["Generación de código","Code review automatizado","Resolución de bugs","Refactoring y documentación"],
        "weights": {"ifbench":0.10,"gpqa":0.10,"hle":0.05,"lcr":0.05,"livecodebench":0.35,"scicode":0.15,"terminalbench":0.10,"price":0.05,"speed":0.05},
        "require": ["livecodebench","scicode"], "color": "#1D9E75",
    },
    "🧮 Razonamiento": {
        "resumen": "Análisis científico y matemáticas",
        "description": "Resolución de problemas complejos, análisis científico o cálculos. Lo más importante es el razonamiento formal (GPQA, HLE) y la capacidad matemática (AIME).",
        "use_cases": ["Análisis científico","Problemas de olimpiada","Razonamiento lógico formal","Investigación y análisis"],
        "weights": {"ifbench":0.10,"gpqa":0.35,"hle":0.25,"lcr":0.10,"aime25":0.15,"price":0.05,"speed":0.00},
        "require": ["gpqa","hle"], "color": "#EF9F27",
    },
    "📄 RAG producción": {
        "resumen": "Chatbot o Q&A sobre documentos propios",
        "description": "El modelo responde preguntas de usuarios finales basándose en documentos largos. Lo más importante es que lea contexto largo bien (LCR), siga instrucciones y responda con latencia aceptable.",
        "use_cases": ["Chatbot sobre base de conocimiento","Q&A sobre documentación interna","Asistente de soporte técnico","Búsqueda aumentada con generación"],
        "weights": {"ifbench":0.25,"gpqa":0.15,"hle":0.15,"lcr":0.30,"price":0.05,"speed":0.10},
        "require": ["ifbench","lcr"], "color": "#D85A30",
    },
    "⚡ Low-cost": {
        "resumen": "Producción de alto volumen, mínimo coste",
        "description": "Quieres el máximo rendimiento al menor precio, para millones de llamadas. El precio y la velocidad pesan mucho más que los benchmarks de calidad.",
        "use_cases": ["Alto volumen de inferencias","Clasificación y extracción masiva","Tareas simples y repetitivas","MVP y prototipos"],
        "weights": {"ifbench":0.20,"gpqa":0.10,"hle":0.05,"lcr":0.05,"price":0.40,"speed":0.20},
        "require": ["ifbench"], "color": "#888",
    },
    "🎛️ Personalizado": {
        "resumen": "Define tus propios pesos",
        "description": "Ajusta manualmente el peso de cada métrica según tus necesidades concretas.",
        "use_cases": [],
        "weights": {"ifbench":0.20,"gpqa":0.20,"hle":0.15,"lcr":0.15,"price":0.15,"speed":0.15},
        "require": [], "color": "#5F5E5A",
    },
}

METRICS_INFO = {
    "ifbench":       ("IFBench",       "Sigue instrucciones de formato exactas"),
    "gpqa":          ("GPQA◆",         "Ciencia de nivel doctorado"),
    "hle":           ("HLE",           "El examen más difícil del mundo"),
    "lcr":           ("LCR",           "Razonamiento sobre documentos 10k-100k tokens"),
    "livecodebench": ("LiveCodeBench", "Código real de competición"),
    "scicode":       ("SciCode",       "Código científico en 16 disciplinas"),
    "tau2":          ("τ²-Bench",      "Agente autónomo resolviendo tickets"),
    "terminalbench": ("TerminalBench", "Tareas en terminal Linux"),
    "aime25":        ("AIME 2025",     "Matemáticas de olimpiada"),
    "price":         ("Precio",        "Menor precio = mayor score"),
    "speed":         ("Velocidad",     "Mayor tok/s = mayor score"),
}

# ─────────────────────────────────────────────
# HELPERS DE FORMATO
# ─────────────────────────────────────────────
def na(v) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v))

def fmt_price(v):
    return "N/A" if na(v) else f"${v:.3f}"

def fmt_num(v, fmt="{:.0f}"):
    return "N/A" if na(v) else fmt.format(v)

def safe_fmt(val, fmt_str):
    try:
        return "N/A" if na(val) else fmt_str.format(val)
    except Exception:
        return "N/A"

# ─────────────────────────────────────────────
# CARGA Y PROCESADO DE DATOS
# ─────────────────────────────────────────────
@st.cache_data(ttl=CACHE_TTL, show_spinner="Cargando datos…")
def fetch_models() -> tuple[pd.DataFrame, str]:
    if not os.path.exists(DATA_PATH):
        st.error("Datos no encontrados. El workflow de GitHub Actions los generará automáticamente.")
        st.stop()
    with open(DATA_PATH, encoding="utf-8") as f:
        payload = json.load(f)
    fetched_at = payload.get("_fetched_at", "")
    rows = []
    for m in payload.get("data", []):
        ev = m.get("evaluations") or {}
        pr = m.get("pricing") or {}
        name    = m["name"]
        creator = m["model_creator"]["name"]

        def ev_val(key):
            v = ev.get(key)
            return None if (v is None or v == 0) else v

        p_total, p_active = extract_params(name)
        rows.append({
            "name":          name,
            "creator":       creator,
            "logo_url":      CREATOR_LOGOS.get(creator),
            "params_b":      p_total,
            "params_str":    format_params(p_total, p_active),
            "release_date":  m.get("release_date", ""),
            "ow_status":     classify_model(name, creator),
            "license":       get_license_info(creator).get("license", "—"),
            "aa_index":      ev.get("artificial_analysis_intelligence_index"),
            "ifbench":       ev_val("ifbench"),
            "gpqa":          ev_val("gpqa"),
            "hle":           ev_val("hle"),
            "lcr":           ev_val("lcr"),
            "livecodebench": ev_val("livecodebench"),
            "scicode":       ev_val("scicode"),
            "tau2":          ev_val("tau2"),
            "terminalbench": ev_val("terminalbench_hard"),
            "aime25":        ev_val("aime_25"),
            "price_blend":   pr.get("price_1m_blended_3_to_1") or None,
            "price_input":   pr.get("price_1m_input_tokens") or None,
            "price_output":  pr.get("price_1m_output_tokens") or None,
            "speed_tps":     m.get("median_output_tokens_per_second") or None,
            "ttft_s":        m.get("median_time_to_first_token_seconds") or None,
        })
    return pd.DataFrame(rows), fetched_at


def compute_score(row: pd.Series, weights: dict) -> float | None:
    NORM = {
        "ifbench":       lambda v: v,
        "gpqa":          lambda v: v,
        "hle":           lambda v: min(v / 0.45, 1.0),
        "lcr":           lambda v: v,
        "livecodebench": lambda v: v,
        "scicode":       lambda v: v,
        "tau2":          lambda v: v,
        "terminalbench": lambda v: v,
        "aime25":        lambda v: v,
        "price":         lambda v: max(0.0, 1.0 - min(v, 8.0) / 8.0) if v > 0 else 0.8,
        "speed":         lambda v: min(v / 300.0, 1.0) if v > 0 else None,
    }
    parts, ws = [], []
    for metric, w in weights.items():
        if w == 0:
            continue
        raw = row.get("price_blend") if metric == "price" else \
              row.get("speed_tps")   if metric == "speed" else \
              row.get(metric)
        if na(raw):
            continue
        nv = NORM[metric](raw) if metric in NORM else None
        if nv is None:
            continue
        parts.append(nv * w)
        ws.append(w)
    if len(ws) < 2 or sum(ws) < 0.3:
        return None
    return round(sum(parts) / sum(ws) * 100, 1)


def model_detail_panel(row: pd.Series):
    logo  = creator_img(row["creator"], 36)
    pill  = lic_pill(row["ow_status"])
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">'
        f'{logo}'
        f'<div>'
        f'<span style="font-size:1.15rem;font-weight:700">{row["name"]}</span>'
        f'<span style="color:#777;font-size:0.85rem"> — {row["creator"]}</span>'
        f'&nbsp;&nbsp;{pill}'
        f'</div></div>',
        unsafe_allow_html=True,
    )
    metrics = [
        ("Params",    row.get("params_str"),     "{}",       "Parámetros totales (MoE = activos entre paréntesis)"),
        ("AA Index",  row.get("aa_index"),       "{:.1f}",   "Nota general de inteligencia (0-100)"),
        ("IFBench",   row.get("ifbench"),        "{:.0%}",   "Sigue instrucciones de formato"),
        ("GPQA◆",     row.get("gpqa"),           "{:.0%}",   "Ciencia nivel doctorado"),
        ("HLE",       row.get("hle"),            "{:.1%}",   "El examen más difícil"),
        ("LCR",       row.get("lcr"),            "{:.0%}",   "Contexto largo 10k-100k tok"),
        ("LiveCode",  row.get("livecodebench"),  "{:.0%}",   "Código de competición"),
        ("SciCode",   row.get("scicode"),        "{:.0%}",   "Código científico"),
        ("τ²-Bench",  row.get("tau2"),           "{:.0%}",   "Agente autónomo"),
        ("Terminal",  row.get("terminalbench"),  "{:.0%}",   "Tareas en terminal"),
        ("AIME 2025", row.get("aime25"),         "{:.0%}",   "Matemáticas olimpiada"),
        ("$/1M",      row.get("price_blend"),    "${:.3f}",  "Precio por millón de tokens"),
        ("Tok/s",     row.get("speed_tps"),      "{:.0f}",   "Velocidad de generación"),
        ("TTFT",      row.get("ttft_s"),         "{:.2f}s",  "Tiempo hasta el primer token"),
    ]
    cols = st.columns(7)
    for i, (lbl, val, fmt, hint) in enumerate(metrics):
        cols[i % 7].metric(lbl, safe_fmt(val, fmt), help=hint)
    st.caption(
        f"Licencia: **{row['license']}** · "
        f"Lanzado: {row['release_date'] or 'N/A'} · "
        f"Precio: {fmt_price(row.get('price_input'))} entrada / {fmt_price(row.get('price_output'))} salida por 1M"
    )


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Filtros")
    st.markdown("**Tipo de licencia**")
    lic_sel = st.multiselect("lic", options=LIC_OPTIONS, default=["Open weight","Open source"],
                             label_visibility="collapsed",
                             help="Open weight = pesos descargables. Open source = además código y datos.")
    st.markdown("**Tamaño del modelo**")
    SIZE_OPTS = ["< 3B", "3–14B", "14–70B", "70–200B", "> 200B", "Sin dato"]
    size_sel = st.multiselect("size", options=SIZE_OPTS, default=SIZE_OPTS,
                              label_visibility="collapsed",
                              help="Filtra por número de parámetros. 'Sin dato' = modelos sin tamaño confirmado públicamente.")
    st.markdown("**Precio máximo** ($/1M tokens blend)")
    max_price = st.slider("price", 0.0, 15.0, 15.0, 0.5, label_visibility="collapsed")
    st.markdown("**Modelos a mostrar en rankings**")
    top_n = st.slider("topn", 5, 50, 25, 5, label_visibility="collapsed")
    st.divider()
    st.caption("Datos: [artificialanalysis.ai](https://artificialanalysis.ai)")

# ─────────────────────────────────────────────
# CARGA
# ─────────────────────────────────────────────
df_all, fetched_at = fetch_models()

with st.sidebar:
    st.metric("Total modelos en BD", len(df_all))
    if fetched_at:
        try:
            st.caption(f"Actualizado: {datetime.datetime.fromisoformat(fetched_at).strftime('%d/%m/%Y %H:%M')} UTC")
        except ValueError:
            pass

# ─────────────────────────────────────────────
# FILTRADO
# ─────────────────────────────────────────────
def _size_ok(p_b, cats):
    if not cats:
        return True
    if p_b is None:
        return "Sin dato" in cats
    if p_b < 3:    return "< 3B" in cats
    if p_b < 14:   return "3–14B" in cats
    if p_b < 70:   return "14–70B" in cats
    if p_b < 200:  return "70–200B" in cats
    return "> 200B" in cats

allowed = [LIC_MAP[l] for l in lic_sel] if lic_sel else list(LIC_MAP.values())
df_base = df_all[df_all["ow_status"].isin(allowed)].copy()
df_base = df_base[df_base["price_blend"].isna() | (df_base["price_blend"] <= max_price)]
df_base = df_base[df_base["params_b"].apply(lambda p: _size_ok(p, size_sel))]

# ─────────────────────────────────────────────
# CABECERA
# ─────────────────────────────────────────────
col_t, col_i = st.columns([3, 2])
with col_t:
    st.title("🏆 Open LLM Tracker")
    st.markdown(
        "Compara modelos de lenguaje con benchmarks reales. "
        "Datos de [Artificial Analysis](https://artificialanalysis.ai), actualizados cada día automáticamente."
    )
try:
    date_str = datetime.datetime.fromisoformat(fetched_at).strftime("%-d de %B de %Y") if fetched_at else "—"
except (ValueError, TypeError):
    date_str = "—"
with col_i:
    st.info(f"📅 Datos del **{date_str}**\n\n{len(df_base)} modelos con los filtros actuales")

st.divider()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_leader, tab_rank, tab_table, tab_guide, tab_lic = st.tabs([
    "🏆 Leaderboard",
    "🎯 Ranking por objetivo",
    "📊 Tabla completa",
    "📖 Qué significa cada dato",
    "📋 Licencias",
])

# ══════════════════════════════════════════════
# TAB 1 — LEADERBOARD
# ══════════════════════════════════════════════
with tab_leader:
    df_lead = (
        df_base[df_base["aa_index"].notna()]
        .sort_values("aa_index", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    if df_lead.empty:
        st.warning("Sin modelos con suficientes datos para los filtros seleccionados.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Modelos en ranking", len(df_lead))
    c2.metric("Mejor AA Index", f"{df_lead['aa_index'].max():.1f}")
    c3.metric("AA Index medio", f"{df_lead['aa_index'].mean():.1f}")
    c4.metric("Empresas / labs", df_lead["creator"].nunique())
    st.caption("Ordenado por **AA Index**. Barras vacías = sin datos para ese benchmark (N/A).")
    st.divider()

    # ── Top 3 medal cards ───────────────────────
    if len(df_lead) >= 3:
        mc1, mc2, mc3 = st.columns(3)
        for col, idx, medal, css in [(mc1,0,"🥇","gold"),(mc2,1,"🥈","silver"),(mc3,2,"🥉","bronze")]:
            r    = df_lead.iloc[idx]
            logo = creator_img(r["creator"], 52)
            pill = lic_pill(r["ow_status"])
            price_line = (
                f"<div style='margin-top:10px;font-size:0.8rem;color:#666;border-top:1px solid #eee;padding-top:8px'>"
                f"💰 {fmt_price(r.get('price_blend'))}/1M &nbsp;·&nbsp; ⚡ {fmt_num(r.get('speed_tps'))} tok/s"
                f"</div>"
            ) if not na(r.get("speed_tps")) else ""
            col.markdown(
                f'<div class="top-card {css}">'
                f'<div style="display:flex;align-items:center;justify-content:center;gap:10px;margin-bottom:10px">'
                f'<span style="font-size:1.8rem">{medal}</span>{logo}</div>'
                f'<div style="font-weight:700;font-size:1.05rem;margin:4px 0 2px">{r["name"]}</div>'
                f'<div style="color:#777;font-size:0.8rem;margin-bottom:8px">{r["creator"]}</div>'
                f'<div style="font-size:2.6rem;font-weight:800;line-height:1;color:#222">{r["aa_index"]:.1f}</div>'
                f'<div style="color:#aaa;font-size:0.68rem;margin-top:2px;margin-bottom:8px">AA Intelligence Index</div>'
                f'{pill}{price_line}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("#### Todos los modelos — haz clic en una fila para ver el detalle completo")

    lead_src_cols = ["logo_url","name","creator","ow_status","params_str","aa_index","ifbench","gpqa","hle","lcr","price_blend","speed_tps"]
    disp_lead = scale_pct(df_lead[lead_src_cols].copy())
    disp_lead["ow_status"] = disp_lead["ow_status"].map(LIC_LABEL).fillna("—")
    disp_lead.insert(0, "#", range(1, len(disp_lead)+1))

    sel_lead = st.dataframe(
        disp_lead,
        column_config={
            "#":           st.column_config.NumberColumn("#", width="small"),
            "logo_url":    st.column_config.ImageColumn("", width="small"),
            "name":        st.column_config.TextColumn("Modelo"),
            "creator":     st.column_config.TextColumn("Empresa / Lab"),
            "ow_status":   st.column_config.TextColumn("Licencia", help="Open source · Open weight · Mixed · Cerrado"),
            "aa_index":    st.column_config.ProgressColumn("AA Index", min_value=0, max_value=100, format="%.1f",
                           help="Nota global 0-100. Combina múltiples benchmarks."),
            "ifbench":     pct_col("IFBench", help="¿Sigue instrucciones de formato? Crítico para outputs parseables."),
            "gpqa":        pct_col("GPQA◆", help="Ciencia de nivel doctorado. Experto humano ≈ 65%."),
            "hle":         pct_col("HLE", max_v=45.0, help="El examen más difícil del mundo. Top modelos ≈ 20-35%."),
            "lcr":         pct_col("LCR", help="Razonamiento sobre documentos largos (10k-100k tokens)."),
            "price_blend": st.column_config.NumberColumn("$/1M", format="$%.3f", help="Precio blend por millón de tokens."),
            "params_str":  st.column_config.TextColumn("Params", help="Parámetros totales del modelo. MoE muestra (activos)."),
            "speed_tps":   st.column_config.NumberColumn("Tok/s", format="%.0f", help="Velocidad de generación."),
        },
        use_container_width=True, hide_index=True,
        selection_mode="single-row", on_select="rerun",
        height=min(38*len(disp_lead)+38, 620),
    )
    if sel_lead.selection.rows:
        st.divider()
        model_detail_panel(df_lead.iloc[sel_lead.selection.rows[0]])


# ══════════════════════════════════════════════
# TAB 2 — RANKING POR OBJETIVO
# ══════════════════════════════════════════════
with tab_rank:
    profile_name = st.selectbox(
        "¿Para qué quieres usar el modelo?",
        list(PROFILES.keys()),
        format_func=lambda x: f"{x}  —  {PROFILES[x]['resumen']}",
    )
    profile = PROFILES[profile_name]
    color   = profile.get("color", "#3B8BD4")

    col_desc, col_w = st.columns([3, 2])
    with col_desc:
        uses = "  ·  ".join(profile["use_cases"])
        st.markdown(
            f"<div style='border-left:4px solid {color};padding:10px 16px;background:#f8f9fa;border-radius:0 8px 8px 0'>"
            f"<span style='font-size:0.9rem;color:#333'>{profile['description']}</span>"
            + (f"<br><span style='font-size:0.78rem;color:#999;margin-top:4px;display:block'>Ej: {uses}</span>" if uses else "")
            + "</div>",
            unsafe_allow_html=True,
        )
    with col_w:
        if profile_name == "🎛️ Personalizado":
            st.markdown("**Ajusta el peso de cada métrica**")
            active_metrics = [k for k in METRICS_INFO if k not in ("price","speed")] + ["price","speed"]
            custom_weights = {}
            wc1, wc2 = st.columns(2)
            for i, metric in enumerate(active_metrics):
                lbl_m, desc = METRICS_INFO[metric]
                val = (wc1 if i%2==0 else wc2).slider(lbl_m, 0, 100, int(profile["weights"].get(metric,0)*100), 5, help=desc)
                custom_weights[metric] = val / 100.0
            total_w = sum(custom_weights.values())
            if total_w == 0:
                st.error("Al menos una métrica debe tener peso > 0"); st.stop()
            weights = {k: v/total_w for k, v in custom_weights.items()}
            require = [k for k, v in weights.items() if v >= 0.10 and k not in ("price","speed")]
        else:
            weights = profile["weights"]
            require = profile["require"]
            st.markdown("**¿Qué pesa más en este perfil?**")
            w_df = pd.DataFrame([
                {"Métrica": METRICS_INFO.get(m,(m,))[0], "Peso": f"{w*100:.0f}%", "Qué mide": METRICS_INFO.get(m,("","—"))[1]}
                for m, w in sorted(weights.items(), key=lambda x: -x[1]) if w > 0
            ])
            st.dataframe(w_df, hide_index=True, use_container_width=True, height=min(len(w_df)*35+38, 260))

    df_rank = df_base.copy()
    df_rank["score"] = df_rank.apply(lambda r: compute_score(r, weights), axis=1)
    for req in require:
        if req in df_rank.columns:
            df_rank = df_rank[df_rank[req].notna()]
    df_rank = df_rank[df_rank["score"].notna()].sort_values("score", ascending=False).head(top_n).reset_index(drop=True)

    if df_rank.empty:
        st.warning("Ningún modelo tiene datos suficientes para este perfil con los filtros actuales."); st.stop()

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Modelos rankeados", len(df_rank))
    c2.metric("Score máximo", df_rank["score"].max())
    c3.metric("Score medio", round(df_rank["score"].mean(), 1))
    c4.metric("Empresas / labs", df_rank["creator"].nunique())
    st.caption("El **Score** (0-100) usa los pesos del perfil. Si falta un benchmark, ese peso se redistribuye. Sin datos suficientes el modelo no aparece.")

    # Top 3 compacto con logo
    if len(df_rank) >= 3:
        tm1, tm2, tm3 = st.columns(3)
        for col, idx, medal in [(tm1,0,"🥇"),(tm2,1,"🥈"),(tm3,2,"🥉")]:
            r    = df_rank.iloc[idx]
            logo = creator_img(r["creator"], 28)
            pill = lic_pill(r["ow_status"])
            col.markdown(
                f'<div style="border:1px solid #e5e7eb;border-radius:8px;padding:10px 12px">'
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">'
                f'<span style="font-size:1.4rem">{medal}</span>{logo}'
                f'<span style="font-weight:700;font-size:0.95rem">{r["name"]}</span></div>'
                f'<div style="color:#555;font-size:0.8rem;margin-bottom:4px">{r["creator"]}</div>'
                f'<div style="display:flex;align-items:center;gap:8px">'
                f'<span style="font-size:1.3rem;font-weight:700">{r["score"]}</span>'
                f'<span style="color:#888;font-size:0.75rem">Score</span>'
                f'&nbsp;{pill}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown(f"#### Top {len(df_rank)}  —  haz clic en una fila para ver el detalle")

    top_metric_keys = [k for k,v in sorted(weights.items(), key=lambda x:-x[1]) if v>0 and k not in ("price","speed")][:4]
    rank_cols = ["logo_url","name","creator","ow_status","params_str","score"] + [k for k in top_metric_keys if k in df_rank.columns] + ["price_blend","speed_tps"]
    disp_rank = scale_pct(df_rank[rank_cols].copy(), [k for k in top_metric_keys if k in PCT_BENCH])
    disp_rank["ow_status"] = disp_rank["ow_status"].map(LIC_LABEL).fillna("—")
    disp_rank.insert(0, "#", range(1, len(disp_rank)+1))

    col_cfg = {
        "#":           st.column_config.NumberColumn("#", width="small"),
        "logo_url":    st.column_config.ImageColumn("", width="small"),
        "name":        st.column_config.TextColumn("Modelo"),
        "creator":     st.column_config.TextColumn("Empresa / Lab"),
        "ow_status":   st.column_config.TextColumn("Licencia"),
        "score":       st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f",
                       help="Nota ponderada según el perfil seleccionado (0-100)."),
        "params_str":  st.column_config.TextColumn("Params", help="Parámetros totales. MoE muestra (activos)."),
        "price_blend": st.column_config.NumberColumn("$/1M", format="$%.3f"),
        "speed_tps":   st.column_config.NumberColumn("Tok/s", format="%.0f"),
    }
    for k in top_metric_keys:
        if k in disp_rank.columns:
            lbl_m, desc = METRICS_INFO.get(k, (k,""))
            max_v = 45.0 if k == "hle" else 100.0
            col_cfg[k] = pct_col(lbl_m, max_v=max_v, help=desc)

    sel_rank = st.dataframe(
        disp_rank, column_config=col_cfg,
        use_container_width=True, hide_index=True,
        selection_mode="single-row", on_select="rerun",
        height=min(38*len(disp_rank)+38, 620),
    )
    if sel_rank.selection.rows:
        st.divider()
        model_detail_panel(df_rank.iloc[sel_rank.selection.rows[0]])


# ══════════════════════════════════════════════
# TAB 3 — TABLA COMPLETA
# ══════════════════════════════════════════════
with tab_table:
    fc1, fc2 = st.columns([2, 3])
    with fc1:
        search = st.text_input("🔍 Buscar modelo o empresa", "")
    with fc2:
        cr_filt = st.multiselect("Filtrar por empresa / lab", sorted(df_base["creator"].unique()))

    df_tbl = df_base.copy()
    df_tbl["score"] = df_tbl.apply(lambda r: compute_score(r, weights), axis=1)
    if search:
        df_tbl = df_tbl[
            df_tbl["name"].str.contains(search, case=False, na=False) |
            df_tbl["creator"].str.contains(search, case=False, na=False)
        ]
    if cr_filt:
        df_tbl = df_tbl[df_tbl["creator"].isin(cr_filt)]
    df_tbl = df_tbl.sort_values("score", ascending=False, na_position="last")

    tbl_cols = ["logo_url","name","creator","ow_status","params_str","score","aa_index",
                "ifbench","gpqa","hle","lcr","tau2","livecodebench","scicode",
                "price_blend","speed_tps","release_date"]
    disp_tbl = scale_pct(df_tbl[tbl_cols].copy())
    disp_tbl["ow_status"] = disp_tbl["ow_status"].map(LIC_LABEL).fillna("—")

    st.dataframe(
        disp_tbl,
        column_config={
            "logo_url":     st.column_config.ImageColumn("", width="small"),
            "name":         st.column_config.TextColumn("Modelo"),
            "creator":      st.column_config.TextColumn("Empresa / Lab"),
            "ow_status":    st.column_config.TextColumn("Licencia"),
            "score":        st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f",
                            help="Nota según el perfil activo en 'Ranking por objetivo'."),
            "aa_index":     st.column_config.ProgressColumn("AA Index", min_value=0, max_value=100, format="%.1f",
                            help="Nota general de Artificial Analysis (0-100)."),
            "ifbench":      pct_col("IFBench",   help="Sigue instrucciones de formato exactas."),
            "gpqa":         pct_col("GPQA◆",     help="Ciencia nivel doctorado."),
            "hle":          pct_col("HLE", max_v=45.0, help="El examen más difícil del mundo."),
            "lcr":          pct_col("LCR",        help="Razonamiento en contexto largo."),
            "tau2":         pct_col("τ²-Bench",   help="Agente autónomo resolviendo tickets."),
            "livecodebench":pct_col("LiveCode",   help="Código real de competición."),
            "scicode":      pct_col("SciCode",    help="Código científico en 16 disciplinas."),
            "price_blend":  st.column_config.NumberColumn("$/1M", format="$%.3f"),
            "params_str":   st.column_config.TextColumn("Params", help="Parámetros totales del modelo. MoE muestra (activos)."),
            "speed_tps":    st.column_config.NumberColumn("Tok/s", format="%.0f"),
            "release_date": st.column_config.TextColumn("Lanzamiento"),
        },
        use_container_width=True, hide_index=True, height=560,
    )
    st.caption(f"{len(disp_tbl)} modelos · Score con perfil: **{profile_name}** · Barras vacías = sin datos (N/A)")
    csv = df_tbl.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar CSV", csv, file_name=f"llm_tracker_{datetime.date.today()}.csv", mime="text/csv")


# ══════════════════════════════════════════════
# TAB 4 — GUÍA DE BENCHMARKS
# ══════════════════════════════════════════════
with tab_guide:
    st.markdown("### ¿Qué significa cada número?")
    st.markdown("Explicación en castellano llano de qué mide cada benchmark y cómo interpretarlo. Sin jerga. Haz clic para expandir.")
    st.divider()
    for b in BENCH_GUIDE:
        with st.expander(f"{b['emoji']} **{b['nombre']}** — *{b['en_cristiano']}*"):
            ca, cb = st.columns([3, 1])
            ca.markdown(b["detalle"])
            cb.markdown(f"**Escala:**\n\n{b['escala']}")
    st.divider()
    st.markdown("### ¿Cómo se calcula el Score de cada perfil?")
    st.markdown("""
El **Score** es una nota de 0 a 100 que resume qué tan bueno es un modelo para un caso de uso concreto:

1. Cada benchmark tiene un **peso** distinto según el perfil (en *Coding* pesa LiveCodeBench; en *Agentes*, τ²-Bench).
2. Cada métrica se **normaliza** a escala 0-1 para que sean comparables entre sí.
3. Si un modelo **no tiene dato** para una métrica, esa métrica se salta y el peso se redistribuye.
4. Si le faltan demasiados datos (menos del 30% del peso cubierto), el modelo **no aparece** en el ranking.
5. El **precio** se invierte (más barato = mayor score). La **velocidad** usa 300 tok/s como referencia máxima.
    """)


# ══════════════════════════════════════════════
# TAB 5 — LICENCIAS
# ══════════════════════════════════════════════
with tab_lic:
    st.markdown("### ¿Puedo usar estos modelos libremente?")
    st.markdown("Depende de la licencia. Antes de usar cualquier modelo en producción, lee siempre la licencia oficial.")
    leg_col, _ = st.columns([2, 3])
    with leg_col:
        for status, label in LIC_LABEL.items():
            color = LIC_COLOR[status]
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">'
                f'<span class="pill" style="background:{color};min-width:90px;text-align:center">{label}</span>'
                f'<span style="font-size:0.88rem;color:#444">'
                + {"open_source":"Pesos + código + datos públicos (OSI).",
                   "open_weight":"Solo pesos descargables, con restricciones comerciales.",
                   "mixed":"Depende del modelo concreto — leer cada uno.",
                   "closed":"Solo vía API. Sin acceso a los pesos."}[status]
                + f'</span></div>',
                unsafe_allow_html=True,
            )
    st.markdown("")
    rows_lic = []
    for creator, info in sorted(CREATOR_LICENSE.items()):
        label = LIC_LABEL.get(info["status"], info["status"])
        rows_lic.append({
            "Empresa / Lab": creator,
            "Tipo":          label,
            "Licencia":      info["license"],
            "Notas":         info["notes"],
            "HuggingFace":   info["hf"] or "—",
        })
    st.dataframe(pd.DataFrame(rows_lic), use_container_width=True, hide_index=True, height=520)
