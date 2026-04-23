# AA Open Source LLM Tracker

Streamlit app que consume la API gratuita de Artificial Analysis para mostrar
un ranking de modelos open source/open weight para generación de golden datasets RAGAS.

## Stack
- **Streamlit** — UI
- **Streamlit Community Cloud** — hosting gratuito
- **AA API** — datos (1000 req/día gratis, caché 24h)

## Despliegue en Streamlit Community Cloud (100% gratis)

### 1. Sube el repo a GitHub
```bash
git init
git add .
git commit -m "init"
git remote add origin https://github.com/TU_USUARIO/aa-llm-tracker.git
git push -u origin main
```

### 2. Despliega en Streamlit Cloud
1. Ve a https://share.streamlit.io
2. Haz login con tu cuenta de GitHub
3. Click "New app"
4. Selecciona tu repo → rama `main` → archivo `app.py`
5. En "Advanced settings" → Secrets, añade:
   ```toml
   AA_API_KEY = "aa_QhLJsejooeSnQwSdQMzdoyKFCCvJfOVC"
   ```
6. Click "Deploy" → URL pública en ~2 minutos

### 3. (Opcional) Usar el secret en vez del input manual
Si usas Streamlit secrets, sustituye en app.py:
```python
api_key = st.secrets.get("AA_API_KEY", "")
```

## Recarga de datos
`@st.cache_data(ttl=86400)` recarga automáticamente cada 24h.
No necesitas cron jobs ni scripts externos — Streamlit lo gestiona.

## Archivos
- `app.py` — aplicación principal
- `license_map.py` — clasificación de licencias por creator (editable)
- `requirements.txt` — dependencias
- `.streamlit/config.toml` — tema y config
