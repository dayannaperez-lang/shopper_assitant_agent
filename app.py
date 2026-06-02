"""
app.py — Shopper Assistant Agent · Gradio UI
Ejecutar: python app.py
"""

import os
import re
import json
import html
import warnings
import unicodedata

import gradio as gr
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

warnings.filterwarnings("ignore")
load_dotenv()

# ── Cliente OpenAI ───────────────────────────────────────────────────────────

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL  = "gpt-4o-mini"

# ── Cargar catálogo ──────────────────────────────────────────────────────────

DATA_PATH = "data/processed/supermercado_nutricion.json"

try:
    df_agente = pd.read_json(DATA_PATH)
    print(f"Catálogo cargado: {len(df_agente):,} productos")
except FileNotFoundError:
    df_agente = pd.DataFrame()
    print(f"No se encontró {DATA_PATH}. Ejecuta primero 01_scraping_y_datos.ipynb")


# ── Lógica del agente (3 etapas) ─────────────────────────────────────────────

def parse_user_intent(user_request: str) -> dict:
    system_prompt = """
Eres un extractor de información de solicitudes de compras en supermercado.
A partir del texto del usuario, extrae ÚNICAMENTE los parámetros mencionados.

Devuelve SIEMPRE un JSON válido con exactamente estas claves (usa null si no se menciona):
{
  "budget": <número en COP o null>,
  "apto_vegetariano": <true/false/null>,
  "apto_vegano": <true/false/null>,
  "azucares_g_max": <número o null>,
  "grasas_totales_g_max": <número o null>,
  "proteina_g_min": <número o null>,
  "calorias_kcal_max": <número o null>,
  "sodio_mg_max": <número o null>,
  "num_productos": <entero entre 3 y 15, por defecto 8>,
  "contexto": <resumen breve de la solicitud en español, máx 60 palabras>
}

Reglas:
- Si dice "vegano"          → apto_vegano: true  Y apto_vegetariano: true
- Si dice "vegetariano"     → apto_vegetariano: true
- Si dice "bajo en azúcar"  sin cifra → azucares_g_max: 5
- Si dice "bajo en grasa"   sin cifra → grasas_totales_g_max: 10
- Si dice "alto en proteína" sin cifra → proteina_g_min: 15
- No incluyas texto fuera del JSON.
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_request},
        ],
        temperature=0,
        max_tokens=300,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def filter_catalog(intent: dict, max_products: int = 20) -> pd.DataFrame:
    if df_agente.empty:
        return pd.DataFrame()

    filt = df_agente.copy()

    if intent.get("budget") and "precio_cop" in filt.columns:
        filt = filt[filt["precio_cop"] <= intent["budget"]]

    if intent.get("apto_vegano") and "apto_vegano" in filt.columns:
        filt = filt[filt["apto_vegano"] == True]
    elif intent.get("apto_vegetariano") and "apto_vegetariano" in filt.columns:
        filt = filt[filt["apto_vegetariano"] == True]

    nutri_map = {
        "azucares_g_max"       : ("azucares_g",        "max"),
        "grasas_totales_g_max" : ("grasas_totales_g",  "max"),
        "proteina_g_min"       : ("proteina_g",         "min"),
        "calorias_kcal_max"    : ("calorias_kcal",      "max"),
        "sodio_mg_max"         : ("sodio_mg",           "max"),
    }
    for intent_key, (col, tipo) in nutri_map.items():
        val = intent.get(intent_key)
        if val is not None and col in filt.columns:
            mask = filt[col].isna() | (filt[col] <= val if tipo == "max" else filt[col] >= val)
            filt  = filt[mask]

    if filt.empty:
        return filt

    if intent.get("budget") and "precio_cop" in filt.columns:
        filt = filt.sort_values("precio_cop")

    if "subcategoria" in filt.columns:
        n_cats  = filt["subcategoria"].nunique()
        per_cat = max(1, max_products // max(n_cats, 1))
        sample  = (
            filt.groupby("subcategoria", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), per_cat), random_state=42))
        )
        if len(sample) < max_products:
            extra  = filt.drop(sample.index).sample(
                min(max_products - len(sample), len(filt) - len(sample)), random_state=42
            )
            sample = pd.concat([sample, extra])
        return sample.head(max_products)

    return filt.sample(min(max_products, len(filt)), random_state=42)


def generate_basket(user_request: str, intent: dict, products_df: pd.DataFrame) -> str:
    restricciones = []
    if intent.get("budget"):
        restricciones.append(f"Presupuesto máximo: ${intent['budget']:,.0f} COP")
    if intent.get("apto_vegano"):
        restricciones.append("Dieta: VEGANA")
    elif intent.get("apto_vegetariano"):
        restricciones.append("Dieta: VEGETARIANA")
    for k, label in [
        ("azucares_g_max",      "Azúcares ≤ {} g/100g"),
        ("grasas_totales_g_max","Grasas totales ≤ {} g/100g"),
        ("proteina_g_min",      "Proteína ≥ {} g/100g"),
        ("calorias_kcal_max",   "Calorías ≤ {} kcal/100g"),
        ("sodio_mg_max",        "Sodio ≤ {} mg/100g"),
    ]:
        if intent.get(k):
            restricciones.append(label.format(intent[k]))

    restricciones_str = (
        "\n".join(f"  • {r}" for r in restricciones)
        if restricciones else "  • Sin restricciones especiales"
    )

    cols_display = [
        "nombre", "marca", "subcategoria", "precio_cop",
        "calorias_kcal", "proteina_g", "azucares_g",
        "grasas_totales_g", "fibra_g", "contenido",
    ]
    available    = [c for c in cols_display if c in products_df.columns]
    products_str = products_df[available].to_markdown(index=False)

    system_prompt = f"""
Eres un asistente de compras inteligente para el supermercado Jumbo Colombia.
Tu objetivo es armar canastas de productos saludables, económicas y personalizadas.

CONTEXTO: {intent.get('contexto', user_request)}

RESTRICCIONES ACTIVAS:
{restricciones_str}

INSTRUCCIONES:
1. Selecciona exactamente {intent.get('num_productos', 8)} productos de la lista.
2. Prioriza variedad de subcategorías.
3. La suma de precios NO debe superar el presupuesto indicado.
4. Por cada producto: nombre, precio COP y una justificación nutricional breve.
5. Al final incluye:
   - PRECIO TOTAL: suma exacta
   - RESUMEN NUTRICIONAL: calorías, proteína y azúcares promedio
6. Responde siempre en español.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f'Solicitud: "{user_request}"\n\nProductos disponibles:\n{products_str}'},
        ],
        temperature=0.7,
        max_tokens=1500,
    )
    return response.choices[0].message.content


# ── Función principal que llama Gradio ───────────────────────────────────────

def shopping_agent(user_request: str) -> tuple[str, str]:
    """
    Retorna (canasta_markdown, debug_info) para mostrarse en la UI.
    """
    if not user_request.strip():
        return "Por favor escribe tu solicitud.", ""

    if df_agente.empty:
        return (
            f"Catálogo no encontrado en `{DATA_PATH}`.\n\n"
            "Ejecuta primero `01_scraping_y_datos.ipynb` para generar los datos.",
            "",
        )

    try:
        # Etapa 1
        intent = parse_user_intent(user_request)

        # Etapa 2
        products_df = filter_catalog(intent)
        if products_df.empty:
            return (
                "😕 No encontré productos con esos criterios.\n\n"
                "Intenta relajar el presupuesto o las restricciones nutricionales.",
                json.dumps(intent, ensure_ascii=False, indent=2),
            )

        # Etapa 3
        canasta = generate_basket(user_request, intent, products_df)

        debug = (
            f"**Intención extraída:**\n```json\n{json.dumps(intent, ensure_ascii=False, indent=2)}\n```\n\n"
            f"**Productos disponibles para el agente:** {len(products_df)}"
        )
        return canasta, debug

    except Exception as e:
        return f"Error: {e}", ""


# ── Interfaz Gradio ──────────────────────────────────────────────────────────

EJEMPLOS = [
    ["Canasta vegetariana para 2 personas, presupuesto $80.000 COP"],
    ["Desayuno saludable bajo en azúcar para una persona"],
    ["Opciones veganas, sin lácteos ni huevo, máximo $60.000 COP"],
    ["Canasta alta en proteína para deportista, $100.000 COP"],
    ["Productos sin gluten y bajos en sodio"],
]

with gr.Blocks(
    title="🛒 Shopper Assistant — Jumbo Colombia",
    theme=gr.themes.Soft(primary_hue="green"),
) as demo:

    gr.Markdown(
        """
        # 🛒 Shopper Assistant — retail 
        Cuéntame qué necesitas en lenguaje natural y armo tu canasta personalizada
        según tu presupuesto, preferencias dietéticas y restricciones nutricionales.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            txt_input = gr.Textbox(
                label="¿Qué necesitas?",
                placeholder="Ej: Canasta vegetariana para 2 personas, presupuesto $80.000 COP, baja en azúcar...",
                lines=3,
            )
            btn = gr.Button("🧺 Generar canasta", variant="primary")
            gr.Examples(
                examples=EJEMPLOS,
                inputs=txt_input,
                label="Ejemplos",
            )

        with gr.Column(scale=4):
            out_canasta = gr.Markdown(label="Canasta sugerida")
            with gr.Accordion("🔍 Detalle del proceso", open=False):
                out_debug = gr.Markdown()

    btn.click(
        fn=shopping_agent,
        inputs=txt_input,
        outputs=[out_canasta, out_debug],
    )
    txt_input.submit(
        fn=shopping_agent,
        inputs=txt_input,
        outputs=[out_canasta, out_debug],
    )

if __name__ == "__main__":
    demo.launch(share=False)
