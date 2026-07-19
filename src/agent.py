import os
import json
import warnings
import pandas as pd
from typing import TypedDict, Optional
from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, END

warnings.filterwarnings("ignore")
load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"

DATA_PATH = "data/processed/supermercado_nutricion.json"

try:
    df_agente = pd.read_json(DATA_PATH)
    print(f"Catálogo cargado: {len(df_agente):,} productos")
except FileNotFoundError:
    df_agente = pd.DataFrame()
    print(f"No se encontró {DATA_PATH}.")


# ── Estado compartido entre nodos ─────────────────────────────────────────────

class ShopperState(TypedDict):
    user_request: str
    history: list[dict]        # memoria conversacional
    intent: Optional[dict]
    products_df: Optional[object]  # pd.DataFrame
    basket: Optional[str]
    error: Optional[str]


# ── Nodo 1: Parsear intención ─────────────────────────────────────────────────

def parse_intent(state: ShopperState) -> ShopperState:
    """Extrae presupuesto, dieta y restricciones del mensaje del usuario."""

    # Construimos el historial como contexto para el LLM
    history_text = ""
    if state["history"]:
        prev = state["history"][-3:]  # últimas 3 interacciones
        history_text = "\n".join(
            f"Usuario: {h['user']}\nCanasta generada: {h['basket'][:200]}..."
            for h in prev
        )

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
- Si dice "vegano" → apto_vegano: true Y apto_vegetariano: true
- Si dice "vegetariano" → apto_vegetariano: true
- Si dice "bajo en azúcar" sin cifra → azucares_g_max: 5
- Si dice "bajo en grasa" sin cifra → grasas_totales_g_max: 10
- Si dice "alto en proteína" sin cifra → proteina_g_min: 15
- No incluyas texto fuera del JSON.
"""
    user_content = state["user_request"]
    if history_text:
        user_content = f"Historial previo:\n{history_text}\n\nNueva solicitud: {state['user_request']}"

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        intent = json.loads(response.choices[0].message.content)
        return {**state, "intent": intent, "error": None}
    except Exception as e:
        return {**state, "intent": None, "error": f"Error al parsear intención: {e}"}


# ── Nodo 2: Filtrar catálogo ──────────────────────────────────────────────────

def filter_catalog(state: ShopperState) -> ShopperState:
    """Filtra el catálogo según la intención extraída."""

    if state.get("error") or state.get("intent") is None:
        return state

    if df_agente.empty:
        return {**state, "error": f"Catálogo no encontrado en `{DATA_PATH}`."}

    intent = state["intent"]
    filt = df_agente.copy()

    if intent.get("budget") and "precio_cop" in filt.columns:
        filt = filt[filt["precio_cop"] <= intent["budget"]]

    if intent.get("apto_vegano") and "apto_vegano" in filt.columns:
        filt = filt[filt["apto_vegano"] == True]
    elif intent.get("apto_vegetariano") and "apto_vegetariano" in filt.columns:
        filt = filt[filt["apto_vegetariano"] == True]

    nutri_map = {
        "azucares_g_max":       ("azucares_g",       "max"),
        "grasas_totales_g_max": ("grasas_totales_g", "max"),
        "proteina_g_min":       ("proteina_g",       "min"),
        "calorias_kcal_max":    ("calorias_kcal",    "max"),
        "sodio_mg_max":         ("sodio_mg",         "max"),
    }
    for intent_key, (col, tipo) in nutri_map.items():
        val = intent.get(intent_key)
        if val is not None and col in filt.columns:
            mask = filt[col].isna() | (
                filt[col] <= val if tipo == "max" else filt[col] >= val
            )
            filt = filt[mask]

    max_products = 20
    if not filt.empty:
        if intent.get("budget") and "precio_cop" in filt.columns:
            filt = filt.sort_values("precio_cop")
        if "subcategoria" in filt.columns:
            n_cats = filt["subcategoria"].nunique()
            per_cat = max(1, max_products // max(n_cats, 1))
            sample = (
                filt.groupby("subcategoria", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), per_cat), random_state=42))
            )
            if len(sample) < max_products:
                extra = filt.drop(sample.index).sample(
                    min(max_products - len(sample), len(filt) - len(sample)),
                    random_state=42,
                )
                sample = pd.concat([sample, extra])
            filt = sample.head(max_products)
        else:
            filt = filt.sample(min(max_products, len(filt)), random_state=42)

    return {**state, "products_df": filt}


# ── Nodo 3: Generar canasta ───────────────────────────────────────────────────

def generate_basket(state: ShopperState) -> ShopperState:
    """Genera la canasta personalizada con GPT."""

    if state.get("error"):
        return state

    intent = state["intent"]
    products_df = state["products_df"]

    restricciones = []
    if intent.get("budget"):
        restricciones.append(f"Presupuesto máximo: ${intent['budget']:,.0f} COP")
    if intent.get("apto_vegano"):
        restricciones.append("Dieta: VEGANA")
    elif intent.get("apto_vegetariano"):
        restricciones.append("Dieta: VEGETARIANA")
    for k, label in [
        ("azucares_g_max",       "Azúcares ≤ {} g/100g"),
        ("grasas_totales_g_max", "Grasas totales ≤ {} g/100g"),
        ("proteina_g_min",       "Proteína ≥ {} g/100g"),
        ("calorias_kcal_max",    "Calorías ≤ {} kcal/100g"),
        ("sodio_mg_max",         "Sodio ≤ {} mg/100g"),
    ]:
        if intent.get(k):
            restricciones.append(label.format(intent[k]))

    restricciones_str = (
        "\n".join(f" • {r}" for r in restricciones)
        if restricciones else " • Sin restricciones especiales"
    )

    cols_display = [
        "nombre", "marca", "subcategoria", "precio_cop",
        "calorias_kcal", "proteina_g", "azucares_g",
        "grasas_totales_g", "fibra_g", "contenido",
    ]
    available = [c for c in cols_display if c in products_df.columns]
    products_str = products_df[available].to_markdown(index=False)

    # Incluimos historial en el prompt para memoria conversacional
    history_context = ""
    if state["history"]:
        prev = state["history"][-2:]
        history_context = "\n\nCONTEXTO DE CONVERSACIÓN PREVIA:\n" + "\n".join(
            f"- El usuario pidió antes: '{h['user']}'" for h in prev
        )

    system_prompt = f"""
Eres un asistente de compras inteligente para el supermercado Jumbo Colombia.
Tu objetivo es armar canastas de productos saludables, económicas y personalizadas.

CONTEXTO: {intent.get('contexto', state['user_request'])}{history_context}

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

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f'Solicitud: "{state["user_request"]}"\n\nProductos disponibles:\n{products_str}'},
            ],
            temperature=0.7,
            max_tokens=1500,
        )
        basket = response.choices[0].message.content

        # Guardamos en el historial
        new_history = state["history"] + [
            {"user": state["user_request"], "basket": basket}
        ]

        return {**state, "basket": basket, "history": new_history}
    except Exception as e:
        return {**state, "error": f"Error al generar canasta: {e}"}


# ── Nodo de fallback ──────────────────────────────────────────────────────────

def fallback(state: ShopperState) -> ShopperState:
    """Respuesta amable cuando no hay productos o hay error."""

    if state.get("error"):
        message = f"⚠️ {state['error']}"
    else:
        message = (
            "😕 No encontré productos con esos criterios.\n\n"
            "Intenta relajar el presupuesto o las restricciones nutricionales.\n\n"
            "**Sugerencias:**\n"
            "- Aumenta el presupuesto un 20%\n"
            "- Elimina alguna restricción nutricional\n"
            "- Prueba con categorías más amplias"
        )
    return {**state, "basket": message}


# ── Router: decide si hay productos suficientes ───────────────────────────────

def should_generate(state: ShopperState) -> str:
    if state.get("error"):
        return "fallback"
    products_df = state.get("products_df")
    if products_df is None or (hasattr(products_df, "empty") and products_df.empty):
        return "fallback"
    return "generate_basket"


# ── Construcción del grafo ────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(ShopperState)

    graph.add_node("parse_intent",    parse_intent)
    graph.add_node("filter_catalog",  filter_catalog)
    graph.add_node("generate_basket", generate_basket)
    graph.add_node("fallback",        fallback)

    graph.set_entry_point("parse_intent")
    graph.add_edge("parse_intent", "filter_catalog")
    graph.add_conditional_edges(
        "filter_catalog",
        should_generate,
        {
            "generate_basket": "generate_basket",
            "fallback":        "fallback",
        },
    )
    graph.add_edge("generate_basket", END)
    graph.add_edge("fallback",        END)

    return graph.compile()


# Instancia global del grafo
shopper_graph = build_graph()


# ── Función pública para usar desde app.py ────────────────────────────────────

def run_agent(user_request: str, history: list = None) -> tuple[str, list, str]:
    """
    Ejecuta el grafo y retorna (basket, updated_history, debug_info).
    
    Args:
        user_request: solicitud del usuario en lenguaje natural
        history: lista de interacciones previas (para memoria)
    
    Returns:
        basket: respuesta con la canasta generada
        updated_history: historial actualizado
        debug: info de depuración (intención + productos disponibles)
    """
    if history is None:
        history = []

    initial_state: ShopperState = {
        "user_request": user_request,
        "history":      history,
        "intent":       None,
        "products_df":  None,
        "basket":       None,
        "error":        None,
    }

    result = shopper_graph.invoke(initial_state)

    basket = result.get("basket", "No se pudo generar la canasta.")
    updated_history = result.get("history", history)

    intent = result.get("intent") or {}
    products_df = result.get("products_df")
    n_products = len(products_df) if products_df is not None and not products_df.empty else 0

    debug = (
        f"**Intención extraída:**\n```json\n{json.dumps(intent, ensure_ascii=False, indent=2)}\n```\n\n"
        f"**Productos disponibles para el agente:** {n_products}\n\n"
        f"**Nodos ejecutados:** parse_intent → filter_catalog → "
        f"{'generate_basket' if not result.get('error') and n_products > 0 else 'fallback'}"
    )

    return basket, updated_history, debug