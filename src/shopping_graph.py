"""
shopping_graph.py
Agente de compras Jumbo Colombia implementado con LangGraph.

Grafo de 3 nodos:
    [START] → [parsear_intencion] → [filtrar_catalogo] → [generar_canasta] → [END]

Uso:
    from src.shopping_graph import build_graph, run_agent
    import pandas as pd

    df = pd.read_csv("data/processed/supermercado_nutricion_clean.csv")
    graph = build_graph(df)
    result = run_agent(graph, "Canasta vegetariana para 2 personas, $80.000 COP")
    print(result["canasta"])
"""

from __future__ import annotations

import json
import os
import re
import html
import unicodedata
import warnings
from typing import TypedDict

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, START, END

warnings.filterwarnings("ignore")
load_dotenv()

# ── Configuración del cliente OpenAI ────────────────────────────────────────

def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY no encontrada. "
            "Crea un archivo .env con OPENAI_API_KEY=sk-... "
            "o expórtala como variable de entorno."
        )
    return OpenAI(api_key=api_key)

MODEL = "gpt-4o-mini"


# ── Estado del grafo ─────────────────────────────────────────────────────────

class ShopperState(TypedDict):
    """
    Estado compartido entre los nodos del grafo.
    Cada nodo lee lo que necesita y escribe su salida.
    """
    user_request : str            # Solicitud original del usuario
    intent       : dict           # Resultado de parse_user_intent (nodo 1)
    productos_df : pd.DataFrame   # Productos filtrados            (nodo 2)
    canasta      : str            # Respuesta final del agente     (nodo 3)
    error        : str | None     # Mensaje de error si algo falla


# ── Utilidades de limpieza (del notebook original) ───────────────────────────

def _limpiar_texto(valor) -> str:
    if pd.isna(valor):
        return valor
    s = str(valor)
    s = html.unescape(s)
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u00b4", "'").replace("\xa0", " ")
    s = re.sub(r"  +", " ", s).strip()
    return s


# ── Nodo 1: Parsear intención del usuario ───────────────────────────────────

def parsear_intencion(state: ShopperState) -> ShopperState:
    """
    Usa GPT-4o mini para extraer parámetros estructurados
    (presupuesto, dieta, restricciones nutricionales) del texto libre.
    """
    client = _get_client()

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
- Si dice "vegano"         → apto_vegano: true Y apto_vegetariano: true
- Si dice "vegetariano"    → apto_vegetariano: true
- Si dice "bajo en azúcar" sin cifra → azucares_g_max: 5
- Si dice "bajo en grasa"  sin cifra → grasas_totales_g_max: 10
- Si dice "alto en proteína" sin cifra → proteina_g_min: 15
- No incluyas texto fuera del JSON.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": state["user_request"]},
            ],
            temperature=0,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        intent = json.loads(response.choices[0].message.content)
        print(f"  [Nodo 1] Intención extraída: {intent.get('contexto', '')}")
        return {**state, "intent": intent, "error": None}

    except Exception as e:
        return {**state, "intent": {}, "error": f"Error en parsear_intencion: {e}"}


# ── Nodo 2: Filtrar catálogo ─────────────────────────────────────────────────

def filtrar_catalogo(state: ShopperState) -> ShopperState:
    """
    Filtra el DataFrame del catálogo según los criterios extraídos
    por el nodo anterior. Devuelve una muestra diversa de hasta 20 productos.
    """
    if state.get("error"):
        return state

    intent = state["intent"]
    df     = state["_catalogo"]          # inyectado al inicializar el grafo
    filt   = df.copy()

    # ── Filtro de presupuesto ─────────────────────────────────────────────
    if intent.get("budget") and "precio_cop" in filt.columns:
        filt = filt[filt["precio_cop"] <= intent["budget"]]

    # ── Filtros dietéticos ───────────────────────────────────────────────
    if intent.get("apto_vegano") and "apto_vegano" in filt.columns:
        filt = filt[filt["apto_vegano"] == True]
    elif intent.get("apto_vegetariano") and "apto_vegetariano" in filt.columns:
        filt = filt[filt["apto_vegetariano"] == True]

    # ── Filtros nutricionales ────────────────────────────────────────────
    nutri_map = {
        "azucares_g_max"       : ("azucares_g",       "max"),
        "grasas_totales_g_max" : ("grasas_totales_g", "max"),
        "proteina_g_min"       : ("proteina_g",        "min"),
        "calorias_kcal_max"    : ("calorias_kcal",     "max"),
        "sodio_mg_max"         : ("sodio_mg",          "max"),
    }
    for intent_key, (col, tipo) in nutri_map.items():
        val = intent.get(intent_key)
        if val is not None and col in filt.columns:
            if tipo == "max":
                filt = filt[filt[col].isna() | (filt[col] <= val)]
            else:
                filt = filt[filt[col].isna() | (filt[col] >= val)]

    if filt.empty:
        msg = (
            "No encontré productos que cumplan todos los criterios. "
            "Te sugiero relajar alguna restricción (presupuesto, dieta o nutrición)."
        )
        print(f"  [Nodo 2] Sin resultados — {msg}")
        return {**state, "productos_df": pd.DataFrame(), "canasta": msg}

    # ── Muestreo diverso por subcategoría ────────────────────────────────
    MAX_PRODUCTS = 20
    if intent.get("budget") and "precio_cop" in filt.columns:
        filt = filt.sort_values("precio_cop")

    n_cats  = filt["subcategoria"].nunique() if "subcategoria" in filt.columns else 1
    per_cat = max(1, MAX_PRODUCTS // max(n_cats, 1))

    if "subcategoria" in filt.columns:
        sample = (
            filt.groupby("subcategoria", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), per_cat), random_state=42))
        )
        if len(sample) < MAX_PRODUCTS and len(filt) > len(sample):
            extra = filt.drop(sample.index).sample(
                min(MAX_PRODUCTS - len(sample), len(filt) - len(sample)),
                random_state=42,
            )
            sample = pd.concat([sample, extra])
    else:
        sample = filt.sample(min(MAX_PRODUCTS, len(filt)), random_state=42)

    sample = sample.head(MAX_PRODUCTS)
    print(f"  [Nodo 2] {len(sample)} productos filtrados en {n_cats} subcategorías")
    return {**state, "productos_df": sample, "error": None}


# ── Nodo 3: Generar canasta ──────────────────────────────────────────────────

def generar_canasta(state: ShopperState) -> ShopperState:
    """
    Usa GPT-4o mini para seleccionar productos del catálogo filtrado
    y generar una canasta personalizada con justificación nutricional.
    """
    # Si ya tiene canasta (sin resultados) o hay error, pasar
    if state.get("canasta") or state.get("error"):
        return state

    client       = _get_client()
    intent       = state["intent"]
    products_df  = state["productos_df"]
    user_request = state["user_request"]

    # ── Restricciones activas ─────────────────────────────────────────────
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
    num_productos = intent.get("num_productos", 8)

    # ── Tabla de productos disponibles ───────────────────────────────────
    cols_display = [
        "nombre", "marca", "subcategoria", "precio_cop",
        "calorias_kcal", "proteina_g", "azucares_g",
        "grasas_totales_g", "fibra_g", "contenido",
    ]
    available = [c for c in cols_display if c in products_df.columns]
    products_str = products_df[available].to_markdown(index=False)

    system_prompt = f"""
Eres un asistente de compras inteligente para el supermercado Jumbo Colombia.
Tu objetivo es armar canastas de productos saludables, económicas y personalizadas.

CONTEXTO: {intent.get('contexto', user_request)}

RESTRICCIONES ACTIVAS:
{restricciones_str}

INSTRUCCIONES:
1. Selecciona exactamente {num_productos} productos de la lista proporcionada.
2. Prioriza variedad de subcategorías.
3. La suma de precios NO debe superar el presupuesto indicado.
   Si no es posible con {num_productos} productos, selecciona los que quepan y explica por qué.
4. Por cada producto: una línea con nombre, precio COP y justificación nutricional breve.
5. Al final muestra:
   - PRECIO TOTAL: suma exacta de los productos seleccionados
   - RESUMEN NUTRICIONAL: calorías promedio, proteína promedio, azúcares promedio
6. Los valores nutricionales son por 100g (sólidos) o 100ml (líquidos).
7. Responde siempre en español.
"""

    user_prompt = f"""
Solicitud: "{user_request}"

Productos disponibles:
{products_str}

Por favor, arma la canasta.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=1500,
        )
        canasta = response.choices[0].message.content
        print(f"  [Nodo 3] Canasta generada ({len(canasta)} caracteres)")
        return {**state, "canasta": canasta, "error": None}

    except Exception as e:
        return {**state, "canasta": "", "error": f"Error en generar_canasta: {e}"}


# ── Constructor del grafo ────────────────────────────────────────────────────

def build_graph(catalogo_df: pd.DataFrame) -> "CompiledGraph":
    """
    Construye y compila el grafo LangGraph del agente de compras.

    Args:
        catalogo_df: DataFrame con el catálogo limpio de productos Jumbo.

    Returns:
        Grafo compilado listo para invocar.

    Ejemplo:
        graph = build_graph(df)
        result = graph.invoke({"user_request": "...", "_catalogo": df, ...})
    """
    # Inyectar el catálogo en los nodos via closure
    def _filtrar(state):
        return filtrar_catalogo({**state, "_catalogo": catalogo_df})

    builder = StateGraph(ShopperState)

    # ── Registrar nodos ───────────────────────────────────────────────────
    builder.add_node("parsear_intencion", parsear_intencion)
    builder.add_node("filtrar_catalogo",  _filtrar)
    builder.add_node("generar_canasta",   generar_canasta)

    # ── Definir flujo lineal ──────────────────────────────────────────────
    builder.add_edge(START,               "parsear_intencion")
    builder.add_edge("parsear_intencion", "filtrar_catalogo")
    builder.add_edge("filtrar_catalogo",  "generar_canasta")
    builder.add_edge("generar_canasta",   END)

    return builder.compile()


# ── Función de entrada principal ─────────────────────────────────────────────

def run_agent(graph, user_request: str, verbose: bool = True) -> dict:
    """
    Ejecuta el agente con una solicitud en lenguaje natural.

    Args:
        graph:        Grafo compilado por build_graph().
        user_request: Solicitud del usuario en español.
        verbose:      Si True, imprime el progreso por nodo.

    Returns:
        dict con keys: user_request, intent, productos_df, canasta, error

    Ejemplo:
        result = run_agent(graph, "Canasta vegana, máximo $60.000 COP")
        print(result["canasta"])
    """
    if verbose:
        print("=" * 60)
        print(f"SHOPPER ASSISTANT AGENT")
        print(f"Solicitud: \"{user_request}\"")
        print("=" * 60)

    initial_state: ShopperState = {
        "user_request" : user_request,
        "intent"       : {},
        "productos_df" : pd.DataFrame(),
        "canasta"      : "",
        "error"        : None,
    }

    result = graph.invoke(initial_state)

    if verbose:
        print("\n" + "=" * 60)
        if result.get("error"):
            print(f"ERROR: {result['error']}")
        else:
            print("CANASTA SUGERIDA:")
            print(result["canasta"])
        print("=" * 60)

    return result
