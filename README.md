# Shopper Assistant Agent — Retail ejemplo

Agente inteligente de compras que ayuda a los usuarios a armar canastas personalizadas de un supermercado de retai, basándose en lenguaje natural, restricciones presupuestarias, preferencias alimenticias e información nutricional.

## ¿Qué hace?

El agente recibe una solicitud como:

> *"Arma una canasta vegetariana para 2 personas con presupuesto de $80.000 COP, baja en azúcar"*

Y devuelve una canasta de productos del catálogo del retail filtrada por precio, categoría y perfil nutricional, con justificación de cada selección.

### Pipeline del agente

```
Solicitud del usuario (lenguaje natural)
        ↓
  parse_user_intent()        ← GPT extrae: presupuesto, dieta, restricciones
        ↓
  filter_catalog()           ← Filtra el catálogo según la intención
        ↓
  generate_basket()          ← GPT genera la canasta con justificación
        ↓
Canasta personalizada con precio estimado
```

---

## Estructura del proyecto

```
shopper-assistant-agent/
├── notebooks/
│   ├── 01_scraping_y_datos.ipynb    # Web scraping + limpieza del catálogo 
│   ├── 02_agente_llama.ipynb        # Prueba inicial con LLaMA local
│   └── 03_agente_openai.ipynb       # Agente final con OpenAI GPT-4o mini
├── src/
│   ├── __init__.py
│   └── web_scrapping_retail.py      # Clase reutilizable de scraping (VTEX API)
├── data/
│   ├── raw/                         # Datos crudos del scraping (no versionados)
│   └── processed/                   # Datos listos para el agente (no versionados)
├── outputs/                         # Resultados de pruebas del agente
├── .env.example                     # Plantilla de variables de entorno
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Notebooks

| # | Notebook | Descripción |
|---|----------|-------------|
| 1 | `01_scraping_y_datos.ipynb` | Extrae el catálogo completo de un retail via API VTEX. Limpia y filtra productos relevantes para alimentación. Genera el dataset base. |
| 2 | `02_agente_llama.ipynb` | **Prueba inicial** con LLaMA corriendo localmente (llama-cpp-python). Permite comparar el alcance y calidad de respuesta vs. la versión con OpenAI. |
| 3 | `03_agente_openai.ipynb` | **Versión final del agente.** Incluye EDA, limpieza avanzada, arquitectura de 3 etapas (parseo → filtrado → generación) con GPT-4o mini, y 5 ejemplos de uso. |

> 💡 Se decidió mantener el notebook con LLaMA (`02`) como referencia de comparación, para evidenciar las diferencias en capacidad de razonamiento nutricional y comprensión de lenguaje natural entre un modelo local y un modelo de frontier.

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/shopper-assistant-agent.git
cd shopper-assistant-agent

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar API Key
cp .env.example .env
# Editar .env y agregar tu OPENAI_API_KEY
```

---

## Variables de entorno

Crea un archivo `.env` en la raíz del proyecto:

```env
OPENAI_API_KEY=sk-...tu-clave-aquí...
```

---

## Uso de la clase de scraping

```python
from src.web_scrapping_retail import WebScrappingRetail
import pandas as pd

scraper = WebScrappingRetail(max_por_categoria=100)
productos = scraper.run()
df = pd.DataFrame(productos)
df.to_csv('data/raw/jumbo_productos.csv', index=False)
```

---

## Ejemplos del agente

```python
# Canasta vegetariana con presupuesto
shopping_agent("Quiero una canasta vegetariana para 2 personas, presupuesto $80.000 COP")

# Desayuno saludable bajo en azúcar
shopping_agent("Arma un desayuno saludable bajo en azúcar para una persona")

# Canasta vegana estricta
shopping_agent("Necesito opciones veganas, sin lácteos ni huevo")

# Restricciones combinadas
shopping_agent("Sin gluten, sin sodio alto, máximo $50.000 COP")
```

---

## Tecnologías

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o_mini-green)
![LLaMA](https://img.shields.io/badge/LLaMA-local-orange)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)

---

## Autores

Proyecto desarrollado en el marco de la Especialización en Analítica y Ciencia de Datos — Universidad de Antioquia, 2026.

Elaborado por: 

Paola Andrea Gomez
Dayanna pérez

---


### Arquitectura

```
[START] → [parsear_intencion] → [filtrar_catalogo] → [generar_canasta] → [END]
```

| Nodo | Responsabilidad |
|------|-----------------|
| `parsear_intencion` | GPT extrae presupuesto, dieta y restricciones nutricionales del texto libre |
| `filtrar_catalogo` | Filtra el DataFrame por precio, dieta y nutrición; muestrea productos diversos por subcategoría |
| `generar_canasta` | GPT selecciona productos y genera la canasta con justificación nutricional y precio total |

### Uso directo

```python
import pandas as pd
from src.shopping_graph import build_graph, run_agent

df    = pd.read_csv("data/processed/supermercado_nutricion_clean.csv")
graph = build_graph(df)

result = run_agent(graph, "Canasta vegana para 2 personas, máximo $80.000 COP")
print(result["canasta"])
```

Ver el notebook `04_agente_langgraph.ipynb` para ejemplos completos.
