"""
Clase para extraer el catálogo de productos de Jumbo Colombia via API VTEX.
"""

import re
import time
import requests
import pandas as pd
from tqdm import tqdm


class WebScrappingRetail:
    """
    Extrae productos del catálogo de Jumbo Colombia (VTEX).

    Uso:
        scraper = WebScrappingRetail(max_por_categoria=100)
        productos = scraper.run()
        df = pd.DataFrame(productos)
    """

    def __init__(self, max_por_categoria: int = None):
        self.BASE_URL = "https://www.jumbocolombia.com"
        self.HEADERS = {
            "User-Agent"     : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept"         : "application/json, text/plain, */*",
            "Accept-Language": "es-CO,es;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer"        : "https://www.jumbocolombia.com/",
            "sec-fetch-dest" : "empty",
            "sec-fetch-mode" : "cors",
            "sec-fetch-site" : "same-origin",
        }
        self.max_por_categoria = max_por_categoria

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _extraer_contenido(self, texto: str):
        """Extrae contenido/tamaño del nombre o descripción del producto via regex."""
        texto = str(texto)
        patrones = [
            r'\d+[\.,]?\d*\s*(?:kg|kgs|gramos|gr(?![a-z])|g(?=[^a-z]|$))',
            r'\d+[\.,]?\d*\s*(?:litros?|lt|ml|cc)',
            r'(?:x|pack\s*x?)\s*\d+|\d+\s*(?:unidades?|und?s?|piezas?)',
            r'\d+[\.,]?\d*\s*(?:tazas?|puestos?|personas?)',
            r'\d+\s*[xX]\s*\d+(?:\s*[xX]\s*\d+)?\s*(?:cm|mm|m(?=[^l]))?',
        ]
        for patron in patrones:
            match = re.search(patron, texto, re.IGNORECASE)
            if match:
                return match.group(0).strip()
        return None

    def _extraer_especificaciones(self, item: dict):
        """Busca contenido/tamaño en atributos estructurados VTEX del producto."""
        specs_dict = {}
        for grupo in item.get("specificationGroups", []):
            for spec in grupo.get("specifications", []):
                nombre = spec.get("name", "")
                valores = spec.get("values", [])
                if nombre and valores:
                    specs_dict[nombre.lower()] = valores[0]

        claves = [
            "contenido", "capacidad", "peso neto", "peso", "volumen",
            "tamaño", "medida", "cantidad", "presentacion", "presentación",
            "gramaje", "litros", "mililitros",
        ]
        for clave in claves:
            if clave in specs_dict:
                return specs_dict[clave]
        return None

    def _get_categorias(self) -> list:
        """Obtiene el árbol de categorías de la API VTEX."""
        arbol = requests.get(
            f"{self.BASE_URL}/api/catalog_system/pub/category/tree/3",
            headers=self.HEADERS,
            timeout=20,
        ).json()

        hojas = []
        for dep in arbol:
            for cat in dep.get("children", []):
                subcats = cat.get("children", [])
                targets = subcats if subcats else [cat]
                for sub in targets:
                    slug = sub.get("url", "").replace(self.BASE_URL, "").strip("/")
                    if slug:
                        hojas.append({
                            "slug"        : slug,
                            "departamento": dep["name"],
                            "categoria"   : cat["name"],
                            "subcategoria": sub["name"],
                        })
        print(f"  {len(hojas)} subcategorías encontradas")
        return hojas

    # ── Método principal ─────────────────────────────────────────────────────

    def run(self) -> list:
        """
        Ejecuta el scraping completo del catálogo de Jumbo.

        Returns:
            list[dict]: Lista de productos con nombre, precio, categoría,
                        marca, contenido y descripción.
        """
        hojas   = self._get_categorias()
        productos = []

        for hoja in tqdm(hojas, desc="Scraping categorías"):
            desde = 0
            while True:
                if self.max_por_categoria and desde >= self.max_por_categoria:
                    break
                try:
                    r = requests.get(
                        f"{self.BASE_URL}/api/catalog_system/pub/products/search/{hoja['slug']}",
                        headers=self.HEADERS,
                        params={"_from": desde, "_to": desde + 49},
                        timeout=20,
                    )
                    if r.status_code == 429:
                        tqdm.write(f"  Rate limit en '{hoja['subcategoria']}' — esperando 15s...")
                        time.sleep(15)
                        continue
                    if r.status_code not in (200, 206):
                        break

                    data = r.json()
                    if not isinstance(data, list) or len(data) == 0:
                        break

                    for item in data:
                        precio = None
                        try:
                            precio = item["items"][0]["sellers"][0]["commertialOffer"]["Price"]
                        except (IndexError, KeyError, TypeError):
                            pass

                        descripcion = re.sub(r"<[^>]+>", " ", str(item.get("description", "")))
                        descripcion = re.sub(r"\s+", " ", descripcion).strip()[:500]

                        contenido = self._extraer_especificaciones(item)
                        if not contenido:
                            contenido = self._extraer_contenido(item.get("productName", ""))
                        if not contenido:
                            contenido = self._extraer_contenido(descripcion)

                        productos.append({
                            "nombre"      : item.get("productName", "").strip(),
                            "departamento": hoja["departamento"],
                            "categoria"   : hoja["categoria"],
                            "subcategoria": hoja["subcategoria"],
                            "marca"       : item.get("brand", "").strip(),
                            "contenido"   : contenido,
                            "precio"      : precio,
                            "descripcion" : descripcion,
                        })

                    if len(data) < 50:
                        break
                    desde += 50
                    time.sleep(0.5)

                except Exception as e:
                    tqdm.write(f"  Error en '{hoja['subcategoria']}': {e}")
                    break

            time.sleep(2)

        print(f"\n  Total productos extraídos: {len(productos):,}")
        return productos
