"""
Pruebas automatizadas del sistema (punto 5.5).
Cubren el módulo de NLP y los endpoints de la API REST.

Ejecución desde la raíz del proyecto:
    python -m pytest -v
"""
import os
import sys

import pytest
from fastapi.testclient import TestClient

RUTA_RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if RUTA_RAIZ not in sys.path:
    sys.path.insert(0, RUTA_RAIZ)

from src.nlp.limpieza_texto import (  # noqa: E402
    normalizar_texto,
    quitar_palabras_vacias,
    aplicar_stemming,
    aplicar_lematizacion,
)
from api.main import app  # noqa: E402


@pytest.fixture(scope="module")
def cliente():
    """Cliente de pruebas; el bloque 'with' ejecuta el arranque (carga de modelos con joblib)."""
    with TestClient(app) as cliente_prueba:
        yield cliente_prueba


# ==============================================================================
# PRUEBAS DEL MÓDULO NLP
# ==============================================================================

def test_normalizacion_quitar_tildes_y_urls():
    """URLs -> 'url', montos en Bs -> 'monto', menciones -> 'usuario' y sin tildes."""
    texto = "Pésima atención en @hotel_central, pagué 500 bolivianos. Ver https://hotel-ejemplo.com"
    resultado = normalizar_texto(texto)

    assert "url" in resultado.split()
    assert "monto" in resultado.split()
    assert "usuario" in resultado.split()
    assert "pesima" in resultado and "atencion" in resultado
    assert "é" not in resultado and "ó" not in resultado
    assert "https" not in resultado and "500" not in resultado


def test_conservacion_palabra_no():
    """La palabra 'no' NO debe eliminarse: invertiría el sentimiento de la reseña."""
    tokens = ["no", "recomiendo", "este", "hotel", "nunca", "de", "la"]
    resultado = quitar_palabras_vacias(tokens)

    assert "no" in resultado
    assert "nunca" in resultado
    # Las conectoras sí deben eliminarse
    assert "de" not in resultado and "la" not in resultado and "este" not in resultado


def test_stemming_vs_lematizacion():
    """Ambas técnicas devuelven listas no vacías y transforman los verbos."""
    tokens = ["tardaron", "recomendado"]
    raices = aplicar_stemming(tokens)
    lemas = aplicar_lematizacion(tokens)

    assert len(raices) == 2 and len(lemas) == 2
    # Stemming recorta la palabra a su raíz
    assert raices[0] != "tardaron" and raices[1] != "recomendado"
    # Lematización devuelve el infinitivo legible
    assert lemas == ["tardar", "recomendar"]


# ==============================================================================
# PRUEBAS DE LA API REST
# ==============================================================================

def test_endpoint_estado_salud(cliente):
    """GET /health responde 200 y confirma que el modelo está cargado."""
    respuesta = cliente.get("/health")
    assert respuesta.status_code == 200

    datos = respuesta.json()
    assert datos["estado"] == "activo"
    assert datos["modelo_cargado"] is True
    assert datos["total_resenas_dataset"] > 0

    # El alias en español debe responder lo mismo
    assert cliente.get("/estado").status_code == 200


def test_endpoint_metricas(cliente):
    """GET /metrics responde 200 e incluye ambos modelos de la escalera."""
    respuesta = cliente.get("/metrics")
    assert respuesta.status_code == 200

    comparacion = respuesta.json()["comparacion_modelos"]
    assert "naive_bayes" in comparacion
    assert "regresion_logistica" in comparacion
    assert "linea_base" in comparacion


def test_prediccion_resena_positiva(cliente):
    """Una reseña claramente positiva debe clasificarse como 'positivo' con su radiografía NLP."""
    texto = "Excelente atención, la comida deliciosa y habitación impecable, muy recomendado"
    respuesta = cliente.post("/predict", json={"texto": texto})
    assert respuesta.status_code == 200

    datos = respuesta.json()
    assert datos["sentimiento"] == "positivo"
    assert datos["probabilidad_positivo"] > datos["probabilidad_negativo"]
    assert "paso_a_paso_nlp" in datos
    assert datos["paso_a_paso_nlp"]["texto_original"] == texto
    assert len(datos["paso_a_paso_nlp"]["tokens"]) > 0


def test_prediccion_resena_negativa_con_no(cliente):
    """Una reseña negativa con la negación 'no' debe clasificarse como 'negativo'."""
    texto = "No recomiendo nada este lugar, pésimo servicio y comida fría"
    respuesta = cliente.post("/predict", json={"texto": texto})
    assert respuesta.status_code == 200

    datos = respuesta.json()
    assert datos["sentimiento"] == "negativo"
    # La negación debe sobrevivir al pipeline de limpieza
    assert "no" in datos["paso_a_paso_nlp"]["sin_palabras_vacias"]


def test_rechazo_texto_vacio_o_invalido(cliente):
    """Textos vacíos o sin palabras reales deben rechazarse con 400 o 422 y mensaje en español."""
    respuesta_vacia = cliente.post("/predict", json={"texto": "   "})
    assert respuesta_vacia.status_code in (400, 422)
    assert "vacío" in respuesta_vacia.json()["detalle"]

    respuesta_simbolos = cliente.post("/predict", json={"texto": "123 !!! ???"})
    assert respuesta_simbolos.status_code in (400, 422)


def test_prediccion_por_lote(cliente):
    """POST /predict/batch devuelve una predicción por reseña y el resumen del lote."""
    textos = [
        "Excelente atención y comida deliciosa, volveremos pronto",
        "Pésimo servicio, la habitación estaba sucia, no recomiendo",
    ]
    respuesta = cliente.post("/predict/batch", json={"textos": textos})
    assert respuesta.status_code == 200

    datos = respuesta.json()
    assert len(datos["predicciones"]) == 2
    assert datos["resumen"]["total_resenas"] == 2
    assert datos["resumen"]["cantidad_positivas"] + datos["resumen"]["cantidad_negativas"] == 2



def test_raiz_sirve_frontend(cliente):
    """GET / sirve el HTML del Frontend y sus archivos estáticos (estilos.css, app.js)."""
    respuesta = cliente.get("/")
    assert respuesta.status_code == 200
    assert "text/html" in respuesta.headers["content-type"]
    assert "Clasificador de Sentimiento de Reseñas en Español" in respuesta.text

    assert cliente.get("/estilos.css").status_code == 200
    assert cliente.get("/app.js").status_code == 200
