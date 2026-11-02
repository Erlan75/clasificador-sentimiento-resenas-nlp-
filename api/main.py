"""
Servidor Backend API REST para el Análisis de Sentimiento de reseñas en español.

REGLA DE ORO (punto 5.4): este backend NO reimplementa la limpieza ni el entrenamiento.
Usa directamente los pipelines serializados con joblib (limpiador -> TF-IDF -> clasificador)
y solo importa `procesar_paso_a_paso` para mostrar la radiografía NLP en el Frontend.

Ejecución desde la raíz del proyecto:
    uvicorn api.main:app --reload
"""
import os
import sys
import json
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Agregamos la raíz del proyecto al path para poder importar `src` y para que joblib
# encuentre la función `limpiar_lote_textos` usada dentro del pipeline serializado
RUTA_RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if RUTA_RAIZ not in sys.path:
    sys.path.insert(0, RUTA_RAIZ)

from src.nlp.limpieza_texto import procesar_paso_a_paso  # noqa: E402
from api.esquemas import (  # noqa: E402
    SolicitudPrediccion,
    SolicitudLote,
    RespuestaPrediccion,
    RespuestaLote,
    RespuestaEstado,
)

RUTA_MODELOS = os.path.join(RUTA_RAIZ, "src", "models")
RUTA_METRICAS = os.path.join(RUTA_MODELOS, "metricas.json")
RUTA_FRONTEND = os.path.join(RUTA_RAIZ, "frontend")

# Archivos serializados que intentamos cargar al iniciar
ARCHIVOS_MODELOS = {
    "produccion": "modelo_sentimiento.joblib",
    "regresion_logistica": "modelo_logistica.joblib",
    "naive_bayes": "modelo_naive_bayes.joblib",
}

# Estado global de la aplicación (se llena en el arranque)
modelos_cargados: dict = {}
metricas: dict = {}


def cargar_artefactos() -> None:
    """Carga con joblib los pipelines y el archivo metricas.json generados por entrenar.py."""
    modelos_cargados.clear()
    for clave, nombre_archivo in ARCHIVOS_MODELOS.items():
        ruta_archivo = os.path.join(RUTA_MODELOS, nombre_archivo)
        if os.path.exists(ruta_archivo):
            modelos_cargados[clave] = joblib.load(ruta_archivo)

    metricas.clear()
    if os.path.exists(RUTA_METRICAS):
        with open(RUTA_METRICAS, mode="r", encoding="utf-8") as archivo:
            metricas.update(json.load(archivo))


@asynccontextmanager
async def ciclo_de_vida(aplicacion: FastAPI):
    cargar_artefactos()
    yield


app = FastAPI(
    title="API de Análisis de Sentimiento - Reseñas en Español",
    description="Clasifica reseñas de restaurantes y hoteles como 'positivo' o 'negativo'.",
    version="1.0.0",
    lifespan=ciclo_de_vida,
)

# CORS abierto para que frontend/index.html pueda consumir la API sin bloqueos
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# MANEJADOR DE ERRORES DE VALIDACIÓN (mensajes en español)
# ==============================================================================

@app.exception_handler(RequestValidationError)
async def manejar_error_validacion(solicitud: Request, error: RequestValidationError):
    """Devuelve HTTP 422 con un detalle claro en español en lugar del formato técnico de Pydantic."""
    errores_en_espanol = []
    for detalle in error.errors():
        mensaje = str(detalle.get("msg", "Dato inválido."))
        # Pydantic antepone "Value error, " a los ValueError personalizados
        mensaje = mensaje.replace("Value error, ", "")
        campo = ".".join(str(parte) for parte in detalle.get("loc", []) if parte != "body")
        errores_en_espanol.append({"campo": campo or "cuerpo", "mensaje": mensaje})

    return JSONResponse(
        status_code=422,
        content={
            "error": "Solicitud inválida",
            "detalle": errores_en_espanol[0]["mensaje"] if errores_en_espanol else "Dato inválido.",
            "errores": errores_en_espanol,
        },
    )


# ==============================================================================
# FUNCIONES AUXILIARES DE PREDICCIÓN
# ==============================================================================

def obtener_modelo(modelo_elegido: str):
    """Devuelve el pipeline solicitado o lanza HTTP 503 si no está disponible."""
    if not modelos_cargados:
        raise HTTPException(
            status_code=503,
            detail="No hay modelos cargados. Ejecute primero: python src/models/entrenar.py",
        )
    if modelo_elegido not in modelos_cargados:
        raise HTTPException(
            status_code=404,
            detail=f"El modelo '{modelo_elegido}' no está disponible en el servidor.",
        )
    return modelos_cargados[modelo_elegido]


def nombre_legible(modelo_elegido: str) -> str:
    """Traduce la clave interna a un nombre legible para mostrar en el Frontend."""
    if modelo_elegido == "produccion":
        nombre_produccion = metricas.get("modelo_en_produccion", {}).get("nombre", "Modelo en producción")
        return f"{nombre_produccion} (producción)"
    return {"regresion_logistica": "Regresión Logística", "naive_bayes": "Naive Bayes Multinomial"}.get(
        modelo_elegido, modelo_elegido
    )


def clave_real_de_produccion() -> str:
    """Identifica si el modelo de producción corresponde a regresión logística o a naive bayes."""
    nombre = metricas.get("modelo_en_produccion", {}).get("nombre", "")
    return "naive_bayes" if "Naive" in nombre else "regresion_logistica"


def predecir_con_pipeline(modelo, textos: list[str]) -> list[dict]:
    """
    Usa el pipeline serializado completo (limpieza + TF-IDF + clasificador)
    para predecir etiqueta y probabilidades de una lista de textos.
    """
    etiquetas = modelo.predict(textos)
    probabilidades = modelo.predict_proba(textos)
    clases = list(modelo.classes_)
    indice_positivo = clases.index("positivo")
    indice_negativo = clases.index("negativo")

    resultados = []
    for etiqueta, fila_prob in zip(etiquetas, probabilidades):
        prob_positivo = float(fila_prob[indice_positivo])
        prob_negativo = float(fila_prob[indice_negativo])
        resultados.append({
            "sentimiento": str(etiqueta),
            "probabilidad_positivo": round(prob_positivo, 4),
            "probabilidad_negativo": round(prob_negativo, 4),
            "confianza": round(max(prob_positivo, prob_negativo), 4),
        })
    return resultados


# ==============================================================================
# ENDPOINTS
# ==============================================================================

@app.get("/", include_in_schema=False)
def servir_frontend():
    """Sirve la aplicación web (frontend/index.html) en la ruta raíz."""
    ruta_index = os.path.join(RUTA_FRONTEND, "index.html")
    if os.path.exists(ruta_index):
        return FileResponse(ruta_index, media_type="text/html")
    return JSONResponse(
        status_code=404,
        content={"detalle": "No se encontró frontend/index.html. Use /docs para probar la API."},
    )


@app.get("/api", tags=["General"])
def informacion_api():
    """Información general de la API (antes estaba en la raíz '/')."""
    return {
        "mensaje": "API de Análisis de Sentimiento en español activa.",
        "documentacion": "/docs",
        "endpoints": ["/health", "/metrics", "/predict", "/predict/batch"],
    }


@app.get("/health", response_model=RespuestaEstado, tags=["General"])
@app.get("/estado", response_model=RespuestaEstado, tags=["General"], include_in_schema=False)
def estado_salud():
    """Indica si el servicio está activo y si los modelos están cargados."""
    return RespuestaEstado(
        estado="activo",
        modelo_cargado="produccion" in modelos_cargados,
        modelo_en_produccion=metricas.get("modelo_en_produccion", {}).get("nombre"),
        modelos_disponibles=sorted(modelos_cargados.keys()),
        total_resenas_dataset=int(metricas.get("total_datos", 0)),
    )


@app.get("/metrics", tags=["Modelos"])
@app.get("/metricas", tags=["Modelos"], include_in_schema=False)
def obtener_metricas():
    """Devuelve el contenido completo de src/models/metricas.json."""
    if not metricas:
        raise HTTPException(
            status_code=503,
            detail="No se encontró metricas.json. Ejecute primero: python src/models/entrenar.py",
        )
    return metricas


@app.post("/predict", response_model=RespuestaPrediccion, tags=["Predicción"])
@app.post("/predecir", response_model=RespuestaPrediccion, tags=["Predicción"], include_in_schema=False)
def predecir(solicitud: SolicitudPrediccion):
    """Predice el sentimiento de una reseña y devuelve la radiografía NLP paso a paso."""
    modelo_principal = obtener_modelo(solicitud.modelo_elegido)
    resultado_principal = predecir_con_pipeline(modelo_principal, [solicitud.texto])[0]

    # Modelo alternativo: si se usa regresión logística (directa o en producción) comparamos con naive bayes y viceversa
    clave_efectiva = (
        clave_real_de_produccion() if solicitud.modelo_elegido == "produccion" else solicitud.modelo_elegido
    )
    clave_alternativa = "naive_bayes" if clave_efectiva == "regresion_logistica" else "regresion_logistica"

    comparacion_alternativa = None
    if clave_alternativa in modelos_cargados:
        resultado_alternativo = predecir_con_pipeline(modelos_cargados[clave_alternativa], [solicitud.texto])[0]
        comparacion_alternativa = {
            "nombre_modelo": nombre_legible(clave_alternativa),
            "sentimiento": resultado_alternativo["sentimiento"],
            "probabilidad_positivo": resultado_alternativo["probabilidad_positivo"],
            "probabilidad_negativo": resultado_alternativo["probabilidad_negativo"],
            "coincide_con_principal": resultado_alternativo["sentimiento"] == resultado_principal["sentimiento"],
        }

    return {
        "texto": solicitud.texto,
        "sentimiento": resultado_principal["sentimiento"],
        "probabilidad_positivo": resultado_principal["probabilidad_positivo"],
        "probabilidad_negativo": resultado_principal["probabilidad_negativo"],
        "confianza": resultado_principal["confianza"],
        "modelo_usado": nombre_legible(solicitud.modelo_elegido),
        "modelo_alternativo": comparacion_alternativa,
        "paso_a_paso_nlp": procesar_paso_a_paso(solicitud.texto),
    }


@app.post("/predict/batch", response_model=RespuestaLote, tags=["Predicción"])
@app.post("/predecir_lote", response_model=RespuestaLote, tags=["Predicción"], include_in_schema=False)
def predecir_lote(solicitud: SolicitudLote):
    """Predice el sentimiento de varias reseñas y devuelve un resumen del lote."""
    modelo = obtener_modelo(solicitud.modelo_elegido)
    resultados = predecir_con_pipeline(modelo, solicitud.textos)

    predicciones = [{"texto": texto, **resultado} for texto, resultado in zip(solicitud.textos, resultados)]
    cantidad_positivas = sum(1 for p in predicciones if p["sentimiento"] == "positivo")
    total = len(predicciones)

    return {
        "modelo_usado": nombre_legible(solicitud.modelo_elegido),
        "predicciones": predicciones,
        "resumen": {
            "total_resenas": total,
            "cantidad_positivas": cantidad_positivas,
            "cantidad_negativas": total - cantidad_positivas,
            "porcentaje_positivas": round(100 * cantidad_positivas / total, 2),
        },
    }



# ==============================================================================
# ARCHIVOS ESTÁTICOS DEL FRONTEND (estilos.css, app.js)
# ==============================================================================
# Se monta al FINAL para que las rutas de la API tengan prioridad.
# Así index.html puede usar rutas relativas ("estilos.css", "app.js") tanto
# servido desde http://127.0.0.1:8000/ como abierto directamente desde el disco.
if os.path.isdir(RUTA_FRONTEND):
    app.mount("/", StaticFiles(directory=RUTA_FRONTEND), name="frontend")
