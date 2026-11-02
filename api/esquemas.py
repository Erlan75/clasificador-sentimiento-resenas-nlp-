"""
Esquemas de validación (Pydantic) en español para la API de Análisis de Sentimiento.
Definen la forma exacta de las solicitudes y respuestas de cada endpoint.
"""
import re
from typing import Optional
from pydantic import BaseModel, Field, field_validator

# Modelos que el usuario puede elegir al predecir
MODELOS_PERMITIDOS = {"produccion", "regresion_logistica", "naive_bayes"}

LONGITUD_MINIMA = 3
LONGITUD_MAXIMA = 2000


def validar_texto_resena(texto: str) -> str:
    """
    Reglas de validación compartidas para cualquier reseña recibida por la API.
    Se usa tanto en predicción individual como en predicción por lote.
    """
    if texto is None or not texto.strip():
        raise ValueError("El texto de la reseña no puede estar vacío ni contener solo espacios.")

    texto_recortado = texto.strip()

    if len(texto_recortado) < LONGITUD_MINIMA:
        raise ValueError(
            f"El texto de la reseña es demasiado corto: debe tener al menos {LONGITUD_MINIMA} caracteres reales."
        )
    if len(texto_recortado) > LONGITUD_MAXIMA:
        raise ValueError(
            f"El texto de la reseña es demasiado largo: el máximo permitido es {LONGITUD_MAXIMA} caracteres "
            f"(se recibieron {len(texto_recortado)})."
        )

    # Contamos palabras que contienen al menos 2 letras reales (incluye tildes y ñ)
    palabras_con_letras = re.findall(r"[a-zA-ZáéíóúÁÉÍÓÚñÑüÜ]{2,}", texto_recortado)
    if len(palabras_con_letras) < 2:
        raise ValueError(
            "Se requiere una frase válida en español con al menos 2 palabras reales "
            "(no solo números, símbolos o emojis)."
        )

    return texto_recortado


# ==============================================================================
# ESQUEMAS DE SOLICITUD
# ==============================================================================

class SolicitudPrediccion(BaseModel):
    """Solicitud para predecir el sentimiento de una sola reseña."""
    texto: str = Field(..., description="Reseña en español a analizar.",
                       examples=["Excelente atención, la comida deliciosa, muy recomendado"])
    modelo_elegido: str = Field("produccion",
                                description="Modelo a usar: 'produccion', 'regresion_logistica' o 'naive_bayes'.")

    @field_validator("texto")
    @classmethod
    def validar_texto(cls, valor: str) -> str:
        return validar_texto_resena(valor)

    @field_validator("modelo_elegido")
    @classmethod
    def validar_modelo_elegido(cls, valor: str) -> str:
        valor_normalizado = (valor or "produccion").strip().lower()
        if valor_normalizado not in MODELOS_PERMITIDOS:
            raise ValueError(
                f"Modelo '{valor}' no reconocido. Opciones válidas: {sorted(MODELOS_PERMITIDOS)}."
            )
        return valor_normalizado


class SolicitudLote(BaseModel):
    """Solicitud para predecir el sentimiento de varias reseñas de golpe (1 a 50)."""
    textos: list[str] = Field(..., min_length=1, max_length=50,
                              description="Lista de reseñas en español (mínimo 1, máximo 50).")
    modelo_elegido: str = Field("produccion",
                                description="Modelo a usar: 'produccion', 'regresion_logistica' o 'naive_bayes'.")

    @field_validator("textos")
    @classmethod
    def validar_textos(cls, valores: list[str]) -> list[str]:
        textos_validos = []
        for posicion, texto in enumerate(valores, start=1):
            try:
                textos_validos.append(validar_texto_resena(texto))
            except ValueError as error:
                raise ValueError(f"Reseña número {posicion}: {error}")
        return textos_validos

    @field_validator("modelo_elegido")
    @classmethod
    def validar_modelo_elegido(cls, valor: str) -> str:
        return SolicitudPrediccion.validar_modelo_elegido(valor)


# ==============================================================================
# ESQUEMAS DE RESPUESTA
# ==============================================================================

class PasoAPasoNLP(BaseModel):
    """Radiografía del pipeline de NLP (misma estructura que procesar_paso_a_paso)."""
    texto_original: str
    texto_normalizado: str
    tokens: list[str]
    sin_palabras_vacias: list[str]
    palabras_eliminadas: list[str]
    resultado_stemming: list[str]
    resultado_lematizacion: list[str]


class ComparacionModeloAlternativo(BaseModel):
    """Predicción del otro modelo para ver si coinciden o discrepan."""
    nombre_modelo: str
    sentimiento: str
    probabilidad_positivo: float
    probabilidad_negativo: float
    coincide_con_principal: bool


class RespuestaPrediccion(BaseModel):
    """Respuesta completa de POST /predict."""
    texto: str
    sentimiento: str = Field(..., description="'positivo' o 'negativo'.")
    probabilidad_positivo: float
    probabilidad_negativo: float
    confianza: float = Field(..., description="Probabilidad de la clase predicha (0 a 1).")
    modelo_usado: str
    modelo_alternativo: Optional[ComparacionModeloAlternativo] = None
    paso_a_paso_nlp: PasoAPasoNLP


class PrediccionIndividualLote(BaseModel):
    """Resultado de una reseña dentro de un lote."""
    texto: str
    sentimiento: str
    probabilidad_positivo: float
    probabilidad_negativo: float
    confianza: float


class ResumenLote(BaseModel):
    total_resenas: int
    cantidad_positivas: int
    cantidad_negativas: int
    porcentaje_positivas: float


class RespuestaLote(BaseModel):
    """Respuesta de POST /predict/batch."""
    modelo_usado: str
    predicciones: list[PrediccionIndividualLote]
    resumen: ResumenLote


class RespuestaEstado(BaseModel):
    """Respuesta de GET /health."""
    estado: str
    modelo_cargado: bool
    modelo_en_produccion: Optional[str] = None
    modelos_disponibles: list[str]
    total_resenas_dataset: int
