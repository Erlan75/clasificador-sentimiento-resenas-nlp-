import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from unidecode import unidecode


def _asegurar_recursos_nltk():
    """Verifica que los recursos básicos de NLTK estén descargados en el entorno."""
    for recurso, ruta in [("stopwords", "corpora/stopwords"), ("punkt", "tokenizers/punkt")]:
        try:
            nltk.data.find(ruta)
        except LookupError:
            try:
                nltk.download(recurso, quiet=True)
            except Exception:
                pass


_asegurar_recursos_nltk()

# ==============================================================================
# 1. LISTA DE PALABRAS VACÍAS (STOP WORDS) ADAPTADA A ANÁLISIS DE SENTIMIENTO
# ==============================================================================
# JUSTIFICACIÓN CRÍTICA:
# En un problema de Análisis de Sentimiento, eliminar palabras de negación como "no",
# "nunca" o "jamás" invierte por completo el significado semántico de una reseña.
# Por ejemplo: "no recomiendo este hotel" pasaría a ser "recomiendo hotel" (falso positivo).
# Del mismo modo, intensificadores como "muy", "más", "poco", "peor" o "mejor" aportan
# carga polar importante cuando se combinan con bigramas.
PALABRAS_CONSERVAR_SENTIMIENTO = {
    "no", "ni", "nunca", "jamas", "sin", "tampoco",
    "nada", "muy", "mas", "poco", "peor", "mejor"
}

try:
    _lista_base_espanol = stopwords.words("spanish")
except LookupError:
    # Lista de respaldo en caso de que el entorno no tenga conexión para descargar nltk.corpus
    _lista_base_espanol = [
        "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un", "para",
        "con", "una", "su", "al", "lo", "como", "más", "pero", "sus", "le", "ya", "o", "este",
        "sí", "porque", "esta", "entre", "cuando", "muy", "sin", "sobre", "también", "me", "hasta",
        "hay", "donde", "quien", "desde", "todo", "nos", "durante", "todos", "uno", "les", "ni",
        "contra", "otros", "ese", "eso", "ante", "ellos", "e", "esto", "mí", "antes", "algunos",
        "qué", "unos", "yo", "otro", "otras", "otra", "él", "tanto", "esa", "estos", "mucho",
        "quienes", "nada", "muchos", "cual", "poco", "ella", "estar", "estas", "algunas", "algo",
        "nosotros", "mi", "mis", "tú", "te", "ti", "tu", "tus", "ellas", "nosotras", "vosotros",
        "vosotras", "os", "mío", "mía", "míos", "mías", "tuyo", "tuya", "tuyos", "tuyas", "suyo",
        "suya", "suyos", "suyas", "nuestro", "nuestra", "nuestros", "nuestras", "vuestro",
        "vuestra", "vuestros", "vuestras", "esos", "esas", "estoy", "estás", "está", "estamos",
        "estáis", "están", "esté", "estés", "estemos", "estéis", "estén", "estaré", "estarás",
        "estará", "estaremos", "estaréis", "estarán", "estaría", "estarías", "estaríamos",
        "estaríais", "estarían", "estaba", "estabas", "estábamos", "estabais", "estaban", "estuve",
        "estuviste", "estuvo", "estuvimos", "estuvisteis", "estuvieron", "hubiera", "fui", "fue",
        "era", "eran", "son", "sea", "sean", "ha", "han", "había", "habían"
    ]

# Pasamos todas las stop words por unidecode para quitar tildes y coincidir con nuestro texto normalizado
PALABRAS_VACIAS_ESPANOL = {
    unidecode(palabra.lower()) for palabra in _lista_base_espanol
} - PALABRAS_CONSERVAR_SENTIMIENTO

# Instancia del radicalizador (Stemmer) en español de NLTK
radicalizador_espanol = SnowballStemmer("spanish")

# Diccionario morfológico amplio del dominio de restaurantes y hoteles para lematización en español
DICCIONARIO_LEMAS_DOMINIO = {
    # Verbos frecuentes conjugados -> infinitivo
    "comimos": "comer", "comiendo": "comer", "comieron": "comer", "comi": "comer",
    "tardaron": "tardar", "tardo": "tardar", "tardaban": "tardar", "tarda": "tardar", "tardar": "tardar",
    "recomendado": "recomendar", "recomiendo": "recomendar", "recomendaron": "recomendar",
    "recomendable": "recomendar", "recomendados": "recomendar", "recomienda": "recomendar",
    "volveria": "volver", "volveremos": "volver", "volvi": "volver", "volvere": "volver", "volver": "volver",
    "gusto": "gustar", "gustaron": "gustar", "gustan": "gustar", "encanto": "encantar", "encantaron": "encantar",
    "estaba": "estar", "estuvieron": "estar", "estan": "estar", "estaban": "estar", "estuvo": "estar",
    "fueron": "ser", "fue": "ser", "era": "ser", "eran": "ser", "son": "ser", "siendo": "ser",
    "llegaron": "llegar", "llego": "llegar", "llegamos": "llegar",
    "pagamos": "pagar", "pague": "pagar", "cobraron": "cobrar", "costaba": "costar",
    "atendieron": "atender", "atendio": "atender", "atienden": "atender",
    "esperamos": "esperar", "esperando": "esperar", "esperar": "esperar",
    "reservamos": "reservar", "resolvio": "resolver", "hicieron": "hacer", "hacia": "hacer",
    "funcionaba": "funcionar", "funciono": "funcionar", "dejaron": "dejar", "dejo": "dejar",
    "pidieron": "pedir", "pedi": "pedir", "dieron": "dar", "dio": "dar", "trajeron": "traer",
    "perdimos": "perder", "perder": "perder", "pierdan": "perder", "pisare": "pisar",
    "supero": "superar", "quedaron": "quedar", "salimos": "salir", "busquen": "buscar",
    "logro": "lograr", "quemar": "quemar", "arruino": "arruinar", "ignoraron": "ignorar",
    "reclame": "reclamar", "quisieron": "querer", "quiso": "querer", "dudaria": "dudar",
    "visitenlos": "visitar", "entren": "entrar", "pueden": "poder", "deberian": "deber",
    # Adjetivos de polaridad y calidad (femeninos, plurales y superlativos -> lema base)
    "excelentes": "excelente", "excelente": "excelente",
    "malisimo": "malo", "malisima": "malo", "malos": "malo", "malas": "malo", "mala": "malo",
    "buenisimo": "bueno", "buenisima": "bueno", "buenos": "bueno", "buenas": "bueno", "buena": "bueno", "buen": "bueno",
    "pesima": "pesimo", "pesimos": "pesimo", "pesimas": "pesimo", "pesimo": "pesimo",
    "sucias": "sucio", "sucios": "sucio", "sucia": "sucio", "sucio": "sucio",
    "limpias": "limpio", "limpios": "limpio", "limpia": "limpio", "limpio": "limpio",
    "impecables": "impecable", "impecable": "impecable",
    "fria": "frio", "frios": "frio", "frias": "frio", "frio": "frio",
    "calientes": "caliente", "caliente": "caliente", "calida": "calido",
    "carisimo": "caro", "carisima": "caro", "caros": "caro", "caras": "caro", "cara": "caro", "caro": "caro",
    "deliciosa": "delicioso", "deliciosos": "delicioso", "deliciosas": "delicioso", "delicioso": "delicioso",
    "rica": "rico", "ricos": "rico", "ricas": "rico", "exquisitos": "exquisito",
    "grosero": "grosero", "grosera": "grosero", "groseros": "grosero",
    "amables": "amable", "amable": "amable", "atentos": "atento", "atenta": "atento",
    "comodisima": "comodo", "comoda": "comodo", "comodas": "comodo", "comodos": "comodo",
    "hermosa": "hermoso", "hermosas": "hermoso", "hermosos": "hermoso",
    "bonita": "bonito", "bonitas": "bonito", "bonitos": "bonito", "linda": "lindo",
    "vacia": "vacio", "vacias": "vacio", "vacios": "vacio",
    "cruda": "crudo", "crudos": "crudo", "crudas": "crudo",
    "salada": "salado", "acida": "acido", "dura": "duro", "rota": "roto",
    "insoportables": "insoportable", "increibles": "increible", "maravillosa": "maravilloso",
    "satisfechos": "satisfecho", "fascinados": "fascinado", "molestos": "molesto",
    "pequena": "pequeno", "pequenas": "pequeno", "minusculas": "minusculo", "diminuto": "diminuto",
    # Sustantivos frecuentes del dominio (plural -> singular)
    "habitaciones": "habitacion",
    "instalaciones": "instalacion",
    "porciones": "porcion",
    "opciones": "opcion",
    "expectativas": "expectativa",
    "estrellas": "estrella",
    "meseros": "mesero", "meseras": "mesero", "mesera": "mesero",
    "platos": "plato",
    "horas": "hora",
    "minutos": "minuto",
    "clientes": "cliente",
    "turistas": "turista",
    "amigos": "amigo",
    "personas": "persona",
    "sabanas": "sabana",
    "manchas": "mancha",
    "cucarachas": "cucaracha",
    "insectos": "insecto",
    "cortinas": "cortina",
    "ventanas": "ventana",
    "postres": "postre",
    "frutas": "fruta",
    "jugos": "jugo",
    "estufas": "estufa",
    "mantas": "manta",
    "ingredientes": "ingrediente",
    "comentarios": "comentario",
    "disculpas": "disculpa",
    "reclamos": "reclamo",
    "fotos": "foto",
    "encargados": "encargado",
}


# ==============================================================================
# 2. FUNCIONES PASO A PASO DEL PIPELINE DE NLP
# ==============================================================================

def normalizar_texto(texto: str) -> str:
    """
    Normaliza un texto en crudo aplicando 8 reglas secuenciales:
    1. Conversión a minúsculas.
    2. Reemplazo de enlaces web por 'url'.
    3. Reemplazo de menciones (@usuario) por 'usuario'.
    4. Reemplazo de montos en bolivianos por 'monto'.
    5. Reemplazo de números sueltos por 'num'.
    6. Eliminación de tildes y diacríticos mediante unidecode.
    7. Eliminación de signos de puntuación y símbolos especiales.
    8. Colapso de espacios múltiples a un único espacio.
    """
    if not isinstance(texto, str):
        texto = str(texto)

    # 1. Convertir todo el texto a minúsculas
    texto_procesado = texto.lower()

    # 2. Reemplazar enlaces web (http://..., https://..., www...) por el token 'url'
    texto_procesado = re.sub(r"https?://\S+|www\.\S+", " url ", texto_procesado)

    # 3. Reemplazar menciones de redes sociales (@usuario) por el token 'usuario'
    texto_procesado = re.sub(r"@\w+", " usuario ", texto_procesado)

    # 4. Reemplazar montos en bolivianos (ej. '500 bolivianos', '500 bs', 'bs. 200') por 'monto'
    texto_procesado = re.sub(
        r"\b\d+\s*(?:bolivianos|boliviano|bs\.?)\b|\bbs\.?\s*\d+\b",
        " monto ",
        texto_procesado
    )

    # 5. Reemplazar otros números o dígitos sueltos por 'num'
    texto_procesado = re.sub(r"\b\d+\b", " num ", texto_procesado)

    # 6. Quitar tildes y caracteres especiales usando unidecode
    texto_procesado = unidecode(texto_procesado)

    # 7. Borrar signos de puntuación, símbolos y emojis dejando solo letras (a-z) y espacios
    texto_procesado = re.sub(r"[^a-z\s]", " ", texto_procesado)

    # 8. Quitar espacios múltiples dejando un solo espacio simple y recortar extremos
    texto_procesado = re.sub(r"\s+", " ", texto_procesado).strip()

    return texto_procesado


def tokenizar_texto(texto: str, metodo: str = "nltk") -> list:
    """
    Divide el texto normalizado en una lista de palabras individuales (tokens).
    Permite elegir entre el tokenizador de NLTK en español o separación simple por espacios.
    """
    if not texto:
        return []

    if metodo == "nltk":
        try:
            return word_tokenize(texto, language="spanish")
        except Exception:
            return texto.split()
    elif metodo == "simple":
        return texto.split()
    else:
        return texto.split()


def quitar_palabras_vacias(lista_tokens: list) -> list:
    """
    Filtra y devuelve únicamente los tokens que no pertenecen a nuestra lista
    personalizada de palabras vacías (conservando negaciones como 'no', 'nunca', etc.).
    """
    return [
        palabra for palabra in lista_tokens
        if palabra not in PALABRAS_VACIAS_ESPANOL and len(palabra) > 1
    ]


def aplicar_stemming(lista_tokens: list) -> list:
    """
    Aplica el algoritmo SnowballStemmer en español para truncar cada palabra
    a su raíz morfológica (por ejemplo: 'recomendado' -> 'recomend').
    """
    return [radicalizador_espanol.stem(palabra) for palabra in lista_tokens]


def _singularizar_palabra_espanol(palabra: str) -> str:
    """
    Reglas heurísticas ligeras de singularización y reducción de sufijos en español
    para palabras no cubiertas directamente en el diccionario de dominio.
    """
    if len(palabra) <= 4:
        return palabra
    if palabra.endswith("ciones"):
        return palabra[:-6] + "cion"
    if palabra.endswith("siones"):
        return palabra[:-6] + "sion"
    if palabra.endswith("isimos") or palabra.endswith("isimas"):
        return palabra[:-6] + "o"
    if palabra.endswith("isimo") or palabra.endswith("isima"):
        return palabra[:-5] + "o"
    if palabra.endswith("mente") and len(palabra) > 7:
        return palabra[:-5]
    if palabra.endswith("ando") or palabra.endswith("iendo"):
        if palabra.endswith("ando"):
            return palabra[:-4] + "ar"
        return palabra[:-5] + "er"
    if palabra.endswith("aron") and len(palabra) > 6:
        return palabra[:-4] + "ar"
    if palabra.endswith("ieron") and len(palabra) > 7:
        return palabra[:-5] + "er"
    return palabra


def aplicar_lematizacion(lista_tokens: list) -> list:
    """
    Transforma cada token a su forma canónica o lema en español utilizando el
    diccionario morfológico del dominio hotelero/gastronómico complementado con
    reglas morfológicas para preservar el significado completo de la palabra.
    """
    lemas_resultado = []
    for palabra in lista_tokens:
        if palabra in DICCIONARIO_LEMAS_DOMINIO:
            lemas_resultado.append(DICCIONARIO_LEMAS_DOMINIO[palabra])
        else:
            lema_heuristico = _singularizar_palabra_espanol(palabra)
            lemas_resultado.append(DICCIONARIO_LEMAS_DOMINIO.get(lema_heuristico, lema_heuristico))
    return lemas_resultado


def limpiar_texto(texto: str, metodo_raiz: str = "lematizacion") -> str:
    """
    Función unificada del pipeline de NLP:
    Ejecuta normalización -> tokenización -> eliminación de palabras vacías ->
    y reducción por lematización (por defecto) o stemming.
    Devuelve el texto limpio unido por espacios simples.
    """
    texto_normalizado = normalizar_texto(texto)
    tokens = tokenizar_texto(texto_normalizado, metodo="nltk")
    tokens_sin_vacias = quitar_palabras_vacias(tokens)

    if metodo_raiz == "stemming":
        tokens_finales = aplicar_stemming(tokens_sin_vacias)
    else:
        tokens_finales = aplicar_lematizacion(tokens_sin_vacias)

    return " ".join(tokens_finales)


def limpiar_lote_textos(lista_textos) -> list:
    """
    Aplica limpiar_texto a una lista, Serie de Pandas o arreglo de textos.
    Definida a nivel de módulo para que FunctionTransformer de scikit-learn
    pueda serializarse y cargarse con joblib sin problemas de pickle.
    """
    return [limpiar_texto(texto, metodo_raiz="lematizacion") for texto in lista_textos]


def procesar_paso_a_paso(texto: str) -> dict:
    """
    Ejecuta la radiografía completa del texto devolviendo un diccionario en español
    con el resultado de cada etapa intermedia (ideal para inspección, API y Frontend).
    """
    texto_normalizado = normalizar_texto(texto)
    tokens = tokenizar_texto(texto_normalizado, metodo="nltk")
    sin_palabras_vacias = quitar_palabras_vacias(tokens)
    palabras_eliminadas = [
        palabra for palabra in tokens if palabra not in sin_palabras_vacias
    ]
    resultado_stemming = aplicar_stemming(sin_palabras_vacias)
    resultado_lematizacion = aplicar_lematizacion(sin_palabras_vacias)

    return {
        "texto_original": texto,
        "texto_normalizado": texto_normalizado,
        "tokens": tokens,
        "sin_palabras_vacias": sin_palabras_vacias,
        "palabras_eliminadas": palabras_eliminadas,
        "resultado_stemming": resultado_stemming,
        "resultado_lematizacion": resultado_lematizacion
    }


if __name__ == "__main__":
    frase_prueba = (
        "¡PÉSIMO SERVICIO en @hotel_central! Nos cobraron 500 bolivianos de más, "
        "tardaron 2 horas y NO recomiendo este lugar para nada: https://hotel-ejemplo.com/reclamo"
    )
    radiografia = procesar_paso_a_paso(frase_prueba)

    print("=" * 80)
    print("PRUEBA AUTOMÁTICA DEL MÓDULO DE LIMPIEZA NLP EN ESPAÑOL")
    print("=" * 80)
    print(f"1. Texto original        : {radiografia['texto_original']}")
    print(f"2. Texto normalizado     : {radiografia['texto_normalizado']}")
    print(f"3. Tokens                : {radiografia['tokens']}")
    print(f"4. Palabras eliminadas   : {radiografia['palabras_eliminadas']}")
    print(f"5. Sin palabras vacías   : {radiografia['sin_palabras_vacias']}")
    print(f"6. Resultado Stemming    : {' '.join(radiografia['resultado_stemming'])}")
    print(f"7. Resultado Lematización: {' '.join(radiografia['resultado_lematizacion'])}")
    print("=" * 80)

