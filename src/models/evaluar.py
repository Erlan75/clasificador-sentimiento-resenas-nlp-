import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)


def calcular_metricas_modelo(
    nombre_modelo: str,
    y_real,
    y_predicho,
    tiempo_entrenamiento: float = 0.0
) -> dict:
    """
    Calcula las métricas de clasificación en español para un modelo evaluado:
    - exactitud (accuracy)
    - f1_macro (métrica principal de la escalera de modelos)
    - precision_macro
    - exhaustividad_recall_macro
    - matriz_confusion y su desglose (VN, FP, FN, VP)
    """
    exactitud = float(accuracy_score(y_real, y_predicho))
    f1_macro = float(f1_score(y_real, y_predicho, average="macro", zero_division=0))
    precision_macro = float(precision_score(y_real, y_predicho, average="macro", zero_division=0))
    exhaustividad_recall_macro = float(recall_score(y_real, y_predicho, average="macro", zero_division=0))

    matriz = confusion_matrix(y_real, y_predicho, labels=["negativo", "positivo"])
    verdaderos_negativos = int(matriz[0, 0])
    falsos_positivos = int(matriz[0, 1])
    falsos_negativos = int(matriz[1, 0])
    verdaderos_positivos = int(matriz[1, 1])

    return {
        "nombre_modelo": nombre_modelo,
        "exactitud": round(exactitud, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_macro": round(precision_macro, 4),
        "exhaustividad_recall_macro": round(exhaustividad_recall_macro, 4),
        "tiempo_entrenamiento_segundos": round(float(tiempo_entrenamiento), 4),
        "matriz_confusion": matriz.tolist(),
        "detalle_matriz_confusion": {
            "verdaderos_negativos": verdaderos_negativos,
            "falsos_positivos": falsos_positivos,
            "falsos_negativos": falsos_negativos,
            "verdaderos_positivos": verdaderos_positivos
        }
    }


def _generar_explicacion_error(texto_original: str, etiqueta_real: str, etiqueta_predicha: str) -> str:
    """
    Analiza lingüísticamente la reseña en español para explicar de forma didáctica
    por qué el clasificador estadístico se equivocó.
    """
    texto_min = texto_original.lower()

    conectores_contraste = ["pero", "sin embargo", "aunque", "lástima", "lastima", "aun así", "aun asi", "menos mal"]
    tiene_contraste = any(conector in texto_min for conector in conectores_contraste)

    indicadores_ironia = ["¡bravo", "bravo por", "¡qué maravilla", "que maravilla", "gracias a", "muy linda la página"]
    tiene_ironia = any(ironia in texto_min for ironia in indicadores_ironia)

    if tiene_ironia:
        return (
            f"Ironía o sarcasmo detectado: La reseña utiliza palabras superficialmente positivas "
            f"(como agradecimientos o exclamaciones de elogio) para expresar una queja real ({etiqueta_real}), "
            f"lo que indujo al modelo a clasificarla como '{etiqueta_predicha}'."
        )

    if tiene_contraste:
        return (
            f"Sentimientos mezclados con conector adversativo ('pero'/'aunque'/'sin embargo'): "
            f"La reseña combina elogios y críticas en la misma oración. El peso de los términos secundarios "
            f"inclinó la frontera de decisión hacia '{etiqueta_predicha}' cuando el balance global era '{etiqueta_real}'."
        )

    if "no " in texto_min or "nunca" in texto_min:
        return (
            f"Estructura de negación compleja: Aunque el pipeline preserva el token 'no', la presencia "
            f"de adjetivos de polaridad opuesta en otra cláusula desplazó la probabilidad hacia '{etiqueta_predicha}'."
        )

    return (
        f"Ambigüedad léxica del dominio: El texto contiene vocabulario descriptivo con pesos TF-IDF "
        f"asociados frecuentemente a la clase '{etiqueta_predicha}', superando la señal de '{etiqueta_real}'."
    )


def extraer_errores_comentados(
    textos_originales,
    textos_limpios,
    y_real,
    y_predicho,
    probabilidades=None,
    cantidad: int = 5
) -> list:
    """
    Identifica los casos del conjunto de prueba donde y_real != y_predicho
    y devuelve una lista de diccionarios con el análisis detallado de cada fallo.
    Si el modelo comete menos de 'cantidad' errores estrictos, incluye también
    los casos de mayor incertidumbre (probabilidad cercana a 0.50) para enriquecer
    el análisis pedagógico.
    """
    lista_originales = list(textos_originales)
    lista_limpios = list(textos_limpios)
    lista_real = list(y_real)
    lista_predicho = list(y_predicho)

    errores_encontrados = []
    indices_usados = set()

    for indice in range(len(lista_real)):
        if lista_real[indice] != lista_predicho[indice]:
            confianza = 0.5
            if probabilidades is not None:
                confianza = float(np.max(probabilidades[indice]))

            explicacion = _generar_explicacion_error(
                lista_originales[indice],
                lista_real[indice],
                lista_predicho[indice]
            )

            errores_encontrados.append({
                "texto_original": lista_originales[indice],
                "texto_limpio": lista_limpios[indice],
                "etiqueta_real": lista_real[indice],
                "etiqueta_predicha": lista_predicho[indice],
                "probabilidad_confianza": round(confianza, 4),
                "explicacion_error": explicacion
            })
            indices_usados.add(indice)

            if len(errores_encontrados) >= cantidad:
                break

    # Si hubo menos de 'cantidad' errores directos, complementamos con los casos de menor confianza (frontera de duda)
    if len(errores_encontrados) < cantidad and probabilidades is not None:
        confianzas_maximas = [float(np.max(prob)) for prob in probabilidades]
        indices_ordenados_por_duda = np.argsort(confianzas_maximas)

        for indice in indices_ordenados_por_duda:
            if indice in indices_usados:
                continue
            confianza = confianzas_maximas[indice]
            etiqueta_opuesta = "negativo" if lista_predicho[indice] == "positivo" else "positivo"
            explicacion = (
                f"Caso límite de alta incertidumbre (confianza {round(confianza * 100, 1)}%): "
                + _generar_explicacion_error(lista_originales[indice], lista_real[indice], etiqueta_opuesta)
            )
            errores_encontrados.append({
                "texto_original": lista_originales[indice],
                "texto_limpio": lista_limpios[indice],
                "etiqueta_real": lista_real[indice],
                "etiqueta_predicha": lista_predicho[indice],
                "probabilidad_confianza": round(confianza, 4),
                "explicacion_error": explicacion
            })
            indices_usados.add(indice)
            if len(errores_encontrados) >= cantidad:
                break

    return errores_encontrados


def extraer_palabras_mas_importantes(
    vectorizador,
    modelo_logistica,
    modelo_nb=None,
    top_n: int = 12
) -> dict:
    """
    Extrae del vectorizador TF-IDF y de los coeficientes de Regresión Logística
    las 'top_n' palabras o bigramas que más empujan hacia 'positivo' y hacia 'negativo'.
    Además extrae los términos con mayor peso logarítmico en Naive Bayes Multinomial.
    """
    nombres_caracteristicas = vectorizador.get_feature_names_out()
    coeficientes = modelo_logistica.coef_[0]

    # En scikit-learn con clases ['negativo', 'positivo'], coef > 0 impulsa a 'positivo' y coef < 0 a 'negativo'
    indices_positivos = np.argsort(coeficientes)[-top_n:][::-1]
    indices_negativos = np.argsort(coeficientes)[:top_n]

    top_positivas_logistica = [
        {"termino": str(nombres_caracteristicas[i]), "peso": round(float(coeficientes[i]), 3)}
        for i in indices_positivos
    ]
    top_negativas_logistica = [
        {"termino": str(nombres_caracteristicas[i]), "peso": round(float(coeficientes[i]), 3)}
        for i in indices_negativos
    ]

    resultado = {
        "regresion_logistica": {
            "top_positivas": top_positivas_logistica,
            "top_negativas": top_negativas_logistica
        }
    }

    if modelo_nb is not None and hasattr(modelo_nb, "feature_log_prob_"):
        # Diferencia de log-probabilidades entre clase positiva (índice 1) y negativa (índice 0)
        log_prob_negativo = modelo_nb.feature_log_prob_[0]
        log_prob_positivo = modelo_nb.feature_log_prob_[1]
        diferencia_log_nb = log_prob_positivo - log_prob_negativo

        idx_nb_pos = np.argsort(diferencia_log_nb)[-top_n:][::-1]
        idx_nb_neg = np.argsort(diferencia_log_nb)[:top_n]

        resultado["naive_bayes"] = {
            "top_positivas": [
                {
                    "termino": str(nombres_caracteristicas[i]),
                    "log_probabilidad": round(float(log_prob_positivo[i]), 3),
                    "diferencia_log": round(float(diferencia_log_nb[i]), 3)
                }
                for i in idx_nb_pos
            ],
            "top_negativas": [
                {
                    "termino": str(nombres_caracteristicas[i]),
                    "log_probabilidad": round(float(log_prob_negativo[i]), 3),
                    "diferencia_log": round(float(diferencia_log_nb[i]), 3)
                }
                for i in idx_nb_neg
            ]
        }

    return resultado

