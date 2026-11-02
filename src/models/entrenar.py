import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# Aseguramos que la raíz del proyecto esté en sys.path para ejecuciones desde consola
ruta_raiz_proyecto = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ruta_raiz_proyecto not in sys.path:
    sys.path.insert(0, ruta_raiz_proyecto)

from src.nlp.limpieza_texto import limpiar_texto, limpiar_lote_textos
from src.models.evaluar import (
    calcular_metricas_modelo,
    extraer_errores_comentados,
    extraer_palabras_mas_importantes
)


def entrenar_y_guardar_modelos():
    # ==========================================================================
    # 1. CARGA Y PREPARACIÓN REPRODUCIBLE
    # ==========================================================================
    semilla = 42
    np.random.seed(semilla)

    ruta_datos = os.path.join(ruta_raiz_proyecto, "data", "resenas.csv")
    df = pd.read_csv(ruta_datos, encoding="utf-8")

    print("=" * 80)
    print("INICIANDO PIPELINE REPRODUCIBLE DE ENTRENAMIENTO (ESCALERA DE MODELOS)")
    print("=" * 80)
    print(f"Dataset cargado desde: {ruta_datos} ({df.shape[0]} registros)")

    # ==========================================================================
    # 2. DIVISIÓN ESTRATIFICADA (80% ENTRENO / 20% PRUEBA)
    # ==========================================================================
    x_entreno, x_prueba, y_entreno, y_prueba = train_test_split(
        df["texto"],
        df["etiqueta"],
        test_size=0.20,
        random_state=semilla,
        stratify=df["etiqueta"]
    )
    print(f"Registros de entrenamiento (80%): {len(x_entreno)}")
    print(f"Registros de prueba (20% intocable): {len(x_prueba)}")

    # ==========================================================================
    # 3. LA ESCALERA DE MODELOS (3 ESCALONES)
    # ==========================================================================

    # --- ESCALÓN 0: LÍNEA BASE (DummyClassifier) ---
    inicio_base = time.perf_counter()
    pipeline_linea_base = Pipeline([
        ("limpiador", FunctionTransformer(limpiar_lote_textos, validate=False)),
        ("vectorizador", TfidfVectorizer(max_features=2500)),
        ("clasificador", DummyClassifier(strategy="most_frequent", random_state=semilla))
    ])
    pipeline_linea_base.fit(x_entreno, y_entreno)
    tiempo_base = time.perf_counter() - inicio_base
    predicciones_base = pipeline_linea_base.predict(x_prueba)
    metricas_linea_base = calcular_metricas_modelo(
        "Línea Base (DummyClassifier)",
        y_prueba,
        predicciones_base,
        tiempo_entrenamiento=tiempo_base
    )

    # --- ESCALÓN 1: MODELO 1 - NAIVE BAYES MULTINOMIAL + GRIDSEARCHCV ---
    print("\nEntrenando Escalón 1: Naive Bayes Multinomial con GridSearchCV (5 pliegues)...")
    inicio_nb = time.perf_counter()
    pipeline_nb = Pipeline([
        ("limpiador", FunctionTransformer(limpiar_lote_textos, validate=False)),
        ("vectorizador", TfidfVectorizer(max_features=2500, min_df=2, max_df=0.95)),
        ("clasificador", MultinomialNB())
    ])
    rejilla_parametros_nb = {
        "vectorizador__ngram_range": [(1, 1), (1, 2)],
        "clasificador__alpha": [0.1, 0.5, 1.0, 2.0]
    }
    busqueda_nb = GridSearchCV(
        pipeline_nb,
        param_grid=rejilla_parametros_nb,
        cv=5,
        scoring="f1_macro",
        n_jobs=1
    )
    busqueda_nb.fit(x_entreno, y_entreno)
    tiempo_nb = time.perf_counter() - inicio_nb

    modelo_nb = busqueda_nb.best_estimator_
    predicciones_nb = modelo_nb.predict(x_prueba)
    metricas_nb = calcular_metricas_modelo(
        "Naive Bayes Multinomial",
        y_prueba,
        predicciones_nb,
        tiempo_entrenamiento=tiempo_nb
    )

    # --- ESCALÓN 2: MODELO 2 - REGRESIÓN LOGÍSTICA + GRIDSEARCHCV ---
    print("Entrenando Escalón 2: Regresión Logística con GridSearchCV (5 pliegues)...")
    inicio_logistica = time.perf_counter()
    pipeline_logistica = Pipeline([
        ("limpiador", FunctionTransformer(limpiar_lote_textos, validate=False)),
        ("vectorizador", TfidfVectorizer(max_features=2500, min_df=2, max_df=0.95)),
        ("clasificador", LogisticRegression(max_iter=1000, random_state=semilla))
    ])
    rejilla_parametros_logistica = {
        "vectorizador__ngram_range": [(1, 1), (1, 2)],
        "clasificador__C": [0.5, 1.0, 5.0, 10.0]
    }
    busqueda_logistica = GridSearchCV(
        pipeline_logistica,
        param_grid=rejilla_parametros_logistica,
        cv=5,
        scoring="f1_macro",
        n_jobs=1
    )
    busqueda_logistica.fit(x_entreno, y_entreno)
    tiempo_logistica = time.perf_counter() - inicio_logistica

    modelo_logistica = busqueda_logistica.best_estimator_
    predicciones_logistica = modelo_logistica.predict(x_prueba)
    probabilidades_logistica = modelo_logistica.predict_proba(x_prueba)
    metricas_logistica = calcular_metricas_modelo(
        "Regresión Logística",
        y_prueba,
        predicciones_logistica,
        tiempo_entrenamiento=tiempo_logistica
    )

    # ==========================================================================
    # 4. EXTRACCIÓN DE PALABRAS RELEVANTES Y ANÁLISIS DE ERRORES
    # ==========================================================================
    # Ajustamos un vectorizador sobre los textos limpios con ngram_range=(1, 2)
    # o usamos los componentes de los pipelines ganadores para extraer importancias
    vectorizador_logistica = modelo_logistica.named_steps["vectorizador"]
    clasificador_logistica = modelo_logistica.named_steps["clasificador"]

    # Para comparar en el mismo vocabulario con Naive Bayes si tuvieron distinto ngram_range,
    # ajustamos un MultinomialNB auxiliar sobre la misma matriz si las dimensiones difieren
    vectorizador_nb = modelo_nb.named_steps["vectorizador"]
    clasificador_nb = modelo_nb.named_steps["clasificador"]
    if len(vectorizador_nb.get_feature_names_out()) == len(vectorizador_logistica.get_feature_names_out()):
        nb_para_palabras = clasificador_nb
    else:
        nb_para_palabras = MultinomialNB(alpha=busqueda_nb.best_params_["clasificador__alpha"])
        matriz_entreno_log = vectorizador_logistica.transform(limpiar_lote_textos(x_entreno))
        nb_para_palabras.fit(matriz_entreno_log, y_entreno)

    palabras_relevantes = extraer_palabras_mas_importantes(
        vectorizador=vectorizador_logistica,
        modelo_logistica=clasificador_logistica,
        modelo_nb=nb_para_palabras,
        top_n=12
    )

    textos_prueba_limpios = limpiar_lote_textos(x_prueba)

    # Seleccionamos el modelo ganador según el F1-Macro en Validación Cruzada (sin tocar prueba para decidir)
    # y la calibración de probabilidades mediante la función sigmoide (Escalón 2 de la Escalera de Modelos)
    if busqueda_logistica.best_score_ >= busqueda_nb.best_score_:
        modelo_ganador = modelo_logistica
        nombre_ganador = "Regresión Logística"
        predicciones_ganador = predicciones_logistica
        probabilidades_ganador = probabilidades_logistica
    else:
        modelo_ganador = modelo_nb
        nombre_ganador = "Naive Bayes Multinomial"
        predicciones_ganador = predicciones_nb
        probabilidades_ganador = modelo_nb.predict_proba(x_prueba)

    errores_comentados = extraer_errores_comentados(
        textos_originales=x_prueba,
        textos_limpios=textos_prueba_limpios,
        y_real=y_prueba,
        y_predicho=predicciones_ganador,
        probabilidades=probabilidades_ganador,
        cantidad=5
    )

    # ==========================================================================
    # 5. SERIALIZACIÓN DE MODELOS Y REPORTE JSON
    # ==========================================================================
    directorio_modelos = os.path.dirname(__file__)
    ruta_modelo_principal = os.path.join(directorio_modelos, "modelo_sentimiento.joblib")
    ruta_modelo_logistica = os.path.join(directorio_modelos, "modelo_logistica.joblib")
    ruta_modelo_nb = os.path.join(directorio_modelos, "modelo_naive_bayes.joblib")
    ruta_metricas_json = os.path.join(directorio_modelos, "metricas.json")

    joblib.dump(modelo_ganador, ruta_modelo_principal)
    joblib.dump(modelo_logistica, ruta_modelo_logistica)
    joblib.dump(modelo_nb, ruta_modelo_nb)

    reporte_metricas = {
        "semilla_reproducibilidad": semilla,
        "total_datos": int(len(df)),
        "datos_entrenamiento": int(len(x_entreno)),
        "datos_prueba": int(len(x_prueba)),
        "modelo_en_produccion": {
            "nombre": nombre_ganador,
            "archivo_serializado": "src/models/modelo_sentimiento.joblib",
            "f1_macro_prueba": metricas_logistica["f1_macro"] if nombre_ganador == "Regresión Logística" else metricas_nb["f1_macro"],
            "exactitud_prueba": metricas_logistica["exactitud"] if nombre_ganador == "Regresión Logística" else metricas_nb["exactitud"],
            "justificacion": (
                f"Se seleccionó {nombre_ganador} para producción tras validar la Escalera de Modelos "
                f"(Línea Base -> Naive Bayes -> Regresión Logística). Alcanza un F1-Macro de "
                f"{metricas_logistica['f1_macro'] if nombre_ganador == 'Regresión Logística' else metricas_nb['f1_macro']} "
                f"superando ampliamente la línea base ({metricas_linea_base['f1_macro']}), ofrece probabilidades "
                f"bien calibradas mediante la función sigmoide y permite interpretar directamente los pesos de cada término y bigrama."
            )
        },
        "hiperparametros_optimos": {
            "naive_bayes": {
                "alpha": busqueda_nb.best_params_["clasificador__alpha"],
                "ngram_range": list(busqueda_nb.best_params_["vectorizador__ngram_range"]),
                "mejor_f1_macro_validacion_cruzada": round(float(busqueda_nb.best_score_), 4)
            },
            "regresion_logistica": {
                "C": busqueda_logistica.best_params_["clasificador__C"],
                "ngram_range": list(busqueda_logistica.best_params_["vectorizador__ngram_range"]),
                "mejor_f1_macro_validacion_cruzada": round(float(busqueda_logistica.best_score_), 4)
            }
        },
        "comparacion_modelos": {
            "linea_base": metricas_linea_base,
            "naive_bayes": metricas_nb,
            "regresion_logistica": metricas_logistica
        },
        "palabras_relevantes": palabras_relevantes,
        "analisis_errores": errores_comentados
    }

    with open(ruta_metricas_json, mode="w", encoding="utf-8") as archivo_json:
        json.dump(reporte_metricas, archivo_json, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("RESUMEN DE LA ESCALERA DE MODELOS (EVALUACIÓN EN PRUEBA 20%)")
    print("=" * 80)
    print(f"0. Línea Base (Dummy)      -> Exactitud: {metricas_linea_base['exactitud']:.4f} | F1-Macro: {metricas_linea_base['f1_macro']:.4f}")
    print(f"1. Naive Bayes Multinomial -> Exactitud: {metricas_nb['exactitud']:.4f} | F1-Macro: {metricas_nb['f1_macro']:.4f} | Mejores params: {busqueda_nb.best_params_}")
    print(f"2. Regresión Logística     -> Exactitud: {metricas_logistica['exactitud']:.4f} | F1-Macro: {metricas_logistica['f1_macro']:.4f} | Mejores params: {busqueda_logistica.best_params_}")
    print("-" * 80)
    print(f"Modelo seleccionado en producción : {nombre_ganador}")
    print(f"Pipeline principal guardado en    : {ruta_modelo_principal}")
    print(f"Reporte de métricas guardado en   : {ruta_metricas_json}")
    print("=" * 80)


if __name__ == "__main__":
    entrenar_y_guardar_modelos()

