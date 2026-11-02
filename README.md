# Clasificador de Sentimiento de Reseñas en Español (Hoteles y Restaurantes)

Sistema completo de **Procesamiento de Lenguaje Natural (NLP)** y **Machine Learning supervisado** para clasificar reseñas de restaurantes y hoteles en español (`positivo` vs. `negativo`), desarrollado para el **Proyecto Integrador**.

Incluye generación reproducible de datos, pipeline de limpieza morfológica en español con preservación de negaciones, entrenamiento automatizado con validación cruzada (**La Escalera de Modelos**), servidor **API REST en FastAPI**, **Frontend interactivo** con radiografía NLP paso a paso y **suite de 10 pruebas automatizadas con `pytest`**.

---

## 1. Arquitectura de las 5 Capas

1. **Capa de Datos (`data/`):** Generador determinista (`semilla = 42`) de **1.200 reseñas realistas** en español (`600` positivas y `600` negativas) con jerga local (`bolivianos`), URLs, menciones `@`, negaciones con `"no"` y 36 casos ambiguos/irónicos.
2. **Capa de NLP (`src/nlp/limpieza_texto.py`):** Normalización de texto, reemplazo de tokens semánticos (`url`, `usuario`, `monto`, `num`), eliminación de tildes (`unidecode`), tokenización NLTK, filtrado de *stop words* **protegiendo negaciones (`no`, `nunca`, `sin`, `nada`)**, *Stemming* (`SnowballStemmer`) y **Lematización** de dominio.
3. **Capa de Machine Learning (`src/models/`):** Entrenamiento reproducible en un solo comando (`entrenar.py`) evaluando 3 escalones (`DummyClassifier` $\rightarrow$ `MultinomialNB` $\rightarrow$ `LogisticRegression`) con `GridSearchCV` (5 pliegues) y serialización de pipelines completos con `joblib`.
4. **Capa de Backend API REST (`api/`):** Servidor FastAPI (`api/main.py`) y validaciones Pydantic en español (`api/esquemas.py`) que carga los artefactos `.joblib` al iniciar sin reentrenar ni duplicar código.
5. **Capa de Frontend y Pruebas (`frontend/`, `tests/`):** Interfaz web moderna servida en `http://127.0.0.1:8000/` con analizador individual, radiografía NLP de 6 pasos, predicción por lotes y dashboard de métricas, respaldada por 10 pruebas en `pytest`.

---

## 2. Estructura del Proyecto

```text
laboratorio_proyecto/
├── data/
│   ├── generar_datos.py                         # Generador reproducible (semilla 42)
│   └── resenas.csv                              # Dataset de 1.200 reseñas (utf-8)
├── notebooks/
│   ├── 01_exploracion_datos.ipynb               # EDA: balance de clases, longitudes, top palabras
│   ├── 02_limpieza_y_experimentos_nlp.ipynb     # Antes/después, Stemming vs Lematización, N-gramas
│   └── 03_entrenamiento_y_evaluacion_modelos.ipynb # Escalera de modelos, matrices y análisis de errores
├── src/
│   ├── __init__.py
│   ├── nlp/
│   │   ├── __init__.py
│   │   └── limpieza_texto.py                    # Pipeline NLP y radiografía paso a paso
│   └── models/
│       ├── __init__.py
│       ├── evaluar.py                           # Cálculo de métricas, top palabras y errores
│       ├── entrenar.py                          # Script reproducible en un solo comando
│       ├── modelo_sentimiento.joblib            # Pipeline ganador en producción (Regresión Logística)
│       ├── modelo_logistica.joblib              # Pipeline Regresión Logística
│       ├── modelo_naive_bayes.joblib            # Pipeline Naive Bayes Multinomial
│       └── metricas.json                        # Reporte completo de métricas, pesos y errores
├── api/
│   ├── __init__.py
│   ├── esquemas.py                              # Esquemas y validadores Pydantic en español
│   └── main.py                                  # Servidor FastAPI + entrega estática del Frontend
├── frontend/
│   ├── index.html                               # Interfaz web en 4 secciones
│   ├── estilos.css                              # Estilos tipo dashboard moderno
│   └── app.js                                   # Consumo de API y renderizado dinámico
├── tests/
│   └── test_sistema.py                          # 10 pruebas automatizadas (NLP + API + Frontend)
├── docs/
│   ├── informe_tecnico.md                       # Informe técnico y Ficha de 9 Casilleros
│   └── presentacion_guion.md                    # Guion de 10 diapositivas y guía de defensa en vivo
├── CONTRIBUCIONES.md                            # Desglose de aportes por cada uno de los 4 roles
├── requirements.txt                             # Dependencias del proyecto
└── README.md                                    # Este archivo
```

---

## 3. Instrucciones de Instalación y Ejecución Paso a Paso (Desde Cero)

### Paso 1: Clonar el repositorio y crear el entorno virtual
```bash
git clone <URL_DEL_REPOSITORIO>
cd laboratorio_proyecto
python -m venv .venv
```

### Paso 2: Activar el entorno virtual
- **En Windows (PowerShell):**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- **En Windows (CMD):**
  ```cmd
  .\.venv\Scripts\activate.bat
  ```
- **En Linux / macOS:**
  ```bash
  source .venv/bin/activate
  ```

### Paso 3: Instalar las dependencias y recursos de NLTK
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

### Paso 4: Generar los datos y entrenar los modelos (Un solo comando)
```bash
# (Opcional) Regenerar data/resenas.csv con semilla fija 42:
python data/generar_datos.py

# Entrenar la Escalera de Modelos y exportar .joblib y metricas.json:
python src/models/entrenar.py
```

### Paso 5: Iniciar el servidor Backend + Frontend con Uvicorn
```bash
uvicorn api.main:app --reload
```
- **Aplicación Web Interactiva (Frontend):** Abrir en el navegador [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Documentación Interactiva Swagger (OpenAPI):** Abrir [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Paso 6: Ejecutar las pruebas automatizadas (`pytest`)
```bash
pytest -v
```
*(Resultado esperado: `10 passed` en verde).*

---

## 4. Documentación de los Endpoints de la API REST

### 4.1. `GET /health` *(alias: `GET /estado`)*
Verifica que el servidor está activo y que los pipelines `.joblib` fueron cargados en memoria.

**Respuesta (`200 OK`):**
```json
{
  "estado": "activo",
  "modelo_cargado": true,
  "modelo_en_produccion": "Regresión Logística",
  "modelos_disponibles": ["naive_bayes", "produccion", "regresion_logistica"],
  "total_resenas_dataset": 1200
}
```

### 4.2. `POST /predict` *(alias: `POST /predecir`)*
Clasifica una reseña en español, compara con el modelo alternativo y devuelve la radiografía NLP de 6 pasos.

**Solicitud (`application/json`):**
```json
{
  "texto": "No recomiendo nada este lugar, pésimo servicio y comida fría",
  "modelo_elegido": "produccion"
}
```

**Respuesta (`200 OK`):**
```json
{
  "texto": "No recomiendo nada este lugar, pésimo servicio y comida fría",
  "sentimiento": "negativo",
  "probabilidad_positivo": 0.0042,
  "probabilidad_negativo": 0.9958,
  "confianza": 0.9958,
  "modelo_usado": "Regresión Logística (producción)",
  "modelo_alternativo": {
    "nombre_modelo": "Naive Bayes Multinomial",
    "sentimiento": "negativo",
    "probabilidad_positivo": 0.0003,
    "probabilidad_negativo": 0.9997,
    "coincide_con_principal": true
  },
  "paso_a_paso_nlp": {
    "texto_original": "No recomiendo nada este lugar, pésimo servicio y comida fría",
    "texto_normalizado": "no recomiendo nada este lugar pesimo servicio y comida fria",
    "tokens": ["no", "recomiendo", "nada", "este", "lugar", "pesimo", "servicio", "y", "comida", "fria"],
    "sin_palabras_vacias": ["no", "recomiendo", "nada", "lugar", "pesimo", "servicio", "comida", "fria"],
    "palabras_eliminadas": ["este", "y"],
    "resultado_stemming": ["no", "recom", "nad", "lug", "pesim", "servici", "com", "fri"],
    "resultado_lematizacion": ["no", "recomendar", "nada", "lugar", "pesimo", "servicio", "comida", "frio"]
  }
}
```

### 4.3. `POST /predict/batch` *(alias: `POST /predecir_lote`)*
Evalúa entre 1 y 50 reseñas en una sola petición y devuelve el resumen estadístico del lote.

**Solicitud (`application/json`):**
```json
{
  "textos": [
    "Excelente atención y comida deliciosa, volveremos pronto",
    "Pésimo servicio, la habitación estaba sucia, no recomiendo"
  ],
  "modelo_elegido": "produccion"
}
```

### 4.4. `GET /metrics` *(alias: `GET /metricas`)*
Devuelve el contenido completo de `src/models/metricas.json`: métricas de la Escalera de Modelos (`linea_base`, `naive_bayes`, `regresion_logistica`), hiperparámetros óptimos de `GridSearchCV`, matrices de confusión, las 12 palabras con mayor peso positivo/negativo y los errores comentados del conjunto de prueba.

---

## 5. Resumen de Resultados Experimentales (Conjunto de Prueba 20%)

| Escalón | Modelo | Hiperparámetros Óptimos (`cv=5`) | F1-Macro CV | F1-Macro Prueba | Exactitud Prueba |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **0** | `DummyClassifier` | `strategy="most_frequent"` | — | `0.3333` | `0.5000` |
| **1** | `MultinomialNB` | `alpha=0.1`, `ngram_range=(1, 2)` | `0.9927` | `0.9958` | `0.9958` |
| **2** | **`LogisticRegression` (Producción)** | **`C=5.0`, `ngram_range=(1, 2)`** | **`0.9958`** | **`0.9917`** | **`0.9917`** |

