# Registro de Contribuciones del Equipo (`CONTRIBUCIONES.md`)

En cumplimiento de los **Puntos 6.1 y 6.2 de la Rúbrica**[cite: 7, 8], todos los integrantes del grupo trabajaron en su propia rama funcional (*feature branch*)[cite: 7, 8] y realizaron commits reales y verificables en la capa de **Procesamiento de Lenguaje Natural (NLP) y/o Machine Learning**[cite: 7, 8], además de liderar su módulo asignado dentro de la arquitectura de 5 capas[cite: 7, 8].

---

## Tabla Resumen de Integrantes y Ramas de Trabajo

| Rol | Nombre Completo del Integrante | Usuario de GitHub | Rama de Trabajo (`git branch`) | Módulo Principal Liderado |
| :--- | :--- | :--- | :--- | :--- |
| **Integrante 1** | Erlan Gaston Garcia Mamani | `@Erlan75` | `feature/datos-y-nlp` | Datos, EDA y Pipeline de Limpieza NLP |
| **Integrante 2** | María Montaño Flores | `@M-2026-M` | `feature/modelos-ml` | Modelado, Entrenamiento y Evaluación ML |
| **Integrante 3** | Ronny Ronaldo Choque Cabana | `@DnaetFirst` | `feature/backend-api` | Backend API REST y Pruebas de NLP/API |
| **Integrante 4** | Luis Miguel Nunez Castro | `@Imnunez` | `feature/frontend-y-docs` | Cuaderno de Evaluación ML, Frontend y Documentación |

---

## Detalle de Responsabilidades, Archivos y Commits por Integrante

### 1. Integrante 1 — Rol: Datos y Pipeline de NLP
- **Nombre:** Erlan Gaston Garcia Mamani (`@Erlan75`)
- **Rama de trabajo:** `feature/datos-y-nlp` 
- **Archivos subidos en su rama:**
  - [`data/generar_datos.py`](data/generar_datos.py) 
  - [`data/resenas.csv`](data/resenas.csv) 
  - [`notebooks/01_exploracion_datos.ipynb`](notebooks/01_exploracion_datos.ipynb) 
  - [`src/nlp/limpieza_texto.py`](src/nlp/limpieza_texto.py) *(funciones base de normalización, tokenización, stop words conservando `"no"` y stemming/lematización)* 
- **Aporte específico obligatorio en NLP / Machine Learning (Punto 6.1):**
  - Generación reproducible del dataset balanceado de 1.200 reseñas en español (`semilla = 42`) con ruido realista del dominio (montos en Bs, URLs, menciones `@`, negaciones e ironía) , análisis exploratorio de datos (EDA)[cite: 7, 8] e implementación del módulo `src/nlp/limpieza_texto.py` (`normalizar_texto`, `tokenizar_texto`, `quitar_palabras_vacias` con protección explícita de `"no"`, `aplicar_stemming`, `aplicar_lematizacion` y `procesar_paso_a_paso`) .
- **Commits exactos registrados:**
  1. `feat(datos): agregar generador reproducible y dataset de 1200 resenas en espanol` 
  2. `feat(nlp): implementar normalizacion, stopwords con negacion protegida y lematizacion` 
  3. `docs(eda): desarrollar cuaderno 01_exploracion_datos con graficos de distribucion` 

---

### 2. Integrante 2 — Rol: Modelado, Entrenamiento y Evaluación ML
- **Nombre:** María Montaño Flores (`@M-2026-M`)
- **Rama de trabajo:** `feature/modelos-ml` 
- **Archivos subidos en su rama:**
  - [`notebooks/02_limpieza_y_experimentos_nlp.ipynb`](notebooks/02_limpieza_y_experimentos_nlp.ipynb) 
  - [`src/models/entrenar.py`](src/models/entrenar.py) 
  - [`src/models/evaluar.py`](src/models/evaluar.py) 
  - [`src/models/metricas.json`](src/models/metricas.json) 
  - [`src/models/modelo_sentimiento.joblib`](src/models/modelo_sentimiento.joblib), [`src/models/modelo_logistica.joblib`](src/models/modelo_logistica.joblib) y [`src/models/modelo_naive_bayes.joblib`](src/models/modelo_naive_bayes.joblib) 
- **Aporte específico obligatorio en NLP / Machine Learning (Punto 6.1):**
  - Desarrollo de los experimentos comparativos en el cuaderno 02 (*Stemming vs. Lematización* y *Unigramas `(1, 1)` vs. Unigramas + Bigramas `(1, 2)`*) , implementación de las funciones de métricas, extracción de términos relevantes y análisis de errores en `src/models/evaluar.py` , y construcción del pipeline reproducible de entrenamiento de La Escalera de Modelos (`DummyClassifier`, `MultinomialNB` y `LogisticRegression`) con `GridSearchCV` (5 pliegues) y serialización `.joblib` en `src/models/entrenar.py` .
- **Commits exactos registrados:**
  1. `feat(experimentos): comparar stemming vs lematizacion y unigramas vs bigramas en cuaderno 02` 
  2. `feat(ml): implementar evaluar.py con F1 macro, matriz de confusion y analisis de errores` 
  3. `feat(ml): crear script reproducible entrenar.py con GridSearchCV y serializacion joblib` 

---

### 3. Integrante 3 — Rol: Backend API REST y Pruebas de NLP/API
- **Nombre:** Ronny Ronaldo Choque Cabana (`@DnaetFirst`)
- **Rama de trabajo:** `feature/backend-api` 
- **Archivos subidos en su rama:**
  - [`tests/test_sistema.py`](tests/test_sistema.py) 
  - [`api/esquemas.py`](api/esquemas.py) 
  - [`api/main.py`](api/main.py) 
- **Aporte específico obligatorio en NLP / Machine Learning (Punto 6.1):**
  - Implementación de las pruebas unitarias y de integración en `tests/test_sistema.py` validando directamente el comportamiento del pipeline de NLP (normalización de URLs/montos/tildes, conservación crítica de la negación `"no"` en stop words, transformación morfológica de *stemming* vs. lematización) y la inferencia de los modelos serializados `.joblib` , además de integrar la inferencia dual (modelo en producción vs. modelo alternativo) en `api/main.py` y las validaciones lingüísticas en `api/esquemas.py` .
- **Commits exactos registrados:**
  1. `test(nlp): implementar pruebas unitarias del pipeline de limpieza, stopwords y modelos` 
  2. `feat(api): definir esquemas Pydantic en espanol con validacion de texto y longitud` 
  3. `feat(api): implementar servidor FastAPI con endpoints /health, /metrics, /predict y /predict/batch` 

---

### 4. Integrante 4 — Rol: Cuaderno de Evaluación ML, Frontend y Documentación
- **Nombre:** Luis Miguel Nunez Castro (`@Imnunez`)
- **Rama de trabajo:** `feature/frontend-y-docs` 
- **Archivos subidos en su rama:**
  - [`notebooks/03_entrenamiento_y_evaluacion_modelos.ipynb`](notebooks/03_entrenamiento_y_evaluacion_modelos.ipynb) 
  - [`frontend/index.html`](frontend/index.html) 
  - [`frontend/estilos.css`](frontend/estilos.css) 
  - [`frontend/app.js`](frontend/app.js) 
  - [`docs/informe_tecnico.md`](docs/informe_tecnico.md) 
  - [`docs/presentacion_guion.md`](docs/presentacion_guion.md) 
  - [`README.md`](README.md) 
  - [`CONTRIBUCIONES.md`](CONTRIBUCIONES.md) 
- **Aporte específico obligatorio en NLP / Machine Learning (Punto 6.1):**
  - Desarrollo completo del cuaderno `notebooks/03_entrenamiento_y_evaluacion_modelos.ipynb` documentando y graficando la Escalera de Modelos, las matrices de confusión lado a lado (`ConfusionMatrixDisplay`), la importancia de características TF-IDF (`coef_` de Regresión Logística y `feature_log_prob_` de Naive Bayes) y el análisis lingüístico de errores del conjunto de prueba , además de construir la visualización interactiva de la Radiografía NLP de 6 pasos en el Frontend y la documentación técnica con la Ficha de 9 Casilleros .
- **Commits exactos registrados:**
  1. `feat(ml-eval): desarrollar cuaderno 03 con matrices de confusion, pesos TF-IDF y errores` 
  2. `feat(frontend): construir interfaz web con radiografia NLP paso a paso y panel de metricas` 
  3. `docs(proyecto): agregar informe tecnico con ficha de 9 casilleros, README y contribuciones` 