/* ==========================================================================
   LÓGICA DEL FRONTEND - Clasificador de Sentimiento de Reseñas en Español
   Consume los endpoints de api/main.py: /health, /predict, /predict/batch, /metrics
   ========================================================================== */

// Si la página se abre directamente desde el disco (file://) apuntamos al servidor local;
// si la sirve FastAPI, usamos el mismo origen.
const URL_API = window.location.protocol === "file:"
  ? "http://127.0.0.1:8000"
  : window.location.origin;

// Palabras que protegemos en el filtrado de stop words (igual que PALABRAS_CONSERVAR_SENTIMIENTO)
const PALABRAS_PROTEGIDAS = new Set([
  "no", "ni", "nunca", "jamas", "sin", "tampoco", "nada", "muy", "mas", "poco", "peor", "mejor"
]);

// Tokens especiales generados por normalizar_texto
const TOKENS_ESPECIALES = new Set(["url", "usuario", "monto", "num"]);

const EJEMPLOS_RAPIDOS = {
  positivo: "¡Excelente atención en @hotel_central! Pagamos 350 bolivianos por la habitación y valió cada centavo: limpia, cómoda y el desayuno delicioso. Reservamos en https://hotel-ejemplo.com, 100% recomendado.",
  negativo: "No me gustó nada la atención. No recomiendo este lugar: la comida llegó fría, tardaron 2 horas y nunca pidieron disculpas. Pésimo servicio.",
  dificil: "Al principio no aparecía nuestra reserva en https://hotel-ejemplo.com, pero el gerente lo resolvió rápido y fue excelente."
};

const LOTE_DE_PRUEBA = [
  "Excelente hotel, las habitaciones impecables y el personal muy amable. Volveremos pronto.",
  "Pésimo servicio, la habitación estaba sucia y no había agua caliente. No recomiendo.",
  "La comida llegó súper caliente y deliciosa, pagamos 120 bolivianos y valió la pena.",
  "Nunca más vuelvo, esperamos 2 horas y la comida llegó fría y cruda.",
  "El lugar es bonito pero la atención fue lenta y el mesero grosero."
];

// --------------------------------------------------------------------------
// UTILIDADES
// --------------------------------------------------------------------------

/** Escapa texto para insertarlo en HTML de forma segura (evita inyección de código). */
function escaparHtml(texto) {
  return String(texto ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

function formatearPorcentaje(valor, decimales = 1) {
  return `${(Number(valor) * 100).toFixed(decimales)}%`;
}

function mostrarElemento(id, visible) {
  document.getElementById(id).classList.toggle("oculto", !visible);
}

function mostrarError(idContenedor, mensaje) {
  const contenedor = document.getElementById(idContenedor);
  contenedor.textContent = mensaje;
  contenedor.classList.remove("oculto");
}

function ocultarError(idContenedor) {
  document.getElementById(idContenedor).classList.add("oculto");
}

/**
 * Llama a la API y devuelve el JSON. Si la respuesta es un error, lanza una
 * excepción con el mensaje en español que devuelve el backend:
 *  - 422 de validación -> { detalle: "..." }
 *  - HTTPException     -> { detail: "..." }
 */
async function llamarApi(ruta, opciones = {}) {
  let respuesta;
  try {
    respuesta = await fetch(`${URL_API}${ruta}`, {
      headers: { "Content-Type": "application/json" },
      ...opciones
    });
  } catch (errorRed) {
    throw new Error(`No se pudo conectar con la API en ${URL_API}. ¿Está ejecutándose uvicorn?`);
  }

  const datos = await respuesta.json().catch(() => ({}));
  if (!respuesta.ok) {
    const mensaje = datos.detalle || datos.detail || `Error HTTP ${respuesta.status}`;
    throw new Error(typeof mensaje === "string" ? mensaje : JSON.stringify(mensaje));
  }
  return datos;
}

function crearChips(lista, obtenerClase = () => "chip") {
  if (!lista || lista.length === 0) {
    return '<span class="texto-tenue">(ninguna)</span>';
  }
  return lista.map(palabra => `<span class="${obtenerClase(palabra)}">${escaparHtml(palabra)}</span>`).join("");
}

// --------------------------------------------------------------------------
// ENCABEZADO: ESTADO DEL SERVICIO (GET /health)
// --------------------------------------------------------------------------

async function consultarEstado() {
  const punto = document.getElementById("punto-estado");
  const texto = document.getElementById("texto-estado");
  try {
    const estado = await llamarApi("/health");
    const conectado = estado.estado === "activo" && estado.modelo_cargado;
    punto.className = `punto-estado ${conectado ? "punto-conectado" : "punto-desconectado"}`;
    texto.textContent = conectado ? "API Conectada" : "API activa sin modelo cargado";

    document.getElementById("estado-modelo").textContent = estado.modelo_en_produccion || "—";
    document.getElementById("estado-total").textContent = estado.total_resenas_dataset.toLocaleString("es");
    document.getElementById("estado-modelos").textContent = estado.modelos_disponibles.join(", ") || "—";

    if (estado.modelo_en_produccion) {
      document.getElementById("opcion-produccion").textContent =
        `Modelo en Producción (${estado.modelo_en_produccion})`;
    }
  } catch (error) {
    punto.className = "punto-estado punto-desconectado";
    texto.textContent = "API Desconectada";
  }
}

// --------------------------------------------------------------------------
// SECCIÓN 1: ANALIZADOR INDIVIDUAL (POST /predict)
// --------------------------------------------------------------------------

function actualizarContadorTexto() {
  const texto = document.getElementById("texto-resena").value;
  const cantidadPalabras = texto.trim() ? texto.trim().split(/\s+/).length : 0;
  document.getElementById("contador-texto").textContent =
    `${texto.length} / 2000 caracteres · ${cantidadPalabras} palabras`;
}

async function analizarResena() {
  const texto = document.getElementById("texto-resena").value;
  const modeloElegido = document.getElementById("selector-modelo").value;
  const boton = document.getElementById("boton-analizar");

  ocultarError("mensaje-error");
  boton.disabled = true;
  boton.textContent = "Analizando...";

  try {
    // La validación la hace la API (esquemas.py); aquí solo mostramos su mensaje
    const resultado = await llamarApi("/predict", {
      method: "POST",
      body: JSON.stringify({ texto: texto, modelo_elegido: modeloElegido })
    });
    mostrarResultado(resultado);
    mostrarRadiografia(resultado.paso_a_paso_nlp);
  } catch (error) {
    mostrarError("mensaje-error", `⚠️ ${error.message}`);
  } finally {
    boton.disabled = false;
    boton.textContent = "Analizar sentimiento";
  }
}

function mostrarResultado(resultado) {
  mostrarElemento("resultado-vacio", false);
  mostrarElemento("resultado-contenido", true);

  const esPositivo = resultado.sentimiento === "positivo";
  const etiqueta = document.getElementById("etiqueta-sentimiento");
  etiqueta.textContent = esPositivo ? "😊 POSITIVO" : "😠 NEGATIVO";
  etiqueta.className = `etiqueta-sentimiento ${esPositivo ? "sentimiento-positivo" : "sentimiento-negativo"}`;

  document.getElementById("resultado-modelo").textContent = resultado.modelo_usado;
  document.getElementById("resultado-confianza").textContent = formatearPorcentaje(resultado.confianza);

  document.getElementById("barra-negativo").style.width = formatearPorcentaje(resultado.probabilidad_negativo, 2);
  document.getElementById("barra-positivo").style.width = formatearPorcentaje(resultado.probabilidad_positivo, 2);
  document.getElementById("porcentaje-negativo").textContent = formatearPorcentaje(resultado.probabilidad_negativo);
  document.getElementById("porcentaje-positivo").textContent = formatearPorcentaje(resultado.probabilidad_positivo);

  const contenedorAlternativo = document.getElementById("contenido-alternativo");
  const alternativo = resultado.modelo_alternativo;
  if (!alternativo) {
    contenedorAlternativo.innerHTML = '<span class="texto-tenue">No hay un modelo alternativo cargado.</span>';
    return;
  }
  const claseSentimiento = alternativo.sentimiento === "positivo" ? "insignia-positivo" : "insignia-negativo";
  const insigniaAcuerdo = alternativo.coincide_con_principal
    ? '<span class="insignia insignia-coincide">✔ Ambos modelos coinciden</span>'
    : '<span class="insignia insignia-discrepa">⚠ Los modelos discrepan</span>';

  contenedorAlternativo.innerHTML = `
    <p style="margin:4px 0"><strong>${escaparHtml(alternativo.nombre_modelo)}</strong> predice
      <span class="insignia ${claseSentimiento}">${escaparHtml(alternativo.sentimiento.toUpperCase())}</span>
      (positivo ${formatearPorcentaje(alternativo.probabilidad_positivo)} · negativo ${formatearPorcentaje(alternativo.probabilidad_negativo)})
    </p>
    ${insigniaAcuerdo}`;
}

// --------------------------------------------------------------------------
// SECCIÓN 2: RADIOGRAFÍA DEL PIPELINE NLP
// --------------------------------------------------------------------------

function mostrarRadiografia(paso) {
  mostrarElemento("radiografia-vacia", false);
  mostrarElemento("flujo-nlp", true);

  const claseToken = palabra => {
    if (TOKENS_ESPECIALES.has(palabra)) return "chip chip-especial";
    if (PALABRAS_PROTEGIDAS.has(palabra)) return "chip chip-protegida";
    return "chip";
  };

  document.getElementById("paso-original").textContent = paso.texto_original;
  document.getElementById("paso-normalizado").textContent = paso.texto_normalizado || "(vacío)";
  document.getElementById("cantidad-tokens").textContent = paso.tokens.length;
  document.getElementById("paso-tokens").innerHTML = crearChips(paso.tokens, claseToken);
  document.getElementById("paso-sin-vacias").innerHTML = crearChips(paso.sin_palabras_vacias, claseToken);
  document.getElementById("paso-eliminadas").innerHTML = crearChips(paso.palabras_eliminadas, () => "chip chip-eliminada");
  document.getElementById("paso-stemming").innerHTML = crearChips(paso.resultado_stemming);
  document.getElementById("paso-lematizacion").innerHTML = crearChips(paso.resultado_lematizacion, palabra =>
    PALABRAS_PROTEGIDAS.has(palabra) ? "chip chip-protegida" : "chip chip-lema"
  );
}

// --------------------------------------------------------------------------
// SECCIÓN 3: PREDICCIÓN POR LOTES (POST /predict/batch)
// --------------------------------------------------------------------------

function obtenerLineasLote() {
  return document.getElementById("texto-lote").value
    .split(/\r?\n/)
    .map(linea => linea.trim())
    .filter(linea => linea.length > 0);
}

function actualizarContadorLote() {
  const cantidad = obtenerLineasLote().length;
  document.getElementById("contador-lote").textContent = `${cantidad} reseña(s)${cantidad > 50 ? " · ⚠ máximo 50" : ""}`;
}

/** Separa una línea CSV respetando comillas dobles. */
function separarLineaCsv(linea) {
  const campos = [];
  let campoActual = "";
  let dentroDeComillas = false;
  for (let i = 0; i < linea.length; i++) {
    const caracter = linea[i];
    if (caracter === '"') {
      if (dentroDeComillas && linea[i + 1] === '"') { campoActual += '"'; i++; }
      else { dentroDeComillas = !dentroDeComillas; }
    } else if (caracter === "," && !dentroDeComillas) {
      campos.push(campoActual); campoActual = "";
    } else {
      campoActual += caracter;
    }
  }
  campos.push(campoActual);
  return campos;
}

/** Lee un .txt (una reseña por línea) o un .csv (usa la columna 'texto' si existe). */
function leerArchivoLote(evento) {
  const archivo = evento.target.files[0];
  if (!archivo) return;

  const lector = new FileReader();
  lector.onload = () => {
    let lineas = String(lector.result).split(/\r?\n/).filter(linea => linea.trim());
    if (archivo.name.toLowerCase().endsWith(".csv") && lineas.length > 0) {
      const cabecera = separarLineaCsv(lineas[0]).map(c => c.trim().toLowerCase());
      const indiceTexto = cabecera.indexOf("texto");
      if (indiceTexto >= 0) {
        lineas = lineas.slice(1).map(linea => separarLineaCsv(linea)[indiceTexto] || "");
      }
    }
    const lineasLimpias = lineas.map(l => l.trim()).filter(Boolean);
    document.getElementById("texto-lote").value = lineasLimpias.slice(0, 50).join("\n");
    actualizarContadorLote();
    if (lineasLimpias.length > 50) {
      mostrarError("mensaje-error-lote", `El archivo tenía ${lineasLimpias.length} reseñas; se cargaron solo las primeras 50.`);
    } else {
      ocultarError("mensaje-error-lote");
    }
  };
  lector.readAsText(archivo, "utf-8");
  evento.target.value = "";
}

async function analizarLote() {
  const textos = obtenerLineasLote();
  const boton = document.getElementById("boton-analizar-lote");
  ocultarError("mensaje-error-lote");

  if (textos.length === 0) {
    mostrarError("mensaje-error-lote", "⚠️ Escriba al menos una reseña (una por línea).");
    return;
  }

  boton.disabled = true;
  boton.textContent = "Analizando...";
  try {
    const resultado = await llamarApi("/predict/batch", {
      method: "POST",
      body: JSON.stringify({ textos: textos, modelo_elegido: document.getElementById("selector-modelo").value })
    });
    mostrarResultadoLote(resultado);
  } catch (error) {
    mostrarError("mensaje-error-lote", `⚠️ ${error.message}`);
  } finally {
    boton.disabled = false;
    boton.textContent = "Analizar lote";
  }
}

function mostrarResultadoLote(resultado) {
  mostrarElemento("resultado-lote", true);
  const resumen = resultado.resumen;
  document.getElementById("lote-total").textContent = resumen.total_resenas;
  document.getElementById("lote-positivas").textContent = resumen.cantidad_positivas;
  document.getElementById("lote-negativas").textContent = resumen.cantidad_negativas;
  document.getElementById("lote-porcentaje").textContent = `${resumen.porcentaje_positivas}%`;
  document.getElementById("lote-modelo").textContent = resultado.modelo_usado;

  document.getElementById("cuerpo-tabla-lote").innerHTML = resultado.predicciones.map((prediccion, indice) => {
    const clase = prediccion.sentimiento === "positivo" ? "insignia-positivo" : "insignia-negativo";
    return `<tr>
      <td>${indice + 1}</td>
      <td>${escaparHtml(prediccion.texto)}</td>
      <td><span class="insignia ${clase}">${escaparHtml(prediccion.sentimiento.toUpperCase())}</span></td>
      <td>${formatearPorcentaje(prediccion.probabilidad_positivo)}</td>
      <td>${formatearPorcentaje(prediccion.confianza)}</td>
    </tr>`;
  }).join("");
}

// --------------------------------------------------------------------------
// SECCIÓN 4: PANEL DE MÉTRICAS (GET /metrics)
// --------------------------------------------------------------------------

async function cargarMetricas() {
  try {
    const metricas = await llamarApi("/metrics");
    mostrarResumenProduccion(metricas);
    mostrarEscalera(metricas.comparacion_modelos, metricas.modelo_en_produccion?.nombre);
    mostrarHiperparametros(metricas.hiperparametros_optimos);
    mostrarMatrices(metricas.comparacion_modelos);
    mostrarPalabrasRelevantes(metricas.palabras_relevantes);
    mostrarErrores(metricas.analisis_errores);
  } catch (error) {
    mostrarError("mensaje-error-metricas", `⚠️ No se pudieron cargar las métricas: ${error.message}`);
  }
}

function mostrarResumenProduccion(metricas) {
  const produccion = metricas.modelo_en_produccion || {};
  document.getElementById("metricas-modelo-produccion").textContent = produccion.nombre || "—";
  document.getElementById("metricas-justificacion").textContent = produccion.justificacion || "";
  document.getElementById("metricas-division").textContent =
    `Datos: ${metricas.total_datos} reseñas · Entrenamiento: ${metricas.datos_entrenamiento} (80%) · ` +
    `Prueba: ${metricas.datos_prueba} (20%) · Semilla: ${metricas.semilla_reproducibilidad}`;
}

function mostrarEscalera(comparacion, nombreProduccion) {
  const escalones = [
    { escalon: "0 · Línea Base", clave: "linea_base", claseBarra: "relleno-base" },
    { escalon: "1 · Naive Bayes", clave: "naive_bayes", claseBarra: "relleno-nb" },
    { escalon: "2 · Regresión Logística", clave: "regresion_logistica", claseBarra: "" }
  ].filter(e => comparacion[e.clave]);

  // Mejor valor de cada métrica para resaltarlo
  const nombresMetricas = ["f1_macro", "exactitud", "precision_macro", "exhaustividad_recall_macro"];
  const mejores = {};
  nombresMetricas.forEach(m => { mejores[m] = Math.max(...escalones.map(e => comparacion[e.clave][m])); });

  document.getElementById("cuerpo-tabla-escalera").innerHTML = escalones.map(e => {
    const datos = comparacion[e.clave];
    const esProduccion = datos.nombre_modelo === nombreProduccion;
    const celda = m => `<td class="${datos[m] === mejores[m] ? "celda-mejor" : ""}">${datos[m].toFixed(4)}</td>`;
    return `<tr class="${esProduccion ? "fila-ganadora" : ""}">
      <td>${e.escalon}</td>
      <td>${escaparHtml(datos.nombre_modelo)} ${esProduccion ? '<span class="insignia insignia-coincide">En producción</span>' : ""}</td>
      ${nombresMetricas.map(celda).join("")}
      <td>${datos.tiempo_entrenamiento_segundos.toFixed(2)}</td>
    </tr>`;
  }).join("");

  document.getElementById("barras-escalera").innerHTML = escalones.map(e => {
    const f1 = comparacion[e.clave].f1_macro;
    return `<div class="fila-barra">
      <span>${e.escalon}</span>
      <div class="pista-barra"><div class="relleno-barra ${e.claseBarra}" style="width:${(f1 * 100).toFixed(1)}%"></div></div>
      <strong>${f1.toFixed(4)}</strong>
    </div>`;
  }).join("");
}

function mostrarHiperparametros(hiperparametros) {
  if (!hiperparametros) return;
  const nb = hiperparametros.naive_bayes;
  const rl = hiperparametros.regresion_logistica;
  const caja = (titulo, filas) => `
    <div class="caja-hiperparametros">
      <strong>${titulo}</strong>
      <dl>${filas.map(([nombre, valor]) => `<dt>${nombre}</dt><dd>${escaparHtml(valor)}</dd>`).join("")}</dl>
    </div>`;

  document.getElementById("contenedor-hiperparametros").innerHTML =
    caja("Naive Bayes Multinomial", [
      ["alpha (suavizado de Laplace)", nb.alpha],
      ["ngram_range", `(${nb.ngram_range.join(", ")})`],
      ["F1-Macro validación cruzada", nb.mejor_f1_macro_validacion_cruzada.toFixed(4)]
    ]) +
    caja("Regresión Logística", [
      ["C (regularización inversa)", rl.C],
      ["ngram_range", `(${rl.ngram_range.join(", ")})`],
      ["F1-Macro validación cruzada", rl.mejor_f1_macro_validacion_cruzada.toFixed(4)]
    ]);
}

function mostrarMatrices(comparacion) {
  const modelos = [
    { clave: "naive_bayes", titulo: "Naive Bayes Multinomial" },
    { clave: "regresion_logistica", titulo: "Regresión Logística" }
  ].filter(m => comparacion[m.clave]);

  document.getElementById("contenedor-matrices").innerHTML = modelos.map(m => {
    const d = comparacion[m.clave].detalle_matriz_confusion;
    const maximo = Math.max(d.verdaderos_negativos, d.falsos_positivos, d.falsos_negativos, d.verdaderos_positivos, 1);
    // Intensidad de color proporcional al valor (azul para aciertos, rojo para errores)
    const celda = (valor, nombre, esAcierto) => {
      const intensidad = 0.12 + 0.78 * (valor / maximo);
      const fondo = esAcierto ? `rgba(37, 99, 235, ${intensidad})` : `rgba(220, 38, 38, ${valor === 0 ? 0.06 : 0.2 + 0.6 * (valor / maximo)})`;
      const colorTexto = intensidad > 0.5 && esAcierto ? "#fff" : "#1e2433";
      return `<div class="celda-valor" style="background:${fondo};color:${colorTexto}">${valor}<small>${nombre}</small></div>`;
    };
    return `<div>
      <h4>${m.titulo} · F1 ${comparacion[m.clave].f1_macro.toFixed(4)}</h4>
      <div class="matriz">
        <div></div><div class="celda-titulo">Predicho: negativo</div><div class="celda-titulo">Predicho: positivo</div>
        <div class="celda-titulo">Real: negativo</div>
        ${celda(d.verdaderos_negativos, "Verdaderos Negativos", true)}
        ${celda(d.falsos_positivos, "Falsos Positivos", false)}
        <div class="celda-titulo">Real: positivo</div>
        ${celda(d.falsos_negativos, "Falsos Negativos", false)}
        ${celda(d.verdaderos_positivos, "Verdaderos Positivos", true)}
      </div>
    </div>`;
  }).join("");
}

function mostrarPalabrasRelevantes(palabrasRelevantes) {
  const datos = palabrasRelevantes?.regresion_logistica;
  if (!datos) return;
  const pesoMaximo = Math.max(
    ...datos.top_positivas.map(p => Math.abs(p.peso)),
    ...datos.top_negativas.map(p => Math.abs(p.peso))
  );
  const filas = (lista, clase) => lista.map(item => `
    <div class="fila-peso">
      <span class="termino" title="${escaparHtml(item.termino)}">${escaparHtml(item.termino)}</span>
      <div><div class="${clase}" style="width:${(Math.abs(item.peso) / pesoMaximo * 100).toFixed(1)}%"></div></div>
      <span class="valor">${item.peso.toFixed(3)}</span>
    </div>`).join("");

  document.getElementById("palabras-positivas").innerHTML = filas(datos.top_positivas, "barra-peso-positivo");
  document.getElementById("palabras-negativas").innerHTML = filas(datos.top_negativas, "barra-peso-negativo");
}

function mostrarErrores(errores) {
  const cuerpo = document.getElementById("cuerpo-tabla-errores");
  if (!errores || errores.length === 0) {
    cuerpo.innerHTML = '<tr><td colspan="6" class="texto-tenue">No hay errores registrados.</td></tr>';
    return;
  }
  cuerpo.innerHTML = errores.map(error => {
    // Si real == predicha, es un acierto con baja confianza que se incluyó como caso límite
    const esError = error.etiqueta_real !== error.etiqueta_predicha;
    const insigniaTipo = esError
      ? '<span class="insignia insignia-error">Error</span>'
      : '<span class="insignia insignia-duda">Acierto dudoso</span>';
    const insigniaEtiqueta = etiqueta =>
      `<span class="insignia ${etiqueta === "positivo" ? "insignia-positivo" : "insignia-negativo"}">${escaparHtml(etiqueta)}</span>`;
    return `<tr>
      <td>${insigniaTipo}</td>
      <td>${escaparHtml(error.texto_original)}</td>
      <td>${insigniaEtiqueta(error.etiqueta_real)}</td>
      <td>${insigniaEtiqueta(error.etiqueta_predicha)}</td>
      <td>${formatearPorcentaje(error.probabilidad_confianza)}</td>
      <td class="texto-tenue">${escaparHtml(error.explicacion_error)}</td>
    </tr>`;
  }).join("");
}

// --------------------------------------------------------------------------
// INICIALIZACIÓN DE EVENTOS
// --------------------------------------------------------------------------

function inicializarAplicacion() {
  document.getElementById("pie-url-api").textContent = URL_API;
  document.getElementById("enlace-docs").href = `${URL_API}/docs`;

  const areaTexto = document.getElementById("texto-resena");
  areaTexto.addEventListener("input", actualizarContadorTexto);
  areaTexto.addEventListener("keydown", evento => {
    // Ctrl + Enter para analizar rápido durante la defensa
    if (evento.key === "Enter" && evento.ctrlKey) analizarResena();
  });

  document.querySelectorAll(".boton-ejemplo").forEach(boton => {
    boton.addEventListener("click", () => {
      areaTexto.value = EJEMPLOS_RAPIDOS[boton.dataset.ejemplo];
      actualizarContadorTexto();
      analizarResena();
    });
  });

  document.getElementById("boton-analizar").addEventListener("click", analizarResena);
  document.getElementById("boton-limpiar").addEventListener("click", () => {
    areaTexto.value = "";
    actualizarContadorTexto();
    ocultarError("mensaje-error");
    mostrarElemento("resultado-vacio", true);
    mostrarElemento("resultado-contenido", false);
    mostrarElemento("radiografia-vacia", true);
    mostrarElemento("flujo-nlp", false);
  });

  const areaLote = document.getElementById("texto-lote");
  areaLote.addEventListener("input", actualizarContadorLote);
  document.getElementById("boton-lote-prueba").addEventListener("click", () => {
    areaLote.value = LOTE_DE_PRUEBA.join("\n");
    actualizarContadorLote();
    ocultarError("mensaje-error-lote");
  });
  document.getElementById("archivo-lote").addEventListener("change", leerArchivoLote);
  document.getElementById("boton-analizar-lote").addEventListener("click", analizarLote);

  consultarEstado();
  cargarMetricas();
  // Refrescamos el indicador de estado cada 15 segundos
  setInterval(consultarEstado, 15000);
}

document.addEventListener("DOMContentLoaded", inicializarAplicacion);

