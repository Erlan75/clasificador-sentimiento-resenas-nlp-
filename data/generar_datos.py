import csv
import random
import os

def generar_conjunto_datos():
    # Semilla fija para garantizar reproducibilidad en los resultados
    random.seed(42)

    canales_disponibles = ["web", "app_movil", "google_maps", "redes_sociales", "tripadvisor"]

    # Plantillas y fragmentos para reseñas positivas claras
    inicios_positivos = [
        "Excelente atención,",
        "¡Increíble experiencia en este lugar!",
        "Nos encantó el servicio,",
        "Todo estuvo maravilloso desde que llegamos,",
        "Súper recomendado al 100%,",
        "Una experiencia inolvidable en el hotel,",
        "El mejor restaurante de la zona,",
        "Muy buena relación calidad y precio,",
        "La estancia fue comodísima y placentera,",
        "¡Qué delicia de comida y qué gran trato!"
    ]

    detalles_positivos = [
        "la comida llegó súper caliente y deliciosa, 100% recomendado.",
        "las habitaciones estaban impecables, limpias y con una vista hermosa.",
        "el personal de recepción fue muy amable y atento en todo momento.",
        "pagamos 150 bolivianos por un almuerzo completo para 2 personas y valió cada centavo.",
        "el desayuno buffet tenía frutas frescas, jugos naturales y pan recién horneado.",
        "reservamos por https://hotel-ejemplo.com y la atención fue excelente de principio a fin.",
        "felicitaciones a @hotel_central y a @restaurante_sabor por su excelente hospitalidad.",
        "el mesero nos atendió en menos de 10 minutos con una sonrisa genuina y mucha amabilidad.",
        "la cama era súper suave, el aire acondicionado funcionaba perfecto y descansamos muy bien.",
        "estamos muy felices con el trato recibido, sin duda volveremos con toda la familia."
    ]

    cierres_positivos = [
        "Definitivamente volveremos pronto.",
        "Sin duda alguna es nuestro sitio favorito en la ciudad.",
        "¡Cinco estrellas bien merecidas!",
        "Superó todas nuestras expectativas de viaje.",
        "Mis amigos quedaron fascinados con la visita.",
        "Se nota el cariño y la dedicación en cada plato.",
        "Un ambiente familiar, seguro y muy acogedor.",
        "Ideal para descansar el fin de semana.",
        "¡Gracias por hacernos sentir como en casa!",
        "Lo recomiendo totalmente a todos los visitantes."
    ]

    # Plantillas y fragmentos para reseñas negativas claras
    inicios_negativos = [
        "Pésimo servicio,",
        "¡Qué decepción tan grande de lugar!",
        "No fue una buena experiencia,",
        "No me gustó nada la atención,",
        "Una total pérdida de tiempo y dinero,",
        "Horrible experiencia en este establecimiento,",
        "Lamentable estado de las instalaciones,",
        "El peor restaurante al que he ido este año,",
        "¡Cuidado! No recomiendo este sitio,",
        "Pésima gestión y trato grosero del personal,"
    ]

    detalles_negativos = [
        "tardaron 2 horas y la habitación estaba sucia, nunca volveré.",
        "no recomiendo este lugar para nada, la comida llegó fría y cruda.",
        "nos cobraron 500 bolivianos de más en la cuenta y nadie quiso responder.",
        "había cucarachas en el baño y las sábanas tenían manchas viejas y mal olor.",
        "escribimos a @hotel_central y en https://hotel-ejemplo.com pero el servicio fue pésimo y lento.",
        "el aire acondicionado no funcionaba, hacía un calor insoportable y no había agua caliente.",
        "esperamos 45 minutos por una mesa vacía y el mesero nos trató de mala gana.",
        "no tenían cambio, no aceptaban tarjeta y encima el plato costaba 120 bolivianos siendo frío.",
        "el ruido de la calle no nos dejó dormir ni 3 horas en toda la noche.",
        "la sopa estaba salada, la carne dura y fría, y no pidieron ni disculpas."
    ]

    cierres_negativos = [
        "No volvería ni aunque fuera gratis.",
        "Cero recomendable, busquen otras opciones.",
        "Una vergüenza total para el sector gastronómico y hotelero.",
        "No pierdan su dinero aquí, muy mal servicio.",
        "Salimos muy molestos y decepcionados.",
        "Pésima atención, no cumplen lo que prometen.",
        "No recomiendo este establecimiento a nadie.",
        "Pésimo desde el principio hasta el final.",
        "Nunca más volveré a este lugar.",
        "Deberían mejorar urgente la limpieza y el trato."
    ]

    # 36 casos difíciles o ambiguos ÚNICOS (sentimientos mezclados, sarcasmo o ironía)
    casos_dificiles_positivos = [
        "La comida era rica pero el mesero tardó 30 minutos; aun así el sabor compensó la espera y volvería.",
        "El hotel es un poco antiguo y la fachada no impresiona, pero la atención fue tan cálida que nos encantó.",
        "Al principio no aparecía nuestra reserva en https://hotel-ejemplo.com, pero el gerente lo resolvió rápido y fue excelente.",
        "Pagué 350 bolivianos que me pareció caro al inicio, aunque la calidad deliciosa de los platos realmente lo vale.",
        "La habitación era pequeña y el ascensor lento, sin embargo la limpieza impecable y el desayuno delicioso nos encantaron.",
        "No esperaba mucho de @restaurante_sabor por las quejas previas, pero no hubo ningún problema y todo estuvo delicioso.",
        "Hacía frío en la terraza y tardaron un poco, menos mal nos trajeron estufas al instante; muy buen servicio.",
        "Tuvimos que esperar 20 minutos en la fila, pero los platos deliciosos y el trato amable hicieron que valiera la pena.",
        "El estacionamiento es incómodo y estrecho, aunque dentro del hotel todo es limpio, cómodo y con excelente atención.",
        "No tenían el plato que pedí y el local estaba lleno, pero la recomendación del chef fue deliciosa y excelente.",
        "Aunque el baño era antiguo y pequeño, la cama era muy cómoda y el personal tan amable que pasamos una gran noche.",
        "El precio subió 50 bolivianos respecto al año pasado, pero siguen manteniendo el mejor sabor y excelente servicio.",
        "Había algo de ruido afuera y tardaron en traer la cuenta, pero la comida caliente y deliciosa lo compensó todo.",
        "Pensé que sería una mala experiencia por los comentarios negativos, pero comimos delicioso y nos trataron súper bien.",
        "No es el hotel más lujoso ni moderno, pero las habitaciones están limpias y te atienden con mucha amabilidad.",
        "La decoración es algo rara y el menú corto, sin embargo los postres y el café son deliciosos y 100% recomendados.",
        "Tardaron 25 minutos en traernos la entrada y hacía calor, pero el plato principal estaba tan delicioso que volveremos.",
        "Al llegar hubo confusión con la mesa y esperamos, pero nos pidieron disculpas amablemente y la cena fue maravillosa."
    ]

    casos_dificiles_negativos = [
        "La comida era rica pero el mesero fue muy grosero y tardó 2 horas, no recomiendo este lugar.",
        "El lugar es muy bonito y limpio para sacar fotos en @hotel_central, lástima que la comida sea fría, cruda y pésima.",
        "¡Qué maravilla esperar 2 horas por una comida fría de 200 bolivianos! Obviamente no volveré jamás.",
        "La ubicación del hotel es excelente y la fachada hermosa, pero las habitaciones estaban sucias y con cucarachas.",
        "El recepcionista fue muy amable y atento al recibirnos, pero la habitación estaba sucia, sin agua caliente y pésima.",
        "Muy bonita la página https://hotel-ejemplo.com y las fotos hermosas, qué pena que la habitación esté sucia y con mal olor.",
        "Los postres estaban ricos y el local limpio, pero el plato principal llegó crudo, frío y el mesero fue grosero.",
        "¡Cinco estrellas al chef que logró servir la comida fría y tardar 2 horas! Pésimo servicio, no volveré.",
        "El ambiente es acogedor y la vista hermosa, aunque no se puede disfrutar nada cuando la atención es pésima y grosera.",
        "Nos recibieron con una sonrisa muy amable, pero luego nos cobraron 500 bolivianos de más y la comida llegó fría.",
        "La vista desde la habitación es hermosa y la cama suave, lástima que el baño estaba sucio y no había agua caliente.",
        "Sí, el restaurante es bonito y famoso, pero el trato grosero y la espera de 2 horas hacen que no lo recomiende.",
        "El desayuno buffet se veía delicioso y variado, sin embargo todo estaba frío, sucio y nadie nos atendió.",
        "Parecía un excelente hotel al llegar, pero el ruido fue insoportable toda la noche y las sábanas estaban sucias.",
        "La atención inicial fue rápida y amable, pero la carne estaba dura, la sopa fría y el precio exagerado; mala experiencia.",
        "¡Felicitaciones a @restaurante_sabor por hacernos esperar 2 horas y traernos todo frío y crudo! Nunca volveremos.",
        "Todo parecía perfecto y muy limpio en recepción, pero en la habitación no funcionaba nada y el trato fue pésimo.",
        "Las instalaciones son hermosas y modernas, pero qué decepción tan grande el servicio lento, frío y cero recomendable."
    ]

    lista_resenas = []

    # Generamos 600 reseñas positivas (18 difíciles únicas + 582 variadas)
    for indice in range(600):
        canal_elegido = random.choice(canales_disponibles)
        if indice < len(casos_dificiles_positivos):
            texto_resena = casos_dificiles_positivos[indice]
        else:
            inicio = random.choice(inicios_positivos)
            detalle = random.choice(detalles_positivos)
            cierre = random.choice(cierres_positivos)
            modo_variacion = indice % 5
            if modo_variacion == 0:
                texto_resena = f"{inicio.upper()} {detalle} {cierre}"
            elif modo_variacion == 1:
                costo = random.randint(100, 450)
                horas = random.randint(1, 3)
                texto_resena = f"{inicio} {detalle} Pagamos {costo} bolivianos por {horas} horas de estancia. {cierre}"
            elif modo_variacion == 2:
                texto_resena = f"{inicio} {detalle} Más detalles en https://hotel-ejemplo.com/op{indice} con @hotel_central. {cierre}"
            elif modo_variacion == 3:
                texto_resena = f"¡{inicio} {detalle} {cierre}!"
            else:
                texto_resena = f"{inicio} {detalle} {cierre}"

        lista_resenas.append({
            "texto": texto_resena,
            "etiqueta": "positivo",
            "canal": canal_elegido
        })

    # Generamos 600 reseñas negativas (18 difíciles únicas + 582 variadas)
    for indice in range(600):
        canal_elegido = random.choice(canales_disponibles)
        if indice < len(casos_dificiles_negativos):
            texto_resena = casos_dificiles_negativos[indice]
        else:
            inicio = random.choice(inicios_negativos)
            detalle = random.choice(detalles_negativos)
            cierre = random.choice(cierres_negativos)
            modo_variacion = indice % 5
            if modo_variacion == 0:
                texto_resena = f"{inicio.upper()} {detalle} {cierre}"
            elif modo_variacion == 1:
                costo = random.randint(100, 450)
                horas = random.randint(1, 3)
                texto_resena = f"{inicio} {detalle} Pagamos {costo} bolivianos por {horas} horas de espera. {cierre}"
            elif modo_variacion == 2:
                texto_resena = f"{inicio} {detalle} Más detalles en https://hotel-ejemplo.com/op{indice} con @hotel_central. {cierre}"
            elif modo_variacion == 3:
                texto_resena = f"¡{inicio} {detalle} {cierre}!"
            else:
                texto_resena = f"{inicio} {detalle} {cierre}"

        lista_resenas.append({
            "texto": texto_resena,
            "etiqueta": "negativo",
            "canal": canal_elegido
        })

    # Mezclamos aleatoriamente todas las reseñas manteniendo la semilla fija
    random.shuffle(lista_resenas)

    ruta_directorio = os.path.dirname(os.path.abspath(__file__))
    ruta_archivo_csv = os.path.join(ruta_directorio, "resenas.csv")

    with open(ruta_archivo_csv, mode="w", encoding="utf-8", newline="") as archivo_salida:
        nombres_columnas = ["id", "texto", "etiqueta", "canal"]
        escritor_csv = csv.DictWriter(archivo_salida, fieldnames=nombres_columnas)
        escritor_csv.writeheader()

        for numero_id, fila_resena in enumerate(lista_resenas, start=1):
            escritor_csv.writerow({
                "id": numero_id,
                "texto": fila_resena["texto"],
                "etiqueta": fila_resena["etiqueta"],
                "canal": fila_resena["canal"]
            })

    print(f"Dataset generado exitosamente en: {ruta_archivo_csv}")
    print(f"Total de registros creados: {len(lista_resenas)}")


if __name__ == "__main__":
    generar_conjunto_datos()
