import os
import cv2
import face_recognition
import numpy as np

def analizar_imagen(nombre_imagen):
    carpeta_desaparecidos = "personas-buscadas"
    rostros_desaparecidos = []
    nombres = []


    for archivo in os.listdir(carpeta_desaparecidos):
        ruta = os.path.join(carpeta_desaparecidos, archivo)
        imagen = face_recognition.load_image_file(ruta)
        encoding = face_recognition.face_encodings(imagen)
        if encoding:
            rostros_desaparecidos.append(encoding[0])
            nombres.append(os.path.splitext(archivo)[0])

    ruta_imagen = os.path.join("imagenes-analizar", nombre_imagen)
    
    # Verificar si la imagen existe
    if not os.path.exists(ruta_imagen):
        print(f"Error: No se encuentra la imagen {nombre_imagen}")
        return

    # Cargar y procesar la imagen
    imagen = face_recognition.load_image_file(ruta_imagen)
    imagen_cv = cv2.imread(ruta_imagen)

    # Verificar si la imagen se cargó correctamente
    if imagen_cv is None:
        print(f"Error: No se pudo cargar la imagen: {ruta_imagen}")
        return

    # Detectar rostros
    face_locations = face_recognition.face_locations(imagen)
    encodings_encontrados = face_recognition.face_encodings(imagen, face_locations)

    # Procesar cada rostro encontrado
    for (top, right, bottom, left), encoding in zip(face_locations, encodings_encontrados):
        resultados = face_recognition.compare_faces(rostros_desaparecidos, encoding)
        
        if True in resultados:
            index = resultados.index(True)
            nombre = nombres[index]
            
            cv2.rectangle(imagen_cv, (left, top), (right, bottom), (0, 0, 255), 2)
            
            etiqueta = f"Desaparecido: {nombre}"
            (w, h), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            
            overlay = imagen_cv.copy()
            cv2.rectangle(overlay, (left, bottom), (left + w + 12, bottom + 25), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.5, imagen_cv, 0.5, 0, imagen_cv)
            
            # Agregar el texto
            cv2.putText(imagen_cv, etiqueta, (left + 6, bottom + 18), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            print(f"¡ALERTA! Persona desaparecida encontrada: {nombre}")
        else:
            # Marcar rostros no identificados solo con rectángulo verde
            cv2.rectangle(imagen_cv, (left, top), (right, bottom), (0, 255, 0), 2)

    # Crear una ventana con tamaño ajustable
    cv2.namedWindow("Análisis de Imagen", cv2.WINDOW_NORMAL)
    cv2.imshow("Análisis de Imagen", imagen_cv)
    
    # Esperar por una tecla
    key = cv2.waitKey(0)
    
    # Guardar la imagen si se presiona 's'
    if key == ord('s'):
        nombre_guardado = f"resultado_{nombre_imagen}"
        cv2.imwrite(nombre_guardado, imagen_cv)
        print(f"Imagen guardada como: {nombre_guardado}")
    
    cv2.destroyAllWindows()


nombre_imagen = "grupo-v2.jpg"  # Reemplaza con el nombre de tu imagen
analizar_imagen(nombre_imagen)
