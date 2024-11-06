from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import re
import time
import skfuzzy as fuzz
import os

# Obtencion de requirimientos de la libreria nltk
os.system('python nltk_requirements.py')
# Obtenemos el valor de la variable de entorno DEBUG
DEBUG_MODE = os.getenv('DEBUG', '0') == '1'

if DEBUG_MODE:
   print("")
else:
    print("\n - Calculando los valores de cada oracion...") 


# Inicializar la lista de stopwords
stop_words = set(stopwords.words('english'))

# Variables para realizar el benchmark
import csv  # Asegúrate de importar csv en la parte superior
total_time = 0  # Para registrar el tiempo total de procesamiento
cant_positivo = 0
cant_negativo = 0
cant_neutro = 0
resultados = []  # Lista para almacenar los resultados de cada línea



def prepate_text(text,option):

    #para preparar el texto para su limpiaza...
    linea_nueva = ""
    for palabra in linea:
        if(option):
            if(palabra == "," or palabra == "\n" or (palabra == "0" or palabra == "1")):
                linea_nueva += " "
            else:
                linea_nueva += palabra
        else:
            if(palabra == "0" or palabra == "1"):
                return palabra 
    return linea_nueva


# Función para limpiar texto (remover letras sueltas, símbolos y stopwords)
def clean_text(text):
    # Tokenizar el texto
    tokens = word_tokenize(text)

    # Filtrar tokens alfabéticos, eliminar letras sueltas y stopwords
    tokens = [word for word in tokens if word.isalpha() and len(word) > 1 and word.lower() not in stop_words]
    
    # Unir los tokens filtrados en una oración
    return ' '.join(tokens)



#---------Valores para la fuzzificación/defuzzificación------
# Rango para salida: [0, 10] en puntos porcentuales
x_positivo = np.arange(0, 1, 0.1)
x_negativo = np.arange(0, 1, 0.1)
x_salida = np.arange(0, 10, 1)

# ----Funciones de pertenencia difusa----

# Funciones de pertenencia difusa para positivo
positivo_bajo = fuzz.trimf(x_positivo, [0, 0, 0.5])
positivo_medio = fuzz.trimf(x_positivo, [0, 0.5, 1])
positivo_alto = fuzz.trimf(x_positivo, [0.5, 1, 1])

# Funciones de pertenencia difusa para negativo
negativo_bajo = fuzz.trimf(x_negativo, [0, 0, 0.5])
negativo_medio = fuzz.trimf(x_negativo, [0, 0.5, 1])
negativo_alto = fuzz.trimf(x_negativo, [0.5, 1, 1])

# Funciones de pertenencia difusa para salida
salida_negativa = fuzz.trimf(x_salida, [0, 0, 5])
salida_neutral = fuzz.trimf(x_salida, [0, 5, 10])
salida_positiva = fuzz.trimf(x_salida, [5, 10, 10])


# Columnas Resultantes
with open ("archivo_nuevo.csv","w") as archivo:
    archivo.write("texto_original" + " , "  "laber_original " + " , " + "Positivo" + " , " + "Negativo" + " , " + "Sentimiento" + "\n")

#Limpiaza del texto(Pre-procesado....)
with open("test_data.csv") as archivo:

    archivo_limpio = open("archivo_nuevo.csv","a")

    # inicializacion de variables
    cantidad = 0
    sent_calculado = ""

    #Leemos linea por linea el archivo
    for linea in archivo:

        #inicio del proceso
        inicio_time = time.time()

        if(linea == 'sentence,sentiment\n'):
            continue

        # limpiamos el texto...    
        texto_limpio = clean_text(prepate_text(linea,1))
        

        # Inicializamos el analizador de sentimientos
        analyzer = SentimentIntensityAnalyzer()

        # Analizamos los sentimientos de la oracion...
        sentiment_scores = analyzer.polarity_scores(linea)
        positivo = sentiment_scores['pos']
        negativo = sentiment_scores['neg']


        # Calculamos los niveles de pertenencia negatica utilizando la libreria fuzz
        tiempo_inicio_fuzz = time.time()

        nivel_pos_bajo = fuzz.interp_membership(
            x_positivo, positivo_bajo, positivo
        )
        nivel_pos_medio = fuzz.interp_membership(
            x_positivo, positivo_medio, positivo
        )
        nivel_pos_alto = fuzz.interp_membership(
            x_positivo, positivo_alto, positivo
        )

        # Calculamos los niveles de pertenencia negatica utilizando la libreria fuzz
        nivel_neg_bajo = fuzz.interp_membership(
            x_negativo, negativo_bajo, negativo
        )
        nivel_neg_medio = fuzz.interp_membership(
            x_negativo, negativo_medio, negativo
        )
        nivel_neg_alto = fuzz.interp_membership(
            x_negativo, negativo_alto, negativo
        )


        # El operador OR significa que tomamos el máximo de estas dos.
        rule_1 = np.fmin(nivel_pos_bajo, nivel_neg_bajo)
        rule_2 = np.fmin(nivel_pos_medio, nivel_neg_bajo)
        rule_3 = np.fmin(nivel_pos_alto, nivel_neg_bajo)
        rule_4 = np.fmin(nivel_pos_bajo, nivel_neg_medio)
        rule_5 = np.fmin(nivel_pos_medio, nivel_neg_medio)
        rule_6 = np.fmin(nivel_pos_alto, nivel_neg_medio)
        rule_7 = np.fmin(nivel_pos_bajo, nivel_neg_alto)
        rule_8 = np.fmin(nivel_pos_medio, nivel_neg_alto)
        rule_9 = np.fmin(nivel_pos_alto, nivel_neg_alto)

        # Aplicacion de las reglas de Mamdani
        n1 = np.fmax(rule_4, rule_7)
        n2 = np.fmax(n1, rule_8)
        activacion_salida_bajo = np.fmin(n2, salida_negativa)

        neu1 = np.fmax(rule_1, rule_5)
        neu2 = np.fmax(neu1, rule_9)
        activacion_salida_medio = np.fmin(neu2, salida_neutral)

        p1 = np.fmax(rule_2, rule_3)
        p2 = np.fmax(p1, rule_6)
        activacion_salida_alto = np.fmin(p2, salida_positiva)

        salida_cero = np.zeros_like(x_salida)

        # Agregacion para calcular el sentimiento final (Verificamos cuál es el valor máximo de los tres conjuntos de salida).
        agregada = np.fmax(
            activacion_salida_bajo, np.fmax(activacion_salida_medio, activacion_salida_alto)
        )
        tiempo_fuzz = time.time() - tiempo_inicio_fuzz


        # Desfuzzificamos...
        tiempo_inicio_defuzz = time.time()

        #utilizamos el metodo del centroide para la defuzzificación
        salida = fuzz.defuzz(x_salida, agregada, "centroid")

        rou_salida = round(salida, 2)

        tiempo_defuzz = time.time() - tiempo_inicio_defuzz


        # Detalles : 3,33 < 6,66 < 10 ----> negative < neutral < positive
        if rou_salida > 0 and rou_salida < 3.33:
            sent_calculado = "Negativa"
            cant_negativo += 1

        elif rou_salida > 3.34 and rou_salida < 6.66:
            sent_calculado = "Neutra"
            cant_neutro += 1

        elif rou_salida > 6.67 and rou_salida < 10:
            sent_calculado = "Positiva"
            cant_positivo += 1


        # Medición del tiempo total que llevó calcular el texto
        exec_time = time.time() - inicio_time
        total_time += exec_time
        tiempo_total = tiempo_fuzz + tiempo_defuzz

        resultados.append([
            prepate_text(linea, 1), positivo, negativo, rou_salida, sent_calculado, exec_time
        ])

        # ---------------------IMPRESIONES-------------------------

        # impresion datos del tweet
        if(sent_calculado != "Neutra"):
            archivo_limpio.write(prepate_text(linea,1) + " , " + prepate_text(linea,0) + " , " + str(positivo) + " , " + str(negativo) + " , " + sent_calculado + "\n")
            cantidad += 1

        # ------------------------DEBUG-------------------------
        if DEBUG_MODE:
            print('Modo debug activado')
            print("texto pre-procesado:" + texto_limpio)
            print("valor del sentimiento : " + str(positivo) + "  " + str(negativo))
            print("defuzzificado:" + str(rou_salida))
            print("sent_calculado: " + str(sent_calculado))
            print(f"Tiempo de ejecución: {exec_time:.4f} segundos\n\n")

    archivo_limpio.close()



    # Benchmark creado en archivo csv
    with open("resultado_benchmark.csv", "w", newline='') as archivo_salida:
        writer = csv.writer(archivo_salida)
        writer.writerow([
            "Oración original", "Label original", "Puntaje Positivo", "Puntaje Negativo", 
            "Resultado de Inferencia", "Resultado Calculado", "Tiempo de Ejecución"
        ])
        writer.writerows(resultados)

    # Benchmark en terminal
    promedio_tiempo = total_time / len(resultados)
    print("\n- Benchmark")
    print(f"    * Total de positivos: {cant_positivo}")
    print(f"    * Total de negativos: {cant_negativo}")
    print(f"    * Total de neutrales: {cant_neutro}")
    print(f"    * Tiempo promedio de ejecución total: {promedio_tiempo:.4f} segundos")
