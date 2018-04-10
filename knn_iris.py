import math
import operator
import json
import random

vectors = []
# Leer el archivo y convertir en vectores
with open("iris.txt") as file:
    for line in file:
        line = line.strip()
        line = line.split(",")
        vectors.append(line)

random.shuffle(vectors)

# Hace dos plieges lo mas simetricamente posible
test=vectors[0:int(len(vectors)/2)]
training=vectors[int(len(vectors)/2):]

def analysis(test_parameter, training_parameter):
    # cada vector del pliege de prueba se vuelve una instancia
    dictionary_test_1 = []

    for instance in test_parameter:
        instance_list = []
        # Cada elemento del vector de prueba se convierte en parametro para el calculo de la distancia
        param1, param2, param3, param4, tag = instance

        # En cada vector de entrenamiento se calcula la distancia
        for training_vector in training_parameter:
            distance = pow(float(param1) - float(training_vector[0]), 2) + \
                       pow(float(param2) - float(training_vector[1]), 2) + \
                       pow(float(param3) - float(training_vector[2]), 2) + \
                       pow(float(param4) - float(training_vector[3]), 2)
            euclidean_distance = math.sqrt(distance)
            # Se entrega un diccionario con el resumen de la prueba y se agrega a una lista
            dictionary_element = {'Instance': instance, 
                                  'Vector': training_vector, 
                                  'Distance': euclidean_distance,
                                  'Class': training_vector[-1]}
            instance_list.append(dictionary_element)

        # Las pruebas se ordenan para obtener k
        instance_list.sort(key=operator.itemgetter('Distance'))
        # La serie de pruebas se agrega como un conjunto de datos
        dictionary_test_1.append(instance_list)

    # Se declaran las variables que almacenaran las predicciones
    total_setosa, total_versicolor, total_virginica = 0, 0, 0
    setosa_setosa, setosa_versicolor, setosa_virginica = 0, 0, 0
    versicolor_setosa, versicolor_versicolor, versicolor_virginica = 0, 0, 0
    virginica_setosa, virginica_versicolor, virginica_virginica = 0, 0, 0
    # Y las predicciones exitosas
    successes = 0

    for i in range(0, len(dictionary_test_1)):
        # counts: Número de coincidencias en el vector de prueba y k
        counts = 0
        # Estas variables contaran si los aciertos son suficientes para determinar si la prediccion fue correcta
        count_setosa_setosa, count_setosa_versicolor, count_setosa_virginica = 0, 0, 0
        count_versicolor_setosa, count_versicolor_versicolor, count_versicolor_virginica = 0, 0, 0
        count_virginica_setosa, count_virginica_versicolor, count_virginica_virginica = 0, 0, 0

        # Evalua la clase del vector de prueba
        if dictionary_test_1[i][0]['Instance'][-1] == 'Iris-setosa':
            total_setosa += 1
        if dictionary_test_1[i][0]['Instance'][-1] == 'Iris-versicolor':
            total_versicolor += 1
        if dictionary_test_1[i][0]['Instance'][-1] == 'Iris-virginica':
            total_virginica += 1

        # Evalua que la clase k1, k2, k3  sea la misma que el de vector de prueba, sino, a que combinacion pertenece
        for k in range(0, 3):
            if dictionary_test_1[i][0]['Instance'][-1] == dictionary_test_1[i][k]['Class']:
                counts += 1
            if dictionary_test_1[i][0]['Instance'][-1] == 'Iris-setosa' \
                    and dictionary_test_1[i][k]['Class'] == 'Iris-setosa':
                count_setosa_setosa += 1
            if dictionary_test_1[i][0]['Instance'][-1] == 'Iris-setosa' \
                    and dictionary_test_1[i][k]['Class'] == 'Iris-versicolor':
                count_setosa_versicolor += 1
            if dictionary_test_1[i][0]['Instance'][-1] == 'Iris-setosa' \
                    and dictionary_test_1[i][k]['Class'] == 'Iris-virginica':
                count_setosa_virginica += 1
            if dictionary_test_1[i][0]['Instance'][-1] == 'Iris-versicolor' \
                    and dictionary_test_1[i][k]['Class'] == 'Iris-setosa':
                count_versicolor_setosa += 1
            if dictionary_test_1[i][0]['Instance'][-1] == 'Iris-versicolor' \
                    and dictionary_test_1[i][k]['Class'] == 'Iris-versicolor':
                count_versicolor_versicolor += 1
            if dictionary_test_1[i][0]['Instance'][-1] == 'Iris-versicolor' \
                    and dictionary_test_1[i][k]['Class'] == 'Iris-virginica':
                count_versicolor_virginica += 1
            if dictionary_test_1[i][0]['Instance'][-1] == 'Iris-virginica' \
                    and dictionary_test_1[i][k]['Class'] == 'Iris-setosa':
                count_virginica_setosa += 1
            if dictionary_test_1[i][0]['Instance'][-1] == 'Iris-virginica' \
                    and dictionary_test_1[i][k]['Class'] == 'Iris-versicolor':
                count_virginica_versicolor += 1
            if dictionary_test_1[i][0]['Instance'][-1] == 'Iris-virginica' \
                    and dictionary_test_1[i][k]['Class'] == 'Iris-virginica':
                count_virginica_virginica += 1

        # Revisar a que categoria pertenece si reunio los suficientes aciertos
        if count_setosa_setosa >= 2:
            setosa_setosa += 1
        if count_setosa_versicolor >= 2:
            setosa_versicolor += 1
        if count_setosa_virginica >= 2:
            setosa_virginica += 1
        if count_versicolor_setosa >= 2:
            versicolor_setosa += 1
        if count_versicolor_versicolor >= 2:
            versicolor_versicolor += 1
        if count_versicolor_virginica >= 2:
            versicolor_virginica += 1
        if count_virginica_setosa >= 2:
            virginica_setosa += 1
        if count_virginica_versicolor >= 2:
            virginica_versicolor += 1
        if count_virginica_virginica >= 2:
            virginica_virginica += 1
        if counts >= 2:
            successes += 1

    summary = {'Vectores Totales': len(dictionary_test_1),  # OK
               'Vectores Acertados': successes,  # OK
               'Eficacia': (successes/len(dictionary_test_1)),  # OK
               'Clase Setosa Total': total_setosa,  # OK
               'Clase Setosa Acertadas': setosa_setosa,  # OK
               'Clasificacion Setosa-Versicolor': setosa_versicolor,
               'Clasificacion Setosa-Virginica': setosa_virginica,
               'Clase Versicolor Totales': total_versicolor,  # OK
               'Clase Versicolor Acertadas': versicolor_versicolor,  # OK
               'Clasificacion Versicolor-Setosa': versicolor_setosa,
               'Clasificacion Versicolor-Virginica': versicolor_virginica,
               'Clase Virginica Totales': total_virginica,  # OK
               'Clase Virginica Acertadas': virginica_virginica,  # OK
               'Clasificacion Virginica-Setosa': virginica_setosa,
               'Clasificacion Virginica-Versicolor': virginica_versicolor
               }
    return summary


resultado1 = analysis(test, training)
resultado2 = analysis(training, test)

matrix_setosa_setosa = (resultado1['Clase Setosa Acertadas'] +
                        resultado2['Clase Setosa Acertadas'])
matrix_setosa_versicolor = (resultado1['Clasificacion Setosa-Versicolor'] +
                            resultado2['Clasificacion Setosa-Versicolor'])
matrix_setosa_virginica = (resultado1['Clasificacion Setosa-Virginica'] +
                           resultado2['Clasificacion Setosa-Virginica'])

matrix_versicolor_setosa = (resultado1['Clasificacion Versicolor-Setosa'] +
                            resultado2['Clasificacion Versicolor-Setosa'])
matrix_versicolor_versicolor = (resultado1['Clase Versicolor Acertadas'] +
                                resultado2['Clase Versicolor Acertadas'])
matrix_versicolor_virginica = (resultado1['Clasificacion Versicolor-Virginica'] +
                               resultado2['Clasificacion Versicolor-Virginica'])

matrix_virginica_setosa = (resultado1['Clasificacion Virginica-Setosa'] +
                           resultado2['Clasificacion Virginica-Setosa'])
matrix_virginica_versicolor = (resultado1['Clasificacion Virginica-Versicolor'] +
                               resultado2['Clasificacion Virginica-Versicolor'])
matrix_virginica_virginica = (resultado1['Clase Virginica Acertadas'] +
                              resultado2['Clase Virginica Acertadas'])

confusion_matrix = [[matrix_setosa_setosa, matrix_setosa_versicolor, matrix_setosa_virginica],
                    [matrix_versicolor_setosa, matrix_versicolor_versicolor, matrix_versicolor_virginica],
                    [matrix_virginica_setosa, matrix_virginica_versicolor, matrix_virginica_virginica]]

print("============= Prueba 1 ==================")
print(json.dumps(resultado1, indent=1))
print()

print("============= Prueba 2 ==================")
print(json.dumps(resultado2, indent=1))
print()

print("=====Matriz de Confusion=====")
from pandas import DataFrame
print(DataFrame(confusion_matrix, 
                columns=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
                index=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']))
print()
accuracy = (matrix_setosa_setosa + matrix_virginica_virginica + matrix_versicolor_versicolor) / \
          (matrix_setosa_setosa + matrix_setosa_versicolor + matrix_setosa_virginica +
           matrix_versicolor_setosa + matrix_versicolor_versicolor + matrix_versicolor_virginica +
           matrix_virginica_setosa + matrix_virginica_versicolor + matrix_virginica_virginica)
print("Eficiencia ="+accuracy.__str__())
input("Press any key...")
