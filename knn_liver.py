import math
import operator
import json
import random

vectors = []

# Leer el archivo y convertir en vectores
with open("liver.txt") as file:
    for line in file:
        line = line.strip()
        line = line.split(",")
        vectors.append(line)

# Los vectores son ordenados de manera aleatoria
random.shuffle(vectors)

test = vectors[0:int(len(vectors) / 2)]
training = vectors[int(len(vectors) / 2):]


def analysis(test_parameter, training_parameter):
    # test are instances
    dictionary_test_1 = []

    for instance in test_parameter:
        instance_list = []
        param1, param2, param3, param4, param5, param6, tag = instance

        for training_vector in training_parameter:
            distance = pow(float(param1) - float(training_vector[0]), 2) + \
                       pow(float(param2) - float(training_vector[1]), 2) + \
                       pow(float(param3) - float(training_vector[2]), 2) + \
                       pow(float(param4) - float(training_vector[3]), 2) + \
                       pow(float(param5) - float(training_vector[4]), 2) + \
                       pow(float(param6) - float(training_vector[5]), 2)
            euclidean_distance = math.sqrt(distance)
            dictionary_element = {'Instance': instance, 
                                  'Vector': training_vector, 
                                  'Distance': euclidean_distance,
                                  'Class': training_vector[-1]}
            instance_list.append(dictionary_element)

        instance_list.sort(key=operator.itemgetter('Distance'))
        dictionary_test_1.append(instance_list)

    total_a, total_b, successes, successes_a, successes_b = 0, 0, 0, 0, 0

    for i in range(0, len(dictionary_test_1)):
        counts = 0

        # Evalua la clase del vector de prueba
        if dictionary_test_1[i][0]['Instance'][6] == '1':
            total_a += 1
        if dictionary_test_1[i][0]['Instance'][6] == '2':
            total_b += 1

        # Evalua que la clase del vector entrenamiento sea la misma que el de prueba
        if dictionary_test_1[i][0]['Instance'][6] == dictionary_test_1[i][0]['Class']:
            counts += 1
        if dictionary_test_1[i][1]['Instance'][6] == dictionary_test_1[i][1]['Class']:
            counts += 1
        if dictionary_test_1[i][2]['Instance'][6] == dictionary_test_1[i][2]['Class']:
            counts += 1
        if counts >= 2:
            successes += 1
            if dictionary_test_1[i][0]['Instance'][6] == '1':
                successes_a += 1
            if dictionary_test_1[i][0]['Instance'][6] == '2':
                successes_b += 1

    failed_a = total_a-successes_a
    failed_b = total_b-successes_b

    summary = {'Vectores Totales': len(dictionary_test_1), 
               'Vectores Acertados': successes,
               'Eficacia': (successes/len(dictionary_test_1)), 
               'Clase 1 Total': total_a,
               'Clase 1 Acertadas': successes_a, 
               'Clase 1 Fallidas': failed_a,
               'Clase 2 Totales': total_b, 
               'Clase 2 Acertadas': successes_b, 
               'Clase 2 Fallidas': failed_b}
    return summary


resultado1 = analysis(test, training)
resultado2 = analysis(training, test)

class_a_success = (resultado1['Clase 1 Acertadas'] +
                   resultado2['Clase 1 Acertadas'])
class_a_failed = (resultado1['Clase 1 Fallidas'] +
                  resultado2['Clase 1 Fallidas'])
class_b_success = (resultado1['Clase 2 Acertadas'] +
                   resultado2['Clase 2 Acertadas'])
class_b_failed = (resultado1['Clase 2 Fallidas'] +
                  resultado2['Clase 2 Fallidas'])

confusion_matrix = [[class_a_success, class_a_failed], 
                    [class_b_failed, class_b_success]]

print("============= Prueba 1 ==================")
print(json.dumps(resultado1, indent=1))
print()

print("============= Prueba 2 ==================")
print(json.dumps(resultado2, indent=1))
print()

print("=====Matriz de Confusion=====")
from pandas import DataFrame
print(DataFrame(confusion_matrix, 
                columns=['Clase 1', 'Clase 2'],
                index=['Clase 1', 'Clase 2']))
print()
accuracy = (class_a_success+class_b_success) / \
         (class_a_success+class_a_failed+class_b_success+class_b_failed)
print("Accuracy = "+accuracy.__str__())
input("Press any key...")
