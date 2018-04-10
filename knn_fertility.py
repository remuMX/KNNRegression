import math
import operator
import json
import random

vectors = []

# Leer el archivo y convertir en vectores
with open("fertility.txt") as file:
    for line in file:
        line = line.strip()
        line = line.split(",")
        vectors.append(line)

random.shuffle(vectors)

# Hace dos plieges lo mas imetricamente posible
test = vectors[0:int(len(vectors) / 2):]
training = vectors[int(len(vectors) / 2):]


def analysis(test_parameter, training_parameter):
    # test are instances
    dictionary_test_1 = []

    for instance in test_parameter:
        instance_list = []
        param1, param2, param3, param4, param5, param6, param7, param8, param9, tag = instance

        for training_vector in training_parameter:
            distance = pow(float(param1) - float(training_vector[0]), 2) + \
                       pow(float(param2) - float(training_vector[1]), 2) + \
                       pow(float(param3) - float(training_vector[2]), 2) + \
                       pow(float(param4) - float(training_vector[3]), 2) + \
                       pow(float(param5) - float(training_vector[4]), 2) + \
                       pow(float(param6) - float(training_vector[5]), 2) + \
                       pow(float(param7) - float(training_vector[6]), 2) + \
                       pow(float(param8) - float(training_vector[7]), 2) + \
                       pow(float(param9) - float(training_vector[8]), 2)
            euclidean_distance = math.sqrt(distance)
            dictionary_element = {'Instance': instance, 
                                  'Vector': training_vector, 
                                  'Distance': euclidean_distance,
                                  'Class': training_vector[-1]}
            instance_list.append(dictionary_element)

        instance_list.sort(key=operator.itemgetter('Distance'))
        dictionary_test_1.append(instance_list)

    total_n, total_o, successes, successes_n, successes_o = 0, 0, 0, 0, 0

    for i in range(0, len(dictionary_test_1)):
        counts = 0
        status = False

        # Evalua la clase del vector de prueba
        if dictionary_test_1[i][0]['Instance'][-1] == 'N':
            total_n += 1
        if dictionary_test_1[i][0]['Instance'][-1] == 'O':
            total_o += 1

        # Evalua que la clase del vector entrenamiento sea la misma que el de prueba
        if dictionary_test_1[i][0]['Instance'][-1] == dictionary_test_1[i][0]['Class']:
            counts += 1
        if dictionary_test_1[i][1]['Instance'][-1] == dictionary_test_1[i][1]['Class']:
            counts += 1
        if dictionary_test_1[i][2]['Instance'][-1] == dictionary_test_1[i][2]['Class']:
            counts += 1
        if counts >= 2:
            successes += 1
            if dictionary_test_1[i][0]['Instance'][-1] == 'N':
                successes_n += 1
            if dictionary_test_1[i][0]['Instance'][-1] == 'O':
                successes_o += 1

    failed_n = total_n-successes_n
    failed_o = total_o-successes_o

    summary = {'Vectores Totales': len(dictionary_test_1), 
               'Vectores Acertados': successes,
               'Eficacia': (successes/len(dictionary_test_1)), 
               'Clase N Total': total_n,
               'Clase N Acertadas': successes_n,
               'Clase N Fallidas': failed_n,
               'Clase O Totales': total_o,
               'Clase O Acertadas': successes_o,
               'Clase O Fallidas': failed_o}
    return summary


resultado1 = analysis(test, training)
resultado2 = analysis(training, test)

class_n_success = (resultado1['Clase N Acertadas'] +
                   resultado2['Clase N Acertadas'])
class_n_failed = (resultado1['Clase N Fallidas'] +
                  resultado2['Clase N Fallidas'])
class_o_success = (resultado1['Clase O Acertadas'] +
                   resultado2['Clase O Acertadas'])
class_o_failed = (resultado1['Clase O Fallidas'] +
                  resultado2['Clase O Fallidas'])

confusion_matrix = [[class_n_success, class_n_failed],
                    [class_o_failed, class_o_success]]

print("============= Prueba 1 ==================")
print(json.dumps(resultado1, indent=1))
print()

print("============= Prueba 2 ==================")
print(json.dumps(resultado2, indent=1))
print()

print("=====Matriz de Confusion=====")
from pandas import DataFrame
print(DataFrame(confusion_matrix, 
                columns=['Clase N', 'Clase O'],
                index=['Clase N', 'Clase O']))
print()
accuracy = (class_n_success+class_o_success) / \
         (class_n_success+class_n_failed+class_o_success+class_o_failed)
print("Precisión = "+accuracy.__str__())
# input("Press any key...")
