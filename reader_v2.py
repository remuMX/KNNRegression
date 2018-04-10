import math
import operator
import json

vectors = []
nearest = []
count = 0

with open("liver.txt") as file:
    for line in file:
        line = line.strip()
        line = line.split(",")
        vectors.append(line)
        count += 1

vectors_class_a = []
vectors_class_b = []

for vector in vectors:
    if vector[6] == '1':
        vectors_class_a.append(vector)
    if vector[6] == '2':
        vectors_class_b.append(vector)

test = vectors_class_a[0:int(len(vectors_class_a) / 2)]
test += vectors_class_b[0:int(len(vectors_class_b) / 2)]

training = vectors_class_a[int(len(vectors_class_a) / 2):]
training += vectors_class_b[int(len(vectors_class_b) / 2):]


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
            dictionary_element = {'Instance': instance, 'Vector': training_vector, 'Distance': euclidean_distance,
                                  'Class': training_vector[-1]}
            instance_list.append(dictionary_element)
        instance_list.sort(key=operator.itemgetter('Distance'))
        dictionary_test_1.append(instance_list)

    total_a = 0
    total_b = 0
    successes = 0
    successes_a = 0
    successes_b = 0
    for i in range(0, len(dictionary_test_1)):
        counts, counts_a, counts_b, counter = 0, 0, 0, 0
        status = False

        if dictionary_test_1[i][0]['Instance'][6] == '1':
            total_a += 1
        if dictionary_test_1[i][0]['Instance'][6] == '2':
            total_b += 1

        while counts < 3:
            # print(dictionary_test_1[i][counter])
            if dictionary_test_1[i][counter]['Class'] == '1':
                counts_a += 1
                counts = counts_a
            if dictionary_test_1[i][counter]['Class'] == '2':
                counts_b += 1
                counts = counts_b
            counter += 1

        if counts_a >= 3:
            if dictionary_test_1[i][0]['Instance'][6] == '1':
                successes_a += 1
                status = True
        if counts_b >= 3:
            if dictionary_test_1[i][0]['Instance'][6] == '2':
                successes_b += 1
                status = True

        if status:
            successes += 1

    failed_a = total_a-successes_a
    failed_b = total_b-successes_b

    summary = {'Vectores Totales': len(dictionary_test_1), 'Vectores Acertados': successes,
               'Eficacia': (successes/len(dictionary_test_1)), 'Clase 1 Total': total_a,
               'Clase 1 Acertadas': successes_a, 'Clase 1 Fallidas': failed_a,
               'Clase 2 Totales': total_b, 'Clase 2 Acertadas': successes_b, 'Clase 2 Fallidas': failed_b}
    return summary


resultado1 = analysis(test, training)
resultado2 = analysis(training, test)

class_a_success = (resultado1['Clase 1 Acertadas']+resultado2['Clase 1 Acertadas'])
class_a_failed = (resultado1['Clase 1 Fallidas']+resultado2['Clase 1 Fallidas'])
class_b_success = (resultado1['Clase 2 Acertadas']+resultado2['Clase 2 Acertadas'])
class_b_failed = (resultado1['Clase 2 Fallidas']+resultado2['Clase 2 Fallidas'])

confusion_matrix = [[class_a_success, class_a_failed], [class_b_failed, class_b_success]]

print("============= Prueba 1 ==================")
print(json.dumps(resultado1, indent=1))
print()

print("============= Prueba 2 ==================")
print(json.dumps(resultado2, indent=1))
print()



print("=====Matriz de Confusion=====")
from pandas import DataFrame
print(DataFrame(confusion_matrix, columns=['Clase1', 'Clase2'], index=['Clase 1', 'Clase 2']))
