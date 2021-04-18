import numpy as np
from math import *


def data_generator():
    input_vector = np.random.random(2)*10
    output_vector = input_vector[0] * input_vector[1]
    out = [input_vector, output_vector]
    # return out
    return [np.array([2, 2]), 4]


def input_to_body(input_vector, transformation_matrix):
    output_vector = np.dot(transformation_matrix, input_vector)
    return output_vector


def adding_biases(input_vector, bias_vector):
    output_vector = input_vector + bias_vector
    return output_vector


def elementwise_sigmoid_activation_function(input_vector):
    output_vector = []
    for i in np.array(input_vector):
        try:
            output_vector.append(1 / (1 + exp(-i)))
        except OverflowError:
            output_vector.append(0)
    output_vector = np.array(output_vector)
    return output_vector


def sigmoid_der(input_vector):
    return np.multiply(elementwise_sigmoid_activation_function(input_vector),
                       (1 - elementwise_sigmoid_activation_function(input_vector)))


# def softmax_function(input_vector):
#     sumexp = 0
#     output_vector = []
#     for i in input_vector:
#         sumexp += exp(i)
#
#     for i in input_vector:
#         output_vector.append(exp(i) / sumexp)
#
#     output_vector = np.array(output_vector)
#
#     return output_vector


def error(expected, real):
    return (expected - real) ** 2


def error_der(expected, real):
    return 2 * (real - expected)


def ReLU(input_vector):
    output_vector = []
    for i in input_vector:
        output_vector.append(max(0.01 * i, 1.01 * i))

    return np.array(output_vector)


def ReLU_der(input_vector):
    output_vector = []
    for i in input_vector:
        output_vector.append(int(i > 0) + 0.01)
    return output_vector


L = 0.00005


for j in range(100):
    first_MATRIX = np.random.rand(2, 2)
    first_bias = np.random.rand(2)

    second_MATRIX = np.random.rand(2, 2)
    second_bias = np.random.rand(2)

    final_MATRIX = np.random.rand(2)
    final_bias = np.random.random()



    for i in range(100000):

        data = data_generator()
        first_layer = np.dot(first_MATRIX, data[0]) + first_bias
        first_layer_ReLU = ReLU(first_layer)
        second_layer = np.dot(second_MATRIX, first_layer_ReLU) + second_bias
        second_layer_ReLU = ReLU(second_layer)
        final_layer = np.dot(final_MATRIX, second_layer_ReLU) + final_bias

        E = error(data[1], final_layer)
        E_der = error_der(data[1], final_layer)

        final_bias += E * L
        #final_MATRIX -= second_layer_ReLU.T * E * L
        #second_bias -= np.dot(ReLU_der(second_layer), final_MATRIX).T * E * L
        #second_MATRIX -= np.dot(first_layer_ReLU.T, np.dot(ReLU_der(second_layer), final_MATRIX).T) * E * L
        #first_bias -= np.dot(ReLU_der(first_layer), np.dot(second_MATRIX, np.dot(ReLU_der(second_layer), final_MATRIX).T)).T * E * L
        #first_MATRIX -= np.dot(data[0].T, np.dot(ReLU_der(first_layer), np.dot(second_MATRIX, np.dot(ReLU_der(second_layer), final_MATRIX).T)).T) * E * L


    print(E, "\n <---- \n", final_bias)


