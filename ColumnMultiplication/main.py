import numpy as np
import matplotlib as mpl
import math
import random

'''This will use a vector of 60 values to represent a 6-digit answer to the problem (****,**)'''


def decompose(data_list):
    decomposed_matrix = []
    for i in range(6):
        decomposed_matrix.append([0] * 10)

    for i in range(6):
        decomposed_matrix[i][data_list[i]] = 1.

    return np.array(decomposed_matrix).flatten().reshape((60, 1))


def concentrate(data_list):

    binary_output = [0]*60

    for i in range(6):
        M_value = 0
        M_index = -1
        for j in range(10):
            if data_list[i*10 + j] > M_value:
                M_value = data_list[i*10 + j][0]
                M_index = i*10 + j
        binary_output[M_index] = 1

    return np.array(binary_output).reshape((60, 1))



def generate_data():
    deconstructed_input = []
    for i in range(6):
        deconstructed_input.append(random.randint(0, 9) * 1.)

    constructed_input = []
    constructed_input.append(deconstructed_input[0] * 10 + deconstructed_input[1] + deconstructed_input[2] * 0.1)
    constructed_input.append(deconstructed_input[3] * 10 + deconstructed_input[4] + deconstructed_input[5] * 0.1)

    constructed_output = round(constructed_input[0] * constructed_input[1], 2)
    deconstructed_output = []
    constructed_outputs_copy = constructed_output

    for i in range(6):
        next_digit = math.floor(round(constructed_outputs_copy / (10 ** (3 - i)), 5))
        deconstructed_output.append(next_digit)
        constructed_outputs_copy -= next_digit * 10 ** (3 - i)

    return constructed_input, np.array(deconstructed_input), constructed_output, deconstructed_output


def initialize_matrices():
    latrices = []
    batrices = []

    latrices.append(np.random.randn(64, 6) * math.sqrt(2 / 6) * 6 / 7)
    batrices.append(np.random.randn(64, 1) * 1 / 7)

    latrices.append(np.random.randn(64, 64) * math.sqrt(2 / 64) * 64 / 65)
    batrices.append(np.random.randn(64, 1) * 1 / 65)

    latrices.append(np.random.randn(64, 64) * math.sqrt(2 / 64) * 64 / 65)
    batrices.append(np.random.randn(64, 1) * 1 / 65)

    latrices.append(np.random.randn(60, 64) * math.sqrt(6 / 124) * 64 / 65)
    batrices.append(np.random.randn(60, 1) * 1 / 65)
    return latrices, batrices


def ReLU(vector):
    result_vector = []
    for i in vector:
        if i[0] < 0:
            result_vector.append([i[0] * 0.001])
        else:
            result_vector.append([i[0]])
    return vector


def ReLU_der(vector):
    result_vector = []
    for i in vector:
        if i[0] < 0:
            result_vector.append([0.001])
        else:
            result_vector.append(1.)
    return np.diagflat(result_vector)


def sigmoid(vector):
    result_vector = []
    for i in vector:
        result_vector.append([sigmoid_scalar(i[0])])
    return np.array(result_vector)


def sigmoid_scalar(s):
    try:
        return 1 / (1 + math.exp(-s))
    except OverflowError:
        if s > 0:
            return 1
        if i < 0:
            return 0


def sigmoid_der(vector):
    result_vector = []
    for i in vector:
        result_vector.append((sigmoid_scalar(i[0])) * (1 - sigmoid_scalar(i[0])))
    return np.diagflat(result_vector)


def ERROR(expected_vector, input_vector):
    output_vector = []
    for i in range(60):
        output_vector.append((expected_vector[i][0] - input_vector[i][0])**2)
    return np.array([[sum(output_vector)]])


def ERROR_der(expected_vector, input_vector):
    output_vector = []
    for i in range(60):
        output_vector.append(2 * (input_vector[i][0] - expected_vector[i][0]))
    return np.array(output_vector)

# def generate_data():
#     inp = np.random.rand(2, 1) * 100
#     inp[0][0] = round(inp[0][0], 1)
#     inp[1][0] = round(inp[1][0], 1)
#     deconstructed_inp = []
#
#     digit = int(inp[0][0] / 10)
#     deconstructed_inp.append([digit])
#     inp[0][0] -= digit
#
#     digit = int(inp[0][0])
#     deconstructed_inp.append([digit])
#     inp[0][0] -= digit
#
#     digit = int(inp[0][0]*10)
#     deconstructed_inp.append([digit])
#     inp[0][0] -= digit
#
#     inp[0][0] = round(inp[0][0], 2)
#
#     digit = int(inp[0][0]*100)
#     deconstructed_inp.append([digit])
#     inp[0][0] -= digit
#
#     inp[0][0] = round(inp[0][0], 2)a
#
#     real_num_output = round(inp[0][0] * inp[1][0], 2)
#
#     out = [[0] * 10, [0] * 10, [0] * 10, [0] * 10, [0] * 10, [0] * 10]
#
#     copy = real_num_output
#
#     thousands = int(copy / 1000)
#     copy -= thousands * 1000
#
#     copy = round(copy, 2)
#
#     hundreds = int(copy / 100)
#     copy -= hundreds * 100
#
#     copy = round(copy, 2)
#
#     tens = int(copy / 10)
#     copy -= tens * 10
#
#     copy = round(copy, 2)
#
#     ones = int(copy)
#     copy -= ones
#
#     copy = round(copy, 2)
#
#     tenths = int(copy * 10)
#     copy -= tenths / 10
#
#     copy = round(copy, 2)
#
#     hundredths = int(copy * 100)
#     copy -= hundredths / 100
#
#     out[0][thousands] = 1
#     out[1][hundreds] = 1
#     out[2][tens] = 1
#     out[3][ones] = 1
#     out[4][tenths] = 1
#     out[5][hundredths] = 1
#     return inp, np.array(out)
#
#
# def construct_data(deconstructed):
#     dig = -1  # Just to fix this stupid warning
#     deconstructed = deconstructed.tolist()
#     real_number = 0
#     real_number = deconstructed[0].index(1) * 1000 + deconstructed[1].index(1) * 100 + deconstructed[2].index(1) * 10 + \
#                   deconstructed[3].index(1) + deconstructed[4].index(1) * 0.1 + deconstructed[5].index(1) * 0.01
#
#     real_number = round(real_number, 2)
#
#     return real_number
#
#
# def definiter(vector):
#     definite_vector = []
#     shape = vector.shape
#     for i in range(shape[0]):
#         M = 0
#         Mint = -1
#         for j in range(shape[1]):
#             if vector[i, j] >= M:
#                 M = vector[i, j]
#                 Mint = j
#         definite_vector.append([0]*10)
#         definite_vector[i][Mint] = 1
#     definite_vector = np.array(definite_vector).reshape(shape)
#     return definite_vector
#
#
#
# g = generate_data()
#
#
# print(g[0].flatten(), " => ", construct_data(g[1]), '\n')
# print(g[1])


L = 0.8


latrices, batrices = initialize_matrices()
delta = [None]*4


for i in range(1000):
    g = generate_data()

    expexted_output = decompose(g[3])
    input_vector = g[1]

    zeroth = np.matmul(latrices[0], input_vector).reshape((64, 1)) + batrices[0]
    zeroth_A = ReLU(zeroth)
    first = np.matmul(latrices[1], zeroth).reshape((64, 1)) + batrices[1]
    first_A = ReLU(first)
    second = np.matmul(latrices[2], first).reshape((64, 1)) + batrices[2]
    second_A = ReLU(second)
    output_vector = np.matmul(latrices[3], first).reshape((60, 1)) + batrices[3]
    output_vector_A = sigmoid(output_vector)

    E_der = ERROR_der(expexted_output, output_vector_A)
    delta[3] = np.matmul(E_der, sigmoid_der(output_vector))
    latrices[3] -= L * np.matmul(delta[3].reshape((60, 1)), second_A.reshape((1, 64)))
    batrices[3] -= L * delta[3].reshape((60, 1))
    print(ERROR(expexted_output, output_vector_A))



    c = concentrate(output_vector)
