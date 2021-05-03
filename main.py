from random import *
import numpy as np
from math import *
import matplotlib.pyplot as plt


def data_generator():
    a = random() * 10
    b = random() * 10
    arr = np.array([[a], [b]])
    d = [arr, a*b]
    return d


def error(expected, real):
    return (expected - real)**2


def demonstration_error(expected, real):
    return expected - real


def error_derivative(expected, real):
    return 2*(real - expected)


def ReLU(v):
    res = []
    s = v.shape
    for j in v:
        res.append(max(j, j*0.01))
    return np.array(res).reshape(s)


def ReLU_der(v):
    res = []
    for it in v:
        if it >= 0:
            res.append(1)
        else:
            res.append(0.01)
    return np.diagflat(res)





first_layer_matrix = ((np.random.rand(16, 2) - 0.5 * np.ones((16, 2))) / sqrt(2/2)) * 1/2

first_bias_matrix = ((np.random.rand(16, 1) - 0.5 * np.ones((16, 1))) / sqrt(2/2)) * 1/2


second_layer_matrix = ((np.random.rand(16, 16) - 0.5 * np.ones((16, 16))) / sqrt(16/2)) * 15/16

second_bias_matrix = ((np.random.rand(16, 1) - 0.5 * np.ones((16, 1))) / sqrt(16/2)) * 1/16


third_layer_matrix = ((np.random.rand(8, 16) - 0.5 * np.ones((8, 16))) / sqrt(16/2)) * 15/16

third_bias_matrix = ((np.random.rand(8, 1) - 0.5 * np.ones((8, 1))) / sqrt(16/2)) * 1/16


fourth_layer_matrix = ((np.random.rand(8, 8) - 0.5 * np.ones((8, 8))) / sqrt(8/2)) * 7/8

fourth_bias_matrix = ((np.random.rand(8, 1) - 0.5 * np.ones((8, 1))) / sqrt(8/2)) * 1/8


fith_layer_matrix = ((np.random.rand(1, 8) - 0.5 * np.ones((1, 8))) / sqrt(1/2)) * 1/2
fith_bias_matrix = ((np.random.rand(1, 1) - 0.5 * np.ones((1, 1))) / sqrt(1/2)) * 1/2




result = []
L = 0.00005  # LEARNING RATE
FRIC = 0.7  # 1 - Friction coefficient

prev_first_matrix_grad = 0
prev_first_bias_grad = 0
prev_second_matrix_grad = 0
prev_second_bias_grad = 0
prev_third_matrix_grad = 0
prev_third_bias_grad = 0
prev_fourth_matrix_grad = 0
prev_fourth_bias_grad = 0
prev_fith_matrix_grad = 0
prev_fith_bias_grad = 0


for i in range(100000):
    first_matrix_grad = 0
    first_bias_grad = 0
    second_matrix_grad = 0
    second_bias_grad = 0
    third_matrix_grad = 0
    third_bias_grad = 0
    fourth_matrix_grad = 0
    fourth_bias_grad = 0
    fith_matrix_grad = 0
    fith_bias_grad = 0


    for j in range(1):
        data = data_generator()

        input_vector = data[0]
        expected_output = data[1]


        first_layer_output_UR = np.matmul(first_layer_matrix, input_vector) + first_bias_matrix

        first_layer_output = ReLU(first_layer_output_UR)


        second_layer_output_UR = np.matmul(second_layer_matrix, first_layer_output) + second_bias_matrix

        second_layer_output = ReLU(second_layer_output_UR)


        third_layer_output_UR = np.matmul(third_layer_matrix, second_layer_output) + third_bias_matrix

        third_layer_output = ReLU(third_layer_output_UR)


        fourth_layer_output_UR = np.matmul(fourth_layer_matrix, third_layer_output) + fourth_bias_matrix

        fourth_layer_output = ReLU(fourth_layer_output_UR)


        fith_layer_output = np.matmul(fith_layer_matrix, fourth_layer_output) + fith_bias_matrix



        E_der = error_derivative(expected_output, fith_layer_output)

        delta_5 = E_der
        delta_4 = np.matmul(delta_5, np.matmul(fith_layer_matrix, ReLU_der(fourth_layer_output_UR)))
        delta_3 = np.matmul(delta_4, np.matmul(fourth_layer_matrix, ReLU_der(third_layer_output_UR)))
        delta_2 = np.matmul(delta_3, np.matmul(third_layer_matrix, ReLU_der(second_layer_output_UR)))
        delta_1 = np.matmul(delta_2, np.matmul(second_layer_matrix, ReLU_der(first_layer_output_UR)))


        first_matrix_grad += L * np.matmul(delta_1.transpose(), input_vector.transpose())
        first_bias_grad += L * delta_1.transpose()
        second_matrix_grad += L * np.matmul(delta_2.transpose(), first_layer_output.transpose())
        second_bias_grad += L * delta_2.transpose()
        third_matrix_grad += L * np.matmul(delta_3.transpose(), second_layer_output.transpose())
        third_bias_grad += L * delta_3.transpose()
        fourth_matrix_grad += L * np.matmul(delta_4.transpose(), third_layer_output.transpose())
        fourth_bias_grad += L * delta_4.transpose()
        fith_matrix_grad += L * np.matmul(delta_5.transpose(), fourth_layer_output.transpose())
        fith_bias_grad += L * delta_5.transpose()

        first_matrix_grad += FRIC * prev_first_matrix_grad
        first_bias_grad += FRIC * prev_first_bias_grad
        second_matrix_grad += FRIC * prev_second_matrix_grad
        second_bias_grad += FRIC * prev_second_bias_grad
        third_matrix_grad += FRIC * prev_third_matrix_grad
        third_bias_grad += FRIC * prev_third_bias_grad
        fourth_matrix_grad += FRIC * prev_fourth_matrix_grad
        fourth_bias_grad += FRIC * prev_fourth_bias_grad
        fith_matrix_grad += FRIC * prev_fith_matrix_grad
        fith_bias_grad += FRIC * prev_fith_bias_grad

        if j == 0:
            print(demonstration_error(expected_output, fith_layer_output), "\n    <-    \n", input_vector, "\n\n")
        if i % 1 == 0:
            result.append(demonstration_error(expected_output, fith_layer_output)[0][0])





    fith_layer_matrix -= fith_matrix_grad
    fith_bias_matrix -= fith_bias_grad

    fourth_layer_matrix -= fourth_matrix_grad
    fourth_bias_matrix -= fourth_bias_grad

    third_layer_matrix -= third_matrix_grad
    third_bias_matrix -= third_bias_grad


    second_layer_matrix -= second_matrix_grad
    second_bias_matrix -= second_bias_grad


    first_layer_matrix -= first_matrix_grad
    first_bias_matrix -= first_bias_grad

    prev_first_matrix_grad = first_matrix_grad
    prev_first_bias_grad = first_bias_grad
    prev_second_matrix_grad = second_matrix_grad
    prev_second_bias_grad = second_bias_grad
    prev_third_matrix_grad = third_matrix_grad
    prev_third_bias_grad = third_bias_grad
    prev_fourth_matrix_grad = fourth_matrix_grad
    prev_fourth_bias_grad = fourth_bias_grad
    prev_fith_matrix_grad = fith_matrix_grad
    prev_fith_bias_grad = fith_bias_grad
    # print(error_derivative(expected_output, third_layer_output))
    # print(error(expected_output, third_layer_output))


np.savez('L100k.npz',
         first_layer_matrix=first_layer_matrix, first_bias_matrix=first_bias_matrix,
         second_layer_matrix=second_layer_matrix, second_bias_matrix=second_bias_matrix,
         third_layer_matrix=third_layer_matrix, third_bias_matrix=third_bias_matrix,
         fourth_layer_matrix=fourth_layer_matrix, fourth_bias_matrix=fourth_bias_matrix,
         fith_layer_matrix=fith_layer_matrix, fith_bias_matrix=fith_bias_matrix)


plt.plot(result)
plt.show()
