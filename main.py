from random import *
import numpy as np
from math import *


#  rand  -->    [[],
#                []],      []
def data_generator():
    a = random() * 10
    b = random() * 10
    arr = np.array([[a], [b]])
    d = [arr, a*b]
    return d


#     [] --> []
def error(expected, real):
    return (expected - real)**2


#     [] --> []
def error_derivative(expected, real):
    return 2*(real - expected)


#     [[],         [[],
#      [],   -->    [],
#      []]          []]
def ReLU(v):
    res = []
    s = v.shape
    for j in v:
        res.append(max(j, j*0.01))
    return np.array(res).reshape(s)


#     [[],         [[], 0, 0
#      [],   -->    0, [], 0
#      []]          0, 0, []]
def ReLU_der(v):
    res = []
    for it in v:
        if it >= 0:
            res.append(1)
        else:
            res.append(0.01)
    return np.diagflat(res)




# [[ , ],
#  [ , ]]
first_layer_matrix = np.random.rand(16, 2)

# [[],
#  []]
first_bias_matrix = np.random.rand(16, 1)

# [[ , ],
#  [ , ]]
second_layer_matrix = np.random.rand(8, 16)

# [[],
#  []]
second_bias_matrix = np.random.rand(8, 1)

# [[ , ]]
third_layer_matrix = np.random.rand(1, 8)


# [[]]
third_bias_matrix = np.random.rand(1, 1)



L = 0.0001  # LEARNING RATE


for i in range(100000):
    data = data_generator()

    # [[],
    #  []]
    input_vector = data[0]

    # scalar
    expected_output = data[1]


    # [[],
    #  []]
    first_layer_output_UR = np.matmul(first_layer_matrix, input_vector) + first_bias_matrix

    # [[],
    #  []]
    first_layer_output = ReLU(first_layer_output_UR)

    # [[],
    #  []]
    second_layer_output_UR = np.matmul(second_layer_matrix, first_layer_output) + second_bias_matrix


    # [[],
    #  []]
    second_layer_output = ReLU(second_layer_output_UR)


    # []
    third_layer_output = np.matmul(third_layer_matrix, second_layer_output) + third_bias_matrix



    E_der = error_derivative(expected_output, third_layer_output)

    delta_3 = E_der
    delta_2 = np.matmul(delta_3, np.matmul(third_layer_matrix, ReLU_der(second_layer_output_UR)))
    delta_1 = np.matmul(delta_2, np.matmul(second_layer_matrix, ReLU_der(first_layer_output_UR)))


    third_bias_matrix -= L * delta_3.transpose()
    third_layer_matrix -= L * np.matmul(delta_3.transpose(), second_layer_output.transpose())

    second_bias_matrix -= L * delta_2.transpose()
    second_layer_matrix -= L * np.matmul(delta_2.transpose(), first_layer_output.transpose())

    first_bias_matrix -= L * delta_1.transpose()
    first_layer_matrix -= L * np.matmul(delta_1.transpose(), input_vector.transpose())

    # print(error_derivative(expected_output, third_layer_output))
    print(expected_output, " -> ", third_layer_output)
    # print(error(expected_output, third_layer_output))



