from random import *
import numpy as np
from math import *

#       -->     [[],
#                []],      []
def data_generator():
    a = random() * 10
    b = random() * 10
    data = [np.array([[a], [b]]), a*b]
    return data


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
    for i in v:
        res.append(max(i, i*0.01))
    return np.array(res)


#     [[],         [[], 0, 0
#      [],   -->    0, [], 0
#      []]          0, 0, []]
def ReLU_der(v):
    res = []
    for i in v:
        if i >= 0:
            res.append(1)
        else:
            res.append(0.01)
    return np.diagflat(res)


def learning():
    data = data_generator()

    # [[],
    #  []]
    input_vector = data[0]

    # [[ , ],
    #  [ , ]]
    first_layer = np.random.rand(2, 2)
    # [[],
    #  []]
    first_bias_matrix = np.random.rand(1, 2)

    # [[ , ],
    #  [ , ]]
    second_layer = np.random.rand(2, 2)

    # [[],
    #  []]
    second_bias_matrix = np.random.rand(2)

    # [ , ]
    third_layer = np.random.rand(2)

    # [[]]
    third_bias_matrix = np.random.rand()

    # scalar
    expected_output = data[2]

    output = np.matmul(third_layer + third_bias_matrix,
                                np.matmul(second_layer + second_bias_matrix,
                                        np.matmul(first_layer + first_bias_matrix, input_vector)))



    print(output)






learning()