import numpy as np
from random import *


def ReLU(v):
    res = []
    s = v.shape
    for j in v:
        res.append(max(j, j*0.01))
    return np.array(res).reshape(s)


def data_generator():
    a = random() * 10
    b = random() * 10
    arr = np.array([[a], [b]])
    d = [arr, a*b]
    return d


data = data_generator()

input_vector = data[0]
expected_output = data[1]

with np.load('L100k.npz') as matrices:
    first_layer_matrix = matrices['first_layer_matrix']

    first_bias_matrix = matrices['first_bias_matrix']


    second_layer_matrix = matrices['second_layer_matrix']

    second_bias_matrix = matrices['second_bias_matrix']


    third_layer_matrix = matrices['third_layer_matrix']

    third_bias_matrix = matrices['third_bias_matrix']


    fourth_layer_matrix = matrices['fourth_layer_matrix']

    fourth_bias_matrix = matrices['fourth_bias_matrix']


    fith_layer_matrix = matrices['fith_layer_matrix']
    fith_bias_matrix = matrices['fith_bias_matrix']


    first_layer_output_UR = np.matmul(first_layer_matrix, input_vector) + first_bias_matrix

    first_layer_output = ReLU(first_layer_output_UR)


    second_layer_output_UR = np.matmul(second_layer_matrix, first_layer_output) + second_bias_matrix

    second_layer_output = ReLU(second_layer_output_UR)


    third_layer_output_UR = np.matmul(third_layer_matrix, second_layer_output) + third_bias_matrix

    third_layer_output = ReLU(third_layer_output_UR)


    fourth_layer_output_UR = np.matmul(fourth_layer_matrix, third_layer_output) + fourth_bias_matrix

    fourth_layer_output = ReLU(fourth_layer_output_UR)


    fith_layer_output = np.matmul(fith_layer_matrix, fourth_layer_output) + fith_bias_matrix
