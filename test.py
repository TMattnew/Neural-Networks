import numpy as np
from random import *
import matplotlib.pyplot as plt


def demonstration_error(expected, real):
    return expected - real


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

l = []
for i in range(1000):
    l.append([])
    for j in range(1000):
        input_vector = np.array([[i/100], [j/100]])
        first_layer_output_UR = np.matmul(first_layer_matrix, input_vector) + first_bias_matrix

        first_layer_output = ReLU(first_layer_output_UR)

        second_layer_output_UR = np.matmul(second_layer_matrix, first_layer_output) + second_bias_matrix

        second_layer_output = ReLU(second_layer_output_UR)

        third_layer_output_UR = np.matmul(third_layer_matrix, second_layer_output) + third_bias_matrix

        third_layer_output = ReLU(third_layer_output_UR)

        fourth_layer_output_UR = np.matmul(fourth_layer_matrix, third_layer_output) + fourth_bias_matrix

        fourth_layer_output = ReLU(fourth_layer_output_UR)

        fith_layer_output = np.matmul(fith_layer_matrix, fourth_layer_output) + fith_bias_matrix


        l[i].append(demonstration_error(i*j/10000, fith_layer_output[0][0]))
        print('i, j = ', i, ', ', j, ' ERROR = ', demonstration_error(i*j/10000, fith_layer_output[0][0]))

l = np.array(l)


fig, (ax0) = plt.subplots(1, 1)

c = ax0.pcolor(l, cmap='RdBu')
ax0.set_title('L=100k')

fig.tight_layout()
plt.show()
