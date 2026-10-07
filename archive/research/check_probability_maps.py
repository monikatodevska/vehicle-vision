import numpy as np

txt_path_elena=r'D:\Monika\VideosTest\slika_test\mil_res\310__cls.txt'
txt_path_monika=r'D:\Monika\VideosTest\slika_test\mil_res\prob_cls1.txt'
# Read probabilities from the text file
with open(txt_path_elena, 'r') as file:
    probabilities_text = file.read()

# Split probabilities by comma and convert them into floats
probabilities = [float(prob.strip()) for prob in probabilities_text.split(',')]

# Reshape probabilities into a 3D matrix
prob_cls=probabilities[0:42*64*4]
matrix = np.array(prob_cls).reshape(42, 64, 4)

# Print the shape of the matrix
print("Shape of the matrix:", matrix.shape)

# Optionally, print the matrix
# print(matrix)




with open(txt_path_monika, 'r') as file:
    probabilities_text_m = file.read()

# Split probabilities by comma and convert them into floats
probabilities_m = [float(prob.strip()) for prob in probabilities_text_m.split(',')]

# Reshape probabilities into a 3D matrix
matrix_m = np.array(probabilities_m).reshape((42, 64, 4))

# Print the shape of the matrix
print("Shape of the matrix:", matrix_m.shape)

# Optionally, print the matrix
# print(matrix_m)


dif=matrix-matrix_m

from copy import deepcopy
import matplotlib.pyplot as plt

# dif_flat = deepcopy(dif)
# dif_flat = dif_flat.flatten()
#
# plt.hist(dif_flat, bins = 100)
# plt.show()

are_equal = np.array_equal(matrix, matrix_m)


if are_equal:
    print("The matrices are identical.")
else:
    print("The matrices are not identical.")
    print(np.max(dif))
    max_index = np.unravel_index(np.argmax(dif), dif.shape)
    print(max_index)
    print(matrix[max_index[0], max_index[1], max_index[2]])
    print(matrix_m[max_index[0], max_index[1], max_index[2]])
