import numpy as np
import matplotlib.pyplot as plt
import methods

dataset = np.loadtxt('data/tren_data1___16.txt')
np.random.shuffle(dataset)
data = dataset[0:399, :]
test = dataset[400:498, :]

activation_function = 'unipolar'
if activation_function == 'bipolar':
    U = np.array([[1, -1, -1, -1, -1],
                 [-1, 1, -1, -1, -1],
                 [-1, -1, 1, -1, -1],
                 [-1, -1, -1, 1, -1],
                 [-1, -1, -1, -1, 1]])
else:
    U = np.array([[1, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0],
                 [0, 0, 1, 0, 0],
                 [0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 1]])

# params
neurons_count = 4
W = np.random.randn(neurons_count, 2)  # nahodné váhy první vrstvy
b = np.random.randn(4, 1)  # náhodné prahy první vrstvy
V = np.random.randn(5, neurons_count)  # nahodné váhy druhé vrstvy
d = np.random.randn(5, 1)  # náhodné prahy druhé vrstvy

lr = 0.01
error_max = 1
error = 0
epochs = 100
errors = []

for i in range(epochs):
    for k in range(len(data)):
        x = data[k, 0:2].reshape(-1,1)
        xi = np.dot(W, x) + b
        z = methods.sigmoid(xi)
        zi = np.dot(V, z) + d
        y = methods.sigmoid(zi)
        u = methods.get_wanted(data, k, U).reshape(-1,1)

        dz = methods.diff(z)
        dy = methods.diff(y)

        part1 = lr * ((u - y)* dy* V).reshape(-1,1)
        print(part1)
        part2 = dz@x.reshape(-1,1)
        W = W + part1@part2


