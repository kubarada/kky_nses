import numpy as np
import matplotlib.pyplot as plt
import methods

data = np.loadtxt('data/tren_data1___16.txt')
activation_function = 'sigmoid'
if activation_function == 'sigmoid':
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
W = np.random.randn(5, 2) # nahodné váhy
b = np.random.randn(5, 1) # náhodné prahy
lr = 0.01
error_max = 1
error = 0
epochs = 100
errors = []
for i in range(epochs):
    for k in range(len(data)):
        x = data[k, 0:2].reshape(-1,1)
        xi = np.dot(W, x) + b
        y = np.sign(xi)
        u = methods.get_wanted(data, k, U).reshape(-1,1)
        error = error + np.dot(1/2, np.dot((u - y).reshape(1,-1),(u-y)))
        print(error)
        W = W + lr * (u - y) * x.reshape(1,-1)
        b = b + lr * (u - y)
    errors.append(int(error))
    if error < error_max:
        break
    else:
        error = 0
        np.random.shuffle(data)
        print(data)

print(errors)

plt.plot(errors)
plt.show()







