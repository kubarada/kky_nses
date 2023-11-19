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

        W = W + np.dot(lr * ((u - y) * dy * V).T, np.ones((5,1))) * dz @x.T
        b = b + np.dot(lr * ((u - y) * dy * V).T, np.ones((5, 1))) * dz
        V = V + lr * ((u - y) * dy)@z.T
        d = d + lr * ((u - y) * dy)

        error = error + np.dot(1 / 2, np.dot((u - y).reshape(1, -1), (u - y)))
    errors.append(int(error))
    if error < error_max:
        break
    else:
        error = 0
        np.random.shuffle(data)

print(errors)

plt.plot(range(len(errors)), errors)
plt.grid()
plt.title('Vývoj chyby')
plt.xlabel('Epocha')
plt.ylabel('Hodnota chyby')
plt.show()

x_max = np.max(data[:, 0])
x_min = np.min(data[:, 0])
y_max = np.max(data[:, 1])
y_min = np.min(data[:, 1])

plt.scatter(data[:, 0], data[:,1])
plt.title('Distribuce dat - train')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
x = np.linspace(x_min-1, x_max+1)
for i in range(len(W)):
    y = (W[i, 0]*x + b[i])/(-W[i, 1])
    plt.plot(x, y)
plt.xlim([x_min-1, x_max+1])
plt.ylim([y_min-1, y_max+1])
plt.show()



