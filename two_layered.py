import numpy as np
import matplotlib.pyplot as plt
import methods

dataset = np.loadtxt('data/tren_data2___16.txt')
np.random.shuffle(dataset)
data = dataset[0:399, :]
data1 = data[data[:, 2] == 1]
data2 = data[data[:, 2] == 2]
data3 = data[data[:, 2] == 3]
data4 = data[data[:, 2] == 4]
data5 = data[data[:, 2] == 5]

test = dataset[400:498, :]
test1 = test[test[:, 2] == 1]
test2 = test[test[:, 2] == 2]
test3 = test[test[:, 2] == 3]
test4 = test[test[:, 2] == 4]
test5 = test[test[:, 2] == 5]

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
epochs = 1500
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

plt.scatter(data1[:, 0], data1[:,1])
plt.scatter(data2[:, 0], data2[:,1])
plt.scatter(data3[:, 0], data3[:,1])
plt.scatter(data4[:, 0], data4[:,1])
plt.scatter(data5[:, 0], data5[:,1])
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

# inference
errors = []

for k in range(len(test)):
    x = test[k, 0:2].reshape(-1,1)
    xi = np.dot(W, x) + b
    z = methods.sigmoid(xi)
    zi = np.dot(V, z) + d
    y = methods.sigmoid(zi)
    u = methods.get_wanted(test, k, U).reshape(-1,1)
    error = error + np.dot(1 / 2, np.dot((u - y).reshape(1, -1), (u - y)))

x_max = np.max(test[:, 0])
x_min = np.min(test[:, 0])
y_max = np.max(test[:, 1])
y_min = np.min(test[:, 1])

plt.scatter(test1[:, 0], test1[:,1])
plt.scatter(test2[:, 0], test2[:,1])
plt.scatter(test3[:, 0], test3[:,1])
plt.scatter(test4[:, 0], test4[:,1])
plt.scatter(test5[:, 0], test5[:,1])
plt.title('Distribuce dat - test')
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

grid_size = 18

# Generate the grid points
x_values = np.arange(-grid_size, grid_size + 1, 0.5)
y_values = np.arange(-grid_size, grid_size + 1, 0.5)

# Create a meshgrid
xx, yy = np.meshgrid(x_values, y_values)

# Stack the coordinates and class (always 0)
grid = np.dstack((xx, yy, np.zeros(xx.shape, dtype=int))).reshape(-1, 3)

for k in range(len(grid)):
    x = grid[k, 0:2].reshape(-1,1)
    xi = np.dot(W, x) + b
    z = methods.sigmoid(xi)
    zi = np.dot(V, z) + d
    y = methods.sigmoid(zi)
    grid[k, 2] = np.argmax(y) + 1

grid1 = grid[grid[:, 2] == 1]
grid2 = grid[grid[:, 2] == 2]
grid3 = grid[grid[:, 2] == 3]
grid4 = grid[grid[:, 2] == 4]
grid5 = grid[grid[:, 2] == 5]

plt.scatter(grid1[:, 0], grid1[:,1], marker='o')
plt.scatter(grid2[:, 0], grid2[:,1], marker='o')
plt.scatter(grid3[:, 0], grid3[:,1], marker='o')
plt.scatter(grid4[:, 0], grid4[:,1], marker='o')
plt.scatter(grid5[:, 0], grid5[:,1], marker='o')
plt.title('Distribuce dat - test')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([x_min-1, x_max+1])
plt.ylim([y_min-1, y_max+1])
plt.show()
