import numpy as np

x = np.random.normal(3.0, 10.0, 3000).reshape(-1,1) # генерируем х
e = np.random.normal(0, 0.3, 3000).reshape(-1,1)  # генерируем ошибку

y_1 = x**2 + e
y_2 = np.sin(x/2) + e
y_3 = np.cos(2*x) + e
y_4 = x - 3 + e
y_5 = -x + e
y_6 = abs(x) + e
y_7 = (x**3)/4 + e

np.savetxt('train_dataset.csv', np.hstack((y_1, y_3, y_4, y_5, y_6, y_7, y_2))) # записываем в файл

x = np.random.normal(3.0, 10.0, 1000).reshape(-1,1)
e = np.random.normal(0, 0.3, 1000).reshape(-1,1)

y_1 = x**2 + e
y_2 = np.sin(x/2) + e
y_3 = np.cos(2*x) + e
y_4 = x - 3 + e
y_5 = -x + e
y_6 = abs(x) + e
y_7 = (x**3)/4 + e

np.savetxt('test_dataset.csv', np.hstack((y_1, y_3, y_4, y_5, y_6, y_7, y_2)))