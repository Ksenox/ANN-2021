import numpy as np

# generate a data
mu_x = 0
mu_e = 0
sigma_x = 10
sigma_e = 0.3
size_train = 1000
size_test = 500

x = np.random.normal(mu_x, sigma_x, size_train)
x = np.reshape(x, (size_train, 1))
e = np.random.normal(mu_e, sigma_e, size_train)
e = np.reshape(e, (size_train, 1))
train_data = np.asarray([
    np.cos(x) + e,              # 1
    -x + e,                     # 2
    (np.sin(x)) * x + e,        # 3
    np.sqrt(np.abs(x)) + e,     # 4
    x ** 2 + e,                 # 5
    x - (x ** 2) / 5 + e])      # 7
train_data = np.reshape(train_data, (size_train, 6))
train_labels = np.asarray([-np.abs(x) + 4])
train_labels = np.reshape(train_labels, (size_train, 1))

np.savetxt('my_dataset_train.csv', np.hstack((train_data, train_labels)), delimiter='\t')

x = np.random.normal(mu_x, sigma_x, size_test)
x = np.reshape(x, (size_test, 1))
e = np.random.normal(mu_e, sigma_e, size_test)
e = np.reshape(e, (size_test, 1))
test_data = np.asarray([
    np.cos(x) + e,              # 1
    -x + e,                     # 2
    (np.sin(x)) * x + e,        # 3
    np.sqrt(np.abs(x)) + e,     # 4
    x ** 2 + e,                 # 5
    x - (x ** 2) / 5 + e])      # 7
test_data = np.reshape(test_data, (size_test, 6))
test_labels = np.asarray([-np.abs(x) + 4])
test_labels = np.reshape(test_labels, (size_test, 1))

np.savetxt('my_dataset_test.csv', np.hstack((test_data, test_labels)), delimiter='\t')
# generate data ends
