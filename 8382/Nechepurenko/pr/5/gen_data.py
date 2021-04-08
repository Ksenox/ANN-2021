from typing import List, Callable

import numpy as np

np.random.seed(42)

fns = [
    lambda x: -np.power(x, 3),
    lambda x: np.log(np.abs(x)),
    lambda x: np.sin(3 * x),
    lambda x: np.exp(x),
    lambda x: x + 4,
    lambda x: -x + np.sqrt(np.abs(x)),
    lambda x: x
]


def generate(
        functions: List[Callable], target_index: int = 0, n_samples: int = 1000,
        x_normal_loc: float = 0., x_normal_scale: float = 1.,
        e_normal_loc: float = 0., e_normal_scale: float = 0.1
) -> np.array:
    X = np.random.normal(loc=x_normal_loc, scale=x_normal_scale, size=n_samples)
    e = np.random.normal(loc=e_normal_loc, scale=e_normal_scale, size=n_samples)
    features = np.array(
        [np.array(function(X) + e) for index, function in enumerate(functions) if index != target_index]
    ).T
    target = np.array([functions[target_index](X) + e]).T
    return np.hstack((features, target))


target_index = 2  # third function
generate_for_task = lambda num: generate(fns, target_index, num, -5., 10., 0., 0.3)
np.savetxt("samples/train.csv", generate_for_task(1000), delimiter=",", fmt='%1.3f')
np.savetxt("samples/test.csv", generate_for_task(200), delimiter=",", fmt='%1.3f')
