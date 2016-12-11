#!usr/local/python3
'''
softmax:
        f(xi) = e^xi / sum(e^xj)
'''
import numpy as np
import matplotlib.pyplot as plt

scores = np.array([3.0, 1.0, 0.2])

# two_dim_scores = np.array([[1, 2, 3, 6],
#                             [2, 4, 5, 6],
#                             [3, 8, 7, 6]])

scores /= 10

def softmax(x):
    '''
        Compute softmax function for x
        Return: f(xi) = e^xi / sum(e^xj)
    '''
    return np.exp(x) / np.sum(np.exp(scores), axis = 0)

print('-One dimension:')
print(softmax(scores))


# plot softmax curves
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth = 2)
plt.show()



