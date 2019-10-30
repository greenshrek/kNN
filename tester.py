
import numpy as np

test = np.genfromtxt('test.csv', delimiter=',')
#training = np.genfromtxt('trainingset.csv', delimiter=',')

#test = np.array([[5.5,787]])

min = test.min(axis=0)
max = test.max(axis=0)

print(min, max)

n = test-min
d = max-min
print(np.nan_to_num(np.true_divide(n,d)))