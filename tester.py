
import numpy as np

x = np.array([[5, 1 , 3 ,2]])
print(x)
ind = np.argsort(x, axis = 1)
print(ind)
print(np.take_along_axis(x, ind, axis=1))

k =2
idx = np.argpartition(x, k)
return np.take(y, idx[:k])

#print(x[x[:,1].argsort()])