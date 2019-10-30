
import numpy as np

labeled_classes = np.array([[1, 1, 1, 2, 2, 2 ,3, 3, 3]])
distances = np.array([[10 ,9, 5, 5, 6, 5, 1, 2, 2]])


mapping = {}


for i in range(0, distances.size):
    n = 10
    index = labeled_classes[0][i]
    if index in mapping:
        mapping[labeled_classes[0][i]] = mapping[labeled_classes[0][i]] + 1/distances[0][i]
    else:
        mapping[index] = 1/(distances[0][i]**n)

print(mapping)
print(max(mapping, key=mapping.get))

print("distances: ",distances)
print("labeled classes>>", labeled_classes)
index_maxvotes = np.argmax(np.reciprocal(distances)/(np.sum(np.reciprocal(distances))))

print(np.argmax(np.reciprocal(distances)/(np.sum(np.reciprocal(distances)))))

