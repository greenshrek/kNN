## Imports and Loading Data
import collections
import numpy as np
import operator

X_test = np.genfromtxt('data/classification/testData.csv', delimiter=',')
X_train = np.genfromtxt('data/classification/trainingData.csv', delimiter=',')

#X_train = np.genfromtxt('trainingset.csv', delimiter=',')
#X_test = np.genfromtxt('test.csv', delimiter=',')

#X_test = np.array([[4,7,1,9,3,2]])

def calculateDistances(training_data, test_instance):
    num_of_features = 10

    result = np.empty(shape=[0,0])
    ind = np.empty(shape=[0])

    #slice numpy array to leave the class columnn
    training_data = training_data[:,:-1]
    test_instance = test_instance[:-1]
    
    #reshaping the single query instance into 2d numpy so that it can be computed for eucledian distance
    test_instance = np.reshape(test_instance, (1, num_of_features))

    result = np.sqrt(-2 * np.dot(test_instance, training_data.T) + np.sum(training_data**2,    axis=1) + np.sum(test_instance**2, axis=1)[:, np.newaxis])
    
    ind = np.argsort(result, axis=1) # sorts along last axis (across)
    return result, ind


def predictkNNClass(training_data, test_instance, k):
    num_of_features = 10
    result = np.empty(shape=[0,0])
    distances = np.empty(shape=[0,0])
    ind = np.empty(shape=[0])    

    class_vector_training = training_data[:,num_of_features]
    class_vector_test = test_instance[:,num_of_features]

    accurate_predictions = 0
    knnResult = []
    #print(class_vector)
    test_data_size = test_instance.shape[0]
    for i in range(0, test_data_size):
        #print(test_instance[i])
        result = []
        temp_dict = {}
        result = calculateDistances(training_data, test_instance[i])
        distances = result[0]
        ind = result[1]
        
        #print("test vector: ", class_vector_test[i])
        #print("distance: ",distances)
        #print("indices: ", ind)
        #print("results>>")
        #print("taken from class: ", np.take(class_vector_training, ind))
        #print("selected from class: ", (np.take(class_vector_training, ind))[:,0:k])
        unique, counts = np.unique((np.take(class_vector_training, ind))[:,0:k], return_counts=True)
        temp_dict = dict(zip(unique, counts))
        #print(temp_dict)
        maxvotes = max(temp_dict, key=temp_dict.get)
        #print("max value: ",maxvotes)
        #print("selected distance: ", np.take(distances,ind))
        knnResult.append(maxvotes)
        if maxvotes == class_vector_test[i]:
            accurate_predictions +=1
        #print("count of each element: ", collections.Counter((np.take(class_vector, ind))[:,0:k]))
    print(knnResult)
    print("accuracy: ", ((accurate_predictions/test_data_size)*100))


predictkNNClass(X_train, X_test, 1)