"""
Author: Pranav Srivastava
Detail: Basic KNN
"""
import numpy as np
import sys

class KNNClassification:

    def __init__(self, k, trainingdataset, testdataset):
        self.k = k
        self.num_of_features = 10
        self.trainingdataset = trainingdataset
        self.testdataset = testdataset
        self.accurate_predictions = 0
        self.accuracy = 0


    def calculateDistances(self, training_data, test_instance):

        result = np.empty(shape=[0,0])
        ind = np.empty(shape=[0])

        #slice numpy array to leave the class columnn
        training_data = training_data[:,:-1]
        test_instance = test_instance[:-1]
        
        #reshaping the single query instance into 2d numpy so that it can be computed for eucledian distance
        test_instance = np.reshape(test_instance, (1, self.num_of_features))

        result = np.sqrt(-2 * np.dot(test_instance, training_data.T) + np.sum(training_data**2,    axis=1) + np.sum(test_instance**2, axis=1)[:, np.newaxis])
        
        ind = np.argsort(result, axis=1) # sorts along last axis (across)
        return result, ind


    def predictkNNClass(self):
        
        training_data = self.trainingdataset
        test_instance = self.testdataset

        result = np.empty(shape=[0,0])
        distances = np.empty(shape=[0,0])
        ind = np.empty(shape=[0])    

        class_vector_training = training_data[:,self.num_of_features]
        class_vector_test = test_instance[:,self.num_of_features]

        knnResult = []

        test_data_size = test_instance.shape[0]
        for i in range(0, test_data_size):

            result = []
            temp_dict = {}

            # send a single query instance to the get the predicrted class
            result = self.calculateDistances(training_data, test_instance[i])
            distances = result[0]
            ind = result[1]
            
            #key value pair for the class and corresponding count is prepared below
            unique, counts = np.unique((np.take(class_vector_training, ind))[:,0:self.k], return_counts=True)
            temp_dict = dict(zip(unique, counts))

            #class with maximum votes is saved as the classification
            maxvotes = max(temp_dict, key=temp_dict.get)
            print(maxvotes)
            knnResult.append(maxvotes)

            if maxvotes == class_vector_test[i]:
                self.accurate_predictions +=1

        print(knnResult)
        #accuracy percentage is calculated
        self.accuracy = ((self.accurate_predictions/test_data_size)*100)

#load datasets into numpy array
testdataset = np.genfromtxt('data/classification/testData.csv', delimiter=',')
trainingdataset = np.genfromtxt('data/classification/trainingData.csv', delimiter=',')

k = int(sys.argv[1])
filename = sys.argv[2]
remark = sys.argv[3]

#initializing the class
knnc = KNNClassification(k, trainingdataset, testdataset)

knnc.predictkNNClass()

#create data for the results file
results = []
results.append(knnc.k)
results.append(knnc.accuracy)
results.append(filename)
results.append(remark)

#append results to the file
with open('test_results.txt', 'a') as f:
    f.write((str("%s\n" % results)))