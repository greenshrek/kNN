import numpy as np
import sys

class DistanceWeightedKNNClassification:

    def __init__(self, k, trainingdataset, testdataset):
        self.k = k
        self.num_of_features = 10
        self.trainingdataset = trainingdataset
        self.testdataset = testdataset
        self.accurate_predictions = 0
        self.accuracy = 0


    def calculateDistances(self,training_data, test_instance):

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

    def votesDW(self, distances, labeled_classes):
        mapping = {}
        n = 1
        for i in range(0, distances.size):
            index = labeled_classes[0][i]
            print("index: ", index)
            if index in mapping:
                mapping[labeled_classes[0][i]] = mapping[labeled_classes[0][i]] + 1/distances[0][i]
            else:
                mapping[index] = 1/(distances[0][i]**n)
            print(distances[0][i])
        return max(mapping, key=mapping.get)

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

            result = self.calculateDistances(training_data, test_instance[i])
            
            distances = result[0]
            ind = result[1]
            
            k_near_distances = (np.take(distances, ind))[:,0:self.k]
            labeled_classes = (np.take(class_vector_training, ind))[:,0:self.k]

            voteDW = self.votesDW(k_near_distances, labeled_classes)
            print(k_near_distances,labeled_classes)
            print("vote>>",voteDW)

            knnResult.append(voteDW)

            if voteDW == class_vector_test[i]:
                self.accurate_predictions +=1

        print("predictions: ",knnResult)
        print("accuracy: ", ((self.accurate_predictions/test_data_size)*100))
        self.accuracy = ((self.accurate_predictions/test_data_size)*100)


testdataset = np.genfromtxt('data/classification/testData.csv', delimiter=',')
trainingdataset = np.genfromtxt('data/classification/trainingData.csv', delimiter=',')

#trainingdataset = np.genfromtxt('trainingset.csv', delimiter=',')
#testdataset = np.genfromtxt('test.csv', delimiter=',')

k = int(sys.argv[1])
filename = sys.argv[2]
remark = sys.argv[3]


dwknnc = DistanceWeightedKNNClassification(k, trainingdataset, testdataset)

dwknnc.predictkNNClass()

results = []
results.append(dwknnc.k)
results.append(dwknnc.accuracy)
results.append(filename)
results.append(remark)

with open('test_results.txt', 'a') as f:
    f.write((str("%s\n" % results)))