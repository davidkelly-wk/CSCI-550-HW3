import numpy as np
from statistics import mean, mode, StatisticsError 

# function parameters: trainng dataset, test point, number of neighbors k
class Knn:
    @staticmethod
    def knn(trainSet, testPoint, k):
        # store indices and distances in 2D array
        distances = np.zeros(shape=(len(trainSet), 2))

        # loop through training set to find distances between test point and each training set point
        for i in range(len(trainSet)):    
            # update to use our own distance metric
            curDist = np.linalg.norm(testPoint[:-1]-trainSet[i,:-1])
            distances[i][0] = i
            distances[i][1] = curDist
   
        # sort by distance and subset to k neighbors' response values
        sortedDist = sorted(distances, key=lambda x: x[1])
        neighbors = np.zeros(k)
        for i in range(k):
            neighbors[i] = trainSet[int(sortedDist[i][0])][-1]
        
        # return predicted class or regression value
        return Knn.predict(neighbors, trainSet, testPoint, k)

    # predict response variable from neighbors
    @staticmethod
    def predict(neighbors, trainSet, testPoint, k):
        # choose most popular class for classification
        # in case of tie, repeatedly run knn with k-1 until most popular class is found
        try:
            return int(mode(neighbors))
        except StatisticsError:
            return Knn.knn(trainSet, testPoint, k-1)

    def fit(self, trainset, testset, k):
        predicted = []
        for index, x in testset.iterrows():
                predicted.append(Knn.knn(trainset, x, k))
        return predicted
