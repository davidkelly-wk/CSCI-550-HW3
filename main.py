# -*- coding: utf-8 -*-
import load_dataset as ld
import knn
import metrics
import numpy as np
import math
import pandas as pd
from sklearn.model_selection import KFold
from decision_tree import Decision_Tree

class Main:
        def __init__(self):
                #define variable to use inside class which may need tuning
                self.splitlength = 0.75
                self.knn_k_values = [3,5,7]
                self.num_folds = 5
                
                self.alldatasets = ld.LoadDataset().load_data()          #load all datasets 
                #define dataframe to store all the results
                self.allresults = pd.DataFrame(columns=['dataset', 'k', 'fold_number', 'method',
                                                        'accuracy', 'precision', 'recall', 'F1-score'])
                
        def main(self):
                for dataset in self.alldatasets:         #for each dataset call each algorithm
                        print('current dataset ::: {0} \n'.format(dataset))
                        data = self.alldatasets.get(dataset)
                        sqrt_k = int(math.sqrt(len(data)))      #take square root of n as value of k
                        if sqrt_k%2 == 0:
                                sqrt_k -= 1                     #make sqrt_k odd number
                        self.knn_k_values[2] = sqrt_k           #3 values for K including sqrt(n)
                        for k in self.knn_k_values:
                                #k-fold cross validation
                                kf = KFold(n_splits = self.num_folds)
                                fold_number = 1
                                for train_index, test_index in kf.split(data):
                                        #indices to use for test and train
                                        trainset = np.take(data, axis=0, indices=train_index)
                                        testset = np.take(data, axis=0, indices=test_index)
                                        #call knn  
                                        predicted, labels = self.knn(trainset, testset, k)
                                        self.performance_measure(predicted, labels, dataset, k, fold_number, 'KNN')
                                        fold_number += 1

                return self.allresults

        def main_DT(self):
                for dataset in self.alldatasets:         #for each dataset call each algorithm
                        print('current dataset ::: {0} \n'.format(dataset))
                        data = self.alldatasets.get(dataset)
                        #k-fold cross validation
                        kf = KFold(n_splits = self.num_folds)
                        fold_number = 1
                        for train_index, test_index in kf.split(data):
                                #indices to use for test and train
                                trainset = np.take(data, axis=0, indices=train_index)
                                testset = np.take(data, axis=0, indices=test_index)
                                print(trainset.columns)
                                print(len(trainset))
                                #call DT
                                predicted, labels = self.DT(trainset, testset)
                                self.performance_measure(predicted, labels, dataset, 'N/A', fold_number, 'DTree')
                                fold_number += 1

                return self.allresults

        def DT(self, trainset, testset):
                dt = Decision_Tree(int(1), 0.95)
                dt.root = dt.create_dt(trainset, max(trainset[trainset.columns[-1]])+1)
                predicted = dt.classify(trainset)#testset)
                return predicted, trainset[trainset.columns[-1]]#  testset[testset.columns[-1]]


        
        def knn(self, trainset, testset, k):
                predicted = knn.Knn().fit(trainset.values, testset, k)
                return predicted, testset.iloc[:, -1]   #return predicted and actual labels
        
        def performance_measure(self, predicted, labels, dataset, k, fold_number, method):
                acc, prec, recall, f1_score = metrics.confusion_matrix(labels.values, predicted)
                self.update_result(dataset, k, fold_number, method, acc, prec, recall, f1_score)
        
        def update_result(self, dataset, k, fold_number, method, acc, prec, recall, f1_score):
                self.allresults = self.allresults.append({'dataset': dataset,
                                                'k': k, 'fold_number': fold_number, 'method': method, 'accuracy': acc, 'precision': prec,
                                                'recall': recall, 'F1-score': f1_score}, ignore_index=True)
        
        
# results = Main().main()
# results.to_csv('results.csv')
# print(results)

results = Main().main_DT()
results.to_csv('results.csv')
print(results)