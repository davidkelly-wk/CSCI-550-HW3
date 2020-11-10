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
                        for i, j in ([1, 0.95], [15, 0.95], [1, 0.75], [15, 0.75]):
                                for train_index, test_index in kf.split(data):
                                        #indices to use for test and train
                                        trainset = np.take(data, axis=0, indices=train_index)
                                        testset = np.take(data, axis=0, indices=test_index)
                                        print("Attributes")
                                        print(trainset.columns)
                                        print(str.format('training set length : {0}', len(trainset)))
                                        #call DT
                                        predicted, labels = self.DT(trainset, testset, i, j)
                                        self.performance_measure_DT(predicted, labels, dataset, i, j, fold_number, 'DTree')
                                        fold_number += 1

                return self.allresults

        def DT(self, trainset, testset, leaf_size, purity):
                dt = Decision_Tree(int(leaf_size), purity)
                dt.root = dt.create_dt(trainset, max(trainset[trainset.columns[-1]])+1)
                predicted = dt.classify(testset)#testset)
                # print(predicted)
                return predicted, testset[testset.columns[-1]]#  testset[testset.columns[-1]]


        
        def knn(self, trainset, testset, k):
                predicted = knn.Knn().fit(trainset.values, testset, k)
                return predicted, testset.iloc[:, -1]   #return predicted and actual labels
        
        def performance_measure(self, predicted, labels, dataset, k, fold_number, method):
                acc, prec, recall, f1_score = metrics.confusion_matrix(labels.values, predicted)
                self.update_result(dataset, k, fold_number, method, acc, prec, recall, f1_score)

        def performance_measure_DT(self, predicted, labels, dataset, leaf_size, purity, fold_number, method):
                acc, prec, recall, f1_score = metrics.confusion_matrix(labels.values, predicted)
                self.update_result_DT(dataset, leaf_size, purity, fold_number, method, acc, prec, recall, f1_score)
        
        def update_result(self, dataset, k, fold_number, method, acc, prec, recall, f1_score):
                self.allresults = self.allresults.append({'dataset': dataset,
                                                'k': k, 'fold_number': fold_number, 'method': method, 'accuracy': acc, 'precision': prec,
                                                'recall': recall, 'F1-score': f1_score}, ignore_index=True)
        def update_result_DT(self, dataset, leaf_size, purity, fold_number, method, acc, prec, recall, f1_score):
                self.allresults = self.allresults.append({'dataset': dataset,
                                                'leaf_size': leaf_size, 'purity': purity, 'fold_number': fold_number, 'method': method, 'accuracy': acc, 'precision': prec,
                                                'recall': recall, 'F1-score': f1_score}, ignore_index=True)
        
        
knn_results = Main().main()
knn_results.to_csv('knn_results.csv')
print(knn_results)

dtree_results = Main().main_DT()
dtree_results.to_csv('dtree_results.csv')
print(dtree_results)