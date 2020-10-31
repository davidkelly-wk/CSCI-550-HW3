# -*- coding: utf-8 -*-
import numpy as np
     
def confusion_matrix(y_test, y_pred):
        #prevent division by 0
        TP = 0.0000001
        TN = 0.0000001
        FP = 0.0000001
        FN = 0.0000001
        #find unique class label for test and pred
        test_label = np.unique(np.array(y_test))
        pred_label = np.unique(np.array(y_pred))
        
        #work with max labels from test and pred
        if len(test_label) >= len(pred_label):
            labels = test_label
        else:
            labels = pred_label
        #find the confusion matrix values   
        for x in labels:
            for y in range(len(y_test)):
                if y_test[y] == y_pred[y] == x:                 
                    TP += 1
                if y_pred[y]== x and y_test[y] != y_pred[y]:
                    FP += 1
                if y_test[y] == y_pred[y] != x:
                    TN += 1
                if y_pred[y] != x and y_test[y] != y_pred[y]:
                    FN += 1
                    
        accuracy = ((TP + TN)/ (TP + TN + FP + FN))
        precision = TP/(FP + TP)
        recall = TP/(FN + TP)
        return accuracy, precision, recall