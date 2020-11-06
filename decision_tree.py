import pandas as pd
import numpy as np
import copy

class Node:
    def __init__(self):
        self.split_value = None
        self.split_attribute = None
        self.y_branch = None
        self.n_branch = None
        self.label = None

class Decision_Tree:
    def __init__(self, leaf_size, purity):
        # self.dataframe = dataframe
        self.leaf_size = leaf_size
        self.purity = purity
        self.root = None


    def create_dt(self, data, n_classes, data_prev = pd.DataFrame(), root=False):
        # print(data.head())
        node = Node()
        # if len(data) == 0:
        #     return None
        majority_class, purity = self.get_purity(data)
        if len(data) < self.leaf_size or purity > self.purity:
            node.label = majority_class
            return node

        if data.equals(data_prev):
            node.label = majority_class
            return node

        # find attribute to split on
        node.split_attribute, node.split_value = self.find_split(data, n_classes)

        # split data
        data_y = data[data[node.split_attribute] < node.split_value]
        data_n = data[data[node.split_attribute] >= node.split_value]


        # call algorithm on each branch
        if len(data_y) > 0:
            node.y_branch = self.create_dt(data_y, n_classes, data)
        if len(data_n) > 0:
            node.n_branch = self.create_dt(data_n, n_classes, data)

        return node

    def get_purity(self, data):
        # value counts sorts from high to low
        class_total = data[data.columns[-1]].value_counts()
        majority_class = class_total.index[0]
        purity = class_total.iloc[0]/len(data)
        return majority_class, purity


    def find_split(self, data, n_classes):
        split_attribute = None
        split_value = 0
        score = -float('INF')
        for i in range(len(data.columns)-1):
            # print(data.columns[i])#, data.columns[i].type())
            if data[data.columns[i]].dtype != 'O':
                # call numeric attribute evaluator
                mid_point, score_temp = self.evaluate_attribute_numeric(data.columns[i], data, n_classes)
                # print(score_temp)
                if score_temp > score:
                    score = score_temp
                    split_attribute = data.columns[i]
                    split_value = mid_point

        return split_attribute, split_value


    def evaluate_attribute_numeric(self, attribute, data, n_classes):
        # track potential mid points
        mid_points = []
        # track class counts at the mid points
        class_counts = []

        # get classes and totals
        class_totals = data[data.columns[-1]].value_counts().sort_index()
        class_total = np.zeros(n_classes)
        # print(class_totals)
        for i in range(n_classes):
            if i in class_totals.index:
                class_total[i] = class_totals[i]
        # print(class_total)
        # print(class_total)

        # array of the class counts
        class_count = np.zeros(n_classes)

        data_copy = data.copy()
        # sort data by attribute
        data_copy = data_copy.sort_values(by=[attribute])
        # initialize counter variables
        prev_class = data_copy[data_copy.columns[-1]].iloc[0]
        prev_value = data_copy[attribute].iloc[0]

        for index, row in data_copy.iterrows():
            # print(row[-1])
            if row[-1] != prev_class:
                mid_points.append((row[attribute]+prev_value)/2)
                # print(class_count)

                class_counts.append(copy.deepcopy(class_count))

            class_count[int(row[-1])] += 1
            prev_value = row[attribute]
            prev_class = row[-1]

        mid_point = None
        score = -float('INF')
        for i in range(len(mid_points)):
            P_t = np.zeros(n_classes)
            P_y = np.zeros(n_classes)
            P_n = np.zeros(n_classes)
            for j in range(n_classes):
                P_t[j] = class_total[j]/sum(class_total)
                if P_t[j] == 0:
                    P_t[j] += 0.001
                P_y[j] = class_counts[i][j]/sum(class_counts[i])
                if P_y[j] == 0:
                    P_y[j] += 0.001
                P_n[j] = (class_total[j]-class_counts[i][j])/(sum(class_total)-sum(class_counts[i]))
                if P_n[j] == 0:
                    P_n[j] += 0.001
            # get gain
            # print(P_t)
            # print(P_y)
            # print(P_n, '\n')
            score_temp = self.get_score(P_t, P_y, P_n, class_total, class_counts[i], n_classes)
            # print('temp score', score_temp)
            if score_temp > score:
                score = score_temp
                mid_point = mid_points[i]

        return mid_point, score

    def get_score(self, P_t, P_y, P_n, class_total, class_counts, n_classes):
        H_t = 0
        H_y = 0
        H_n = 0
        for i in range(n_classes):
            H_t -= P_t[i] * np.log2(P_t[i])
            H_y -= P_y[i] * np.log2(P_y[i])
            # print(P_n[i])
            H_n -= P_n[i] * np.log2(P_n[i])

        # print(H_n)
        score = H_t - sum(class_counts)/sum(class_total)*H_y - (sum(class_total)-sum(class_counts))/sum(class_total)*H_n
        # print(score)

        return score

    def classify(self, test_data):
        predicted = pd.Series(np.zeros(len(test_data)))
        for i in range(len(test_data)):
            predicted[i] = self.classify_point(test_data.iloc[i], self.root)


        return predicted

    def classify_point(self, data_point, node):
        if node.label != None:
            return node.label
        if data_point[node.split_attribute] <= node.split_value and node.y_branch != None:
            return self.classify_point(data_point, node.y_branch)
        elif data_point[node.split_attribute] > node.split_value and node.n_branch != None:
            return self.classify_point(data_point, node.n_branch)
