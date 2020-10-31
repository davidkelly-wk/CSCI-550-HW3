# -*- coding: utf-8 -*-
import pandas as pd

class LoadDataset:
        def __init__(self):
                self.directory = 'datasets/'
                #self.datafiles = ['car.data','segmentation.data']
                self.datafiles = ['segmentation.data']
                self.alldatasets = {}
                
        def load_data(self):
                for files in self.datafiles:       
                        #read each data file
                        data = pd.read_csv(self.directory + files)
                        #give filename without extension as dict key for each dataset
                        key = files.split('.')[0]
                        self.alldatasets[key] = self.PreprocessingData(key, data)
                return self.alldatasets
                
        def PreprocessingData(self, key, data):
                if key == 'segmentation':
                        data = data.replace({'CLASS': {'BRICKFACE': 0, 'SKY': 1, 'FOLIAGE': 2, 'CEMENT': 3,
                                                       'WINDOW': 4, 'GRASS': 5, 'PATH': 6 }})
                        class_d = data['CLASS']
                        data = data.drop(['CLASS'], axis= 1)
                        data['CLASS'] = class_d
                elif key == 'car':
                        data = data.replace({'low': 0, 'med': 1, 'high': 2, 'vhigh':3, '5more': 5,
                                             'more': 5, 'small': 0, 'big': 2, 'unacc': 0, 'acc': 1, 
                                             'good': 2, 'vgood': 3})
                        data[['doors', 'persons']] = data[['doors', 'persons']].astype(int)
                return data
                