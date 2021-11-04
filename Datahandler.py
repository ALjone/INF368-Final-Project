import numpy as np
import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split
seed_val = 0

class Get_IMDB_data():
    
    def __init__(self,path, subset_size_train = None,subset_size_test = None, random_state = None):
        self.path = path
        self.subset_size = [subset_size_train,subset_size_test]
        self.random_state = random_state
        
    def to_gan_bert(self,train,test, pct):
        labeled, unlabeled = train_test_split(train, test_size= 1-pct , random_state=self.random_state, stratify = train.label)
        
        labeled = list(zip(labeled.iloc[:,0],labeled.iloc[:,1]))
        
        unlabeled = list(zip(unlabeled.iloc[:,0],["blank"]*unlabeled.shape[0]))
        
        test = list(zip(test.iloc[:,0],test.iloc[:,1]))

        return labeled, unlabeled, test

    def create_datasets(self, label_pct = 0.2):
        train_data,test = self.get_data()
        labeled, unlabeled = train_test_split(train_data, test_size= 1-label_pct,
                                            random_state=self.random_state, stratify = train_data.label)
        unlabeled = unlabeled.copy()
        unlabeled.loc[:,"label"] = "blank"
        labeled.to_csv("data/train_labeled.csv",index=False)
        unlabeled.to_csv("data/train_unlabeled.csv",index=False)
        test.to_csv("data/test.csv",index=False)

        
    
    def get_data(self):
        
        
        path_neg_train = self.path+'/train/neg'
        path_pos_train = self.path+'/train/pos'
        
        train_data = pd.concat([self.__get_data(path_neg_train,"neg",self.subset_size[0]),
                                self.__get_data(path_pos_train,"pos",self.subset_size[0])], axis = 0)
        
        path_neg_test = self.path+'/test/neg'
        path_pos_test = self.path+'/test/pos'
        
        test_data = pd.concat([self.__get_data(path_neg_test,"neg",self.subset_size[1]),
                                self.__get_data(path_pos_test,"pos",self.subset_size[1])], axis = 0)
        
        return train_data,test_data
        
        
    
    def __get_data(self,path,label, subset_size = None):
        '''
        input:
            path: the path to the txt files
            label: the label for the txt files
        return:
            a pandas data frame with the txt and the corresponding label. 
        '''
        TAG_RE = re.compile(r'<[^>]+>')

        if self.random_state is not None:
            np.random.seed(self.random_state) 
            
        files = os.listdir(path)
        
        if subset_size is not None:
            if subset_size> len(files):
                raise Exception("Subset_size must be be smaller or equal to the number of text file in the directory")
            files = np.random.choice(files, subset_size//2, replace = False)

        data = pd.DataFrame(None, columns = ["text", "label"])

        for file in files:
            row = {}
            with open(path+"/"+file,encoding='utf8') as f:
                row["text"] = TAG_RE.sub('', f.read())
                row["label"] = label
                data = data.append(row, ignore_index=True)

        return data


get_IMDB_data = Get_IMDB_data("data/aclImdb", 6000,500)
get_IMDB_data.create_datasets(0.02)