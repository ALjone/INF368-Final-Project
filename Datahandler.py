import numpy as np
import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split

class Get_IMDB_data():
    
    def __init__(self,path,  random_state = None):
      self.path = path
      self.random_state = random_state
        
    def to_gan_bert(self,train,test, pct):
      labeled, unlabeled = train_test_split(train, test_size= 1-pct , random_state=self.random_state, stratify = train.label)
      
      labeled = list(zip(labeled.iloc[:,0],labeled.iloc[:,1]))
      
      unlabeled = list(zip(unlabeled.iloc[:,0],["blank"]*unlabeled.shape[0]))
      
      test = list(zip(test.iloc[:,0],test.iloc[:,1]))

      return labeled, unlabeled, test

    def create_datasets(self, labeled_size = 100,unlabeled_size = 5000,test_size = 500):

      labeled,unlabeled,test = self.get_data(labeled_size,unlabeled_size,test_size)
      
      unlabeled = unlabeled.copy()
      unlabeled.loc[:,"label"] = "blank"
      for k,v in labeled.items():
        v.to_csv(f"data/train_labeled_{k}.csv",index=False)
      unlabeled.to_csv("data/train_unlabeled.csv",index=False)
      test.to_csv("data/test.csv",index=False)

        
    
    def get_data(self,labeled_size,unlabeled_size,test_size):
        
        if isinstance(labeled_size,int):
          labeled_sizes = [labeled_size]
        elif isinstance(labeled_size,list):
          labeled_sizes = labeled_size
        
        for labeled_size in labeled_sizes:
          if not isinstance(labeled_size,int):
            raise Exception("labeled_size must be a int or a list of ints")
        
        labeled ={}
        labeled_sizes.sort(reverse=True)
        labeled_size = labeled_sizes[0]
        path_neg_train = self.path+'/train/neg'
        path_pos_train = self.path+'/train/pos'
        train_size = labeled_size + unlabeled_size//2

        neg = self.__get_data(path_neg_train,"neg",train_size)
        pos = self.__get_data(path_pos_train,"pos",train_size)

        labeled[labeled_size] = pd.concat([neg[:labeled_size],pos[:labeled_size]], axis = 0)
        unlabeled = pd.concat([neg[labeled_size:],pos[labeled_size:]], axis = 0)

   

        for labeled_size in labeled_sizes[1:]:
          labeled[labeled_size] = pd.concat([neg[:labeled_size],pos[:labeled_size]], axis = 0)
        
        path_neg_test = self.path+'/test/neg'
        path_pos_test = self.path+'/test/pos'
        
        test_data = pd.concat([self.__get_data(path_neg_test,"neg",test_size//2),
                                self.__get_data(path_pos_test,"pos",test_size//2)], axis = 0)
        
        return labeled,unlabeled,test_data
        
        
    
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
            files = np.random.choice(files, subset_size, replace = False)

        data = pd.DataFrame(None, columns = ["text", "label"])

        for file in files:
            row = {}
            with open(path+"/"+file,encoding='utf8') as f:
                row["text"] = TAG_RE.sub('', f.read())
                row["label"] = label
                data = data.append(row, ignore_index=True)

        return data


class Get_medical_data():
    def __init__(self,from_path,to_path,random_state):
        self.path = from_path
        self.to_path = to_path
        self.random_state = random_state
    
    def create_datasets(self,labeled_size=[5,10,25,50],unlabeled_size= 500,test_size=500):
        labeled = pd.read_table(self.path+"/train.dat",sep = "\t",header =None,names =["label", "text"])
        labeled = labeled[[ "text","label"]]
        unlabeled = pd.read_table(self.path+"/test.dat",sep = "\t",header =None,names =["text"])
        unlabeled["label"] = "blank"

        unlabeled,_ = train_test_split(unlabeled,train_size = unlabeled_size/unlabeled.shape[0], random_state= 0)
        unlabeled =unlabeled.copy()
        test = pd.DataFrame(columns=["text","label"])
        for i in labeled_size:
            train = pd.DataFrame(columns=["text","label"])  
            for label in labeled.label.unique():
                train = pd.concat([train, labeled[labeled.label==label].iloc[-i:]], ignore_index=True)
                train = train.sample(frac=1, random_state=0).reset_index(drop=True)
                train.to_csv(f"{self.to_path}/train_labeled_{i}.csv",index=False)

        for label in labeled.label.unique():
            test = pd.concat([test, labeled[labeled.label==label].iloc[:100]], ignore_index=True)
            test = test.sample(frac=1, random_state=0).reset_index(drop=True)
        test.to_csv(self.to_path+"/test.csv",index=False)
        unlabeled.to_csv(self.to_path+"/train_unlabeled_.csv",index=False)
        
    
        

    

if __name__ == "__main__":
    
    if not os.path.exists('data'):
      os.mkdir("./data")
      url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
      import tensorflow as tf
      dataset = tf.keras.utils.get_file('aclImdb_v1.tar.gz', url,
                                          untar=True, cache_dir='.',
                                          cache_subdir='')

      dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
      get_IMDB_data = Get_IMDB_data("/content/aclImdb",random_state=0)
      get_IMDB_data.create_datasets(labeled_size=[5,10,25,50],unlabeled_size= 5000,test_size=500)
