import os


import numpy as np
import pandas as pd 

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer

import matplotlib.pyplot as plt



tf.get_logger().setLevel('ERROR')


class Bert:
   
  def __init__(self, num_classes,bert_model_name = 'bert_en_cased_L-12_H-768_A-12', random_state = None):
    map_name_to_handle = {
    'bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
    'bert_en_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3',
    'bert_multi_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3',
    'small_bert/bert_en_uncased_L-2_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-2_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-2_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-2_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-4_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-4_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-4_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-4_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-6_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-6_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-6_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-6_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-8_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-8_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-8_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-8_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-10_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-10_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-10_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-10_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-12_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-12_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-12_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1',
    'albert_en_base':
        'https://tfhub.dev/tensorflow/albert_en_base/2',
    'electra_small':
        'https://tfhub.dev/google/electra_small/2',
    'electra_base':
        'https://tfhub.dev/google/electra_base/2',
    'experts_pubmed':
        'https://tfhub.dev/google/experts/bert/pubmed/2',
    'experts_wiki_books':
        'https://tfhub.dev/google/experts/bert/wiki_books/2',
    'talking-heads_base':
        'https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1'}

    map_model_to_preprocess = {
      'bert_en_uncased_L-12_H-768_A-12':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'bert_en_cased_L-12_H-768_A-12':
          'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3',
      'small_bert/bert_en_uncased_L-2_H-128_A-2':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-2_H-256_A-4':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-2_H-512_A-8':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-2_H-768_A-12':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-4_H-128_A-2':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-4_H-256_A-4':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-4_H-512_A-8':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-4_H-768_A-12':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-6_H-128_A-2':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-6_H-256_A-4':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-6_H-512_A-8':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-6_H-768_A-12':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-8_H-128_A-2':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-8_H-256_A-4':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-8_H-512_A-8':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-8_H-768_A-12':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-10_H-128_A-2':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-10_H-256_A-4':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-10_H-512_A-8':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-10_H-768_A-12':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-12_H-128_A-2':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-12_H-256_A-4':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-12_H-512_A-8':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'small_bert/bert_en_uncased_L-12_H-768_A-12':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'bert_multi_cased_L-12_H-768_A-12':
          'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3',
      'albert_en_base':
          'https://tfhub.dev/tensorflow/albert_en_preprocess/3',
      'electra_small':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'electra_base':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'experts_pubmed':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'experts_wiki_books':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
      'talking-heads_base':
          'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'}

    if random_state is not None:
      np.random.seed(random_state)
      tf.random.set_seed(random_state)


    self.tfhub_handle_encoder = map_name_to_handle[bert_model_name]
    self.tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]
    print(f'BERT model selected           : {self.tfhub_handle_encoder}')
    print(f'Preprocess model auto-selected: {self.tfhub_handle_preprocess}')

    self.model = self.__build_classifier_model(num_classes)

    
  def __build_classifier_model(self,num_classes):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(self.tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(self.tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(num_classes, activation='softmax', name='classifier')(net)
    return tf.keras.Model(text_input, net) 

  def plot_model(self):
    return tf.keras.utils.plot_model(self.model)


  def __data_preprocessing(self,data,batch_size):
    self.class_names = pd.factorize(data.label)[1]
    data.label = pd.factorize(data.label)[0]

    data = tf.data.Dataset.from_tensor_slices((data.text, data.label))
    data = data.batch(batch_size)
    data = data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    return data

  def from_path_data_preprocessing(self,path,batch_size):

    data = pd.read_csv(path)

    return self.__data_preprocessing(data,batch_size)

  def data_preprocessing(self,X,y,batch_size):
    if isinstance(X, np.ndarray):
      X =pd.DataFrame(X)
    if isinstance(y, np.ndarray):
      y =pd.DataFrame(y)
  
    data = pd.concat([X,y],axis=1)
    data.columns = ["text","label"]

    
    return self.__data_preprocessing(data,batch_size)


  def __train(self,data,learning_rate,batch_size = 64,epochs=10):
    ## loss
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = tf.metrics.SparseCategoricalAccuracy('accuracy')
    #optimizer

    steps_per_epoch = tf.data.experimental.cardinality(data).numpy()
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1*num_train_steps)

    optimizer = optimization.create_optimizer(init_lr=learning_rate,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')

    self.model.compile(optimizer=optimizer,loss=loss,metrics=metrics)
    return self.model.fit(x=data,epochs=epochs)


  def train(self,X,y, learning_rate,batch_size = 64,epochs=10):

    train_ds = self.data_preprocessing(X,y,batch_size = batch_size)
    self.__train(train_ds,learning_rate,batch_size = batch_size,epochs=epochs)



  def train_from_path(self,path, learning_rate,batch_size = 64, epochs=10):

    train_ds = self.from_path_data_preprocessing (path=path,batch_size = batch_size)
    self.__train(train_ds,learning_rate,batch_size = batch_size,epochs=epochs)


  def __predict_proba(self,X, batch_size=1):
    X = tf.data.Dataset.from_tensor_slices(X)


    X = X.batch(batch_size)

    X = X.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    return self.model.predict(X)

  def predict_label_proba(self,X, batch_size=1):
    y_pred = self.__predict_proba(X,batch_size=batch_size)
    prop = y_pred.max(axis=-1)
    lab =  np.array(list(map(lambda x:self.class_names.to_numpy()[x] ,y_pred.argmax(axis=-1)))) 

    return list(zip(lab,prop))

  def predict(self,X, batch_size=1):
    return self.__predict_proba(X,batch_size=batch_size).argmax(axis=-1)
  
  def evaluate_from_path(self,path):
    test = self.from_path_data_preprocessing (path=path,batch_size = 1)
    return self.model.evaluate(test)

  def evaluate(self,data):
    test = self.__data_preprocessing(data,batch_size=1)
    return self.model.evaluate(test)


    



    

if __name__ == "__main__":
    batch_size = 2
    seed = 42
    bert = Bert(num_classes = 2, random_state = seed)
    bert.train_from_path("/data/train_labeled.csv",learning_rate= 0.001,batch_size = batch_size,epochs = 2)
    test = pd.read_csv("/data/test.csv")
    print(bert.predict_label_proba(test.text[:10]))
