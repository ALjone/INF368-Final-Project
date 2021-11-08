from GPT2Tuner import GPT2Tuner
import pandas as pd
import torch
import random
from Bert import Bert
import gc

class Classifier:
    def __init__(self, data_path) -> None:
        pass
     
    def train_epoch(self):
        pass

    def predict(self, dataPoint):
        l = ["neg", "pos"]
        random.shuffle(l)
        return (l[0], random.random())

class Lambada:
    def __init__(self, data_path: str, batch_size: int = 1, device: torch.device = torch.device("cuda"), G_epochs: int = 20, h_epochs: int = 20, sentences_per_label: int = 100, save_path: str = "samples.txt") -> None:
        G = GPT2Tuner(data_path, device = device, batch_size=batch_size)
        self.h = Classifier(2)
        self.data_path = data_path
        self.save_path = save_path
        self.h_epochs = h_epochs
        self.sentences_per_label = sentences_per_label
        G.train(G_epochs)
        for _ in range(int((sentences_per_label*10)/50)):
            G.save_sentences(50, save_path)
        gc.collect()
    
    def get_sentences(self):
        h = Bert(2)
        h.train_from_path(self.data_path, 0.001, epochs=self.h_epochs)

        with open(self.save_path, "r") as file:
            sentences = file.readlines()
        
        #TODO get the X best of each label, not just top X best total

        #TODO also gather data in dict before putting it in dataframe
        #https://stackoverflow.com/questions/57000903/what-is-the-fastest-and-most-efficient-way-to-append-rows-to-a-dataframe
        labels = []
        cleaned_sentences = []
        for sentence in sentences:
            sentence_parts = sentence.split(maxsplit = 1)
            labels.append(sentence_parts[0])
            cleaned_sentences.append(sentence_parts[1])

        df = pd.DataFrame(columns=["Sentence", "True label", "Predicted label", "Confidence"])
        for label, sentence in zip(labels, cleaned_sentences):
            predictions = h.predict(sentence)
            new_row = {"Sentence": sentence, "True label": label, "Predicted label" : predictions[0], "Confidence" : predictions[1]}
            df = df.append(new_row, ignore_index=True)
        
        candidates = df[df["True label"] == df["Predicted label"]]
        candidates.sort_values("Confidence", ascending=False, inplace=True)
        #Make sure we have enough labels
        if candidates.size >= self.sentences_per_label*len(set(labels)):
            return candidates.head(self.sentences_per_label*len(set(labels)))
        else:
            print("Not enough correct labels synthesized, returning what little we have")
            return candidates