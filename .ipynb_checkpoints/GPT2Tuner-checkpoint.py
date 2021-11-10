from typing import List
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import torch
from customDataset import customDataset
import argparse
import time
import datetime
import numpy as np
import random
from transformers import GPT2LMHeadModel, GPT2Config
from torch.utils.data import DataLoader, RandomSampler
from transformers import GPT2Tokenizer
import gc


class GPT2Tuner:
    def __init__(self, data_path, device = "cpu", batch_size: int = 4, bos: str = '<bos>',
                eos: str = '<eos>', pad: str = '<pad>', cleaning: List = ["\r", "\n", "<br />"]) -> None:
        self.cleaning = cleaning
        self.bos = bos
        self.eos = eos
        self.pad = pad

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token=bos, eos_token=eos, pad_token=pad)
        self.sentences, self.labels = self.__clean_data(data_path)

        self.max_len = max([len(self.tokenizer.encode(s)) for s in self.sentences])
        
        #1024 is the max size for GPT2
        self.dataset = customDataset(self.sentences, self.tokenizer, max_length=min(self.max_len, 1024))

        self.train_dataloader = DataLoader(self.dataset,  sampler = RandomSampler(self.dataset), batch_size = batch_size)    

        self.configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)

        self.model = GPT2LMHeadModel.from_pretrained("gpt2", config=self.configuration)
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.device = torch.device(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.0005)
        self.model.to(device)


    def __clean_data(self, data_path):
        sentences = []
        labels = []
        df = pd.read_csv(data_path)
        for text, label in zip(df.iloc[:,0], df.iloc[:,1]):
            sentences.append(str(label) + self.bos + str(text) + self.eos)
            labels.append(label)

        for cleaning in self.cleaning:
            sentences = [s.replace(cleaning, "") for s in sentences]

        return sentences, list(set(labels))


    def __format_time(self, elapsed):
        return str(datetime.timedelta(seconds=int(round((elapsed)))))

    def __process_one_batch(self, batch):
        b_input_ids = batch[0].to(self.device)
        b_labels = batch[0].to(self.device)
        b_masks = batch[1].to(self.device)
        outputs  = self.model(b_input_ids,  attention_mask = b_masks, labels=b_labels)
        return outputs

    def train(self, epochs):
        gc.collect()

        self.model.train()
        for _ in range(epochs):
            t0 = time.time()
            total_train_loss = 0
            for batch in self.train_dataloader:
                    
                    self.model.zero_grad()        
                    outputs = self.__process_one_batch( batch)
                    loss = outputs[0]  
                    batch_loss = loss.item()
                    total_train_loss += batch_loss

                    loss.backward()
                    self.optimizer.step()

                    
            avg_train_loss = total_train_loss / len(self.train_dataloader)  
            print("avg_train_loss",avg_train_loss)  
            elapsed_time = self.__format_time(time.time() - t0)
            print("elapsed time for 1 training epoch : ",elapsed_time)

    
    def save_sentences(self, num_to_gen, path = "samples.txt"):
        gc.collect()
        self.model.eval()
        for label in self.labels:
            input_seq = label + " " + self.bos
            generated = torch.tensor(self.tokenizer.encode(input_seq)).unsqueeze(0)
            generated = generated.to(self.device)
            sample_outputs = self.model.generate(
                                        generated, 
                                        do_sample=True,   
                                        top_k=30, 
                                        max_length = 250,
                                        top_p=0.90, 
                                        num_return_sequences=num_to_gen
                                        )

            with open(path, "a") as f:
                for sample_output in sample_outputs:
                    f.write(self.tokenizer.decode(sample_output, skip_special_tokens=True).replace("\n", "")+"\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters', prefix_chars='-')
    parser.add_argument('--train_data_path', default='./data/training_subdata.csv', help='Data path of training file')
    parser.add_argument('--output_dir', default='./data', help='Sample output directory')

    parser.add_argument('--output_name', default='generated_samples.txt', help='Sample file name')
    parser.add_argument('--device', default='cpu', help='cpu or cuda')
    parser.add_argument('--batch_size', default='8', help='Batch size for training')
    parser.add_argument('--epochs', default='2', help='Number of epochs')
    parser.add_argument('--samples_per_class', default='100', help='Number of datapoints to genereate for each class')
    parser.add_argument('--torch_seed', default='None', help='Seed to set with torch')
    parser.add_argument('--numpy_seed', default='None', help='Seed to set with numpy')
    parser.add_argument('--random_seed', default='None', help='Seed to set with random')

    args = parser.parse_args()
    if args.torch_seed != "None":
        torch.manual_seed(int(args.torch_seed))
    if args.numpy_seed != "None":
        np.random.seed(int(args.numpy_seed))
    if args.random_seed != "None":
        random.seed(int(args.random_seed))

    tuner = GPT2Tuner(data_path=args.train_data_path, device = args.device, batch_size=int(args.batch_size))
    print("Starting training:")
    tuner.train(int(args.epochs))
    print("Generating sentences")
    tuner.save_sentences(int(args.samples_per_class), path=args.output_dir + "/" + args.output_name)
    print("Finished generating sentences.")
