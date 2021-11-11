from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import torch
import argparse
import time
import datetime
import numpy as np
import random
from transformers import GPT2LMHeadModel, GPT2Config
from torch.utils.data import DataLoader, RandomSampler
from transformers import GPT2Tokenizer
from torch.utils.data import Dataset 
import gc

#NB! Inspiration for the code, as well as some whole lines, are taken from this notebook.
#It is part of a medium.com article
#https://github.com/mcelikkaya/medium_articles/blob/main/gtp2_training.ipynb
class customDataset(Dataset):

    def __init__(self, sequences, tokenizer: GPT2Tokenizer, max_length=1024):

        self.tokenizer = tokenizer 
        self.input_ids = []
        self.attn_masks = []

        for sequence in sequences:      
            encodings = tokenizer(sequence, truncation=True, max_length=min(max_length, 1024), padding="max_length")
                    
            self.input_ids.append(torch.tensor(encodings['input_ids']))
            self.attn_masks.append(torch.tensor(encodings['attention_mask']))
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx] 



class GPT2Tuner:
    """A wrapper for the GPT2 model taht allows the user to fine-tune it on a dataset and then generate sequences."""
    def __init__(self, data_path: str, device: str = "cpu", batch_size: int = 4, bos: str = '<bos>',
                eos: str = '<eos>', pad: str = '<pad>', cleaning: List = ["\r", "\n", "<br />"]) -> None:
        """Parameter:
            data_path(str): The path to where the training data is. Expects a .csv file
            
            device(str): Use "cuda" for GPU-training and "cpu" for CPU training.
            
            batch_size(int): The batch size to be used during training.
            
            bos(str): A token that symbolizes the begining of a sequence.
            
            eos(str): A token that symbolizes the end of a sequence.
            
            pad(str): A token that symbolizes a padding. Used at the end of a document to fulfill GPT2's requirement for 
            1024 tokens.
            
            cleaning(List): A list of words, symbols, characters to be removed from the document. """
        self.cleaning = cleaning
        self.bos = bos
        self.eos = eos
        self.pad = pad

        #Initialize the tokenizer to tokenize our sequences
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token=bos, eos_token=eos, pad_token=pad)
        #Clean the data from the .csv file
        self.sequences, self.labels = self.__clean_data(data_path)

        #The max length of any of the sequences in the dataset, used for setting the length in the dataset
        self.max_len = max([len(self.tokenizer.encode(s)) for s in self.sequences])
        
        #Create a new dataset-object for training. 1024 is the max size for GPT2
        self.dataset = customDataset(self.sequences, self.tokenizer, max_length=min(self.max_len, 1024))

        #Create a dataloader for easy training
        self.train_dataloader = DataLoader(self.dataset,  sampler = RandomSampler(self.dataset), batch_size = batch_size)    


        self.configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)

        #Load the GPT2 model and resize its embeddings
        self.model = GPT2LMHeadModel.from_pretrained("gpt2", config=self.configuration)
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.device = torch.device(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.0005)
        self.model.to(device)


    def __clean_data(self, data_path: str):
        """Cleans the data in the way specified in the cleaning list when initializing the tuner. 
        
        Parameters:
            data_path(str): The path to the data. Excepts a .csv file
            
        Returns:
            sequences(List): A list with all the cleaned sequences.
            labels(List): A list containing the unique labels."""
        sequences = []
        labels = []
        df = pd.read_csv(data_path)

        #Turn the sequences into the correct format to pass into GPT2.
        #We give it the label, say start of sequence, give it the sequence, and then the end of the sequence
        #GPT2 takes only 1024 tokens, so we limit the text to 1021
        for text, label in zip(df.iloc[:,0], df.iloc[:,1]):
            label = str(label)
            sequences.append(str(label) + self.bos + text + self.eos)
            labels.append(label)
        
        #Clean the sequences
        for cleaning in self.cleaning:
            sequences = [s.replace(cleaning, "") for s in sequences]

        return sequences, list(set(labels))


    def __format_time(self, elapsed: int) -> str:
        """Format the time to a pretty format.
        Parameters:
            elapsed(int): The elapsed time.
        Returns:
            A string representing the time"""
        return str(datetime.timedelta(seconds=int(round((elapsed)))))

    def __process_one_batch(self, batch: customDataset) -> torch.Tensor:
        """Takes in a batch, processes it, and returns the results from the model.
        Parameters:
            batch(customDataset): A data point (or multiple.
            
        Returns:
            The outputs of the model"""
        b_input_ids = batch[0].to(self.device)
        b_labels = batch[0].to(self.device)
        b_masks = batch[1].to(self.device)
        outputs  = self.model(b_input_ids,  attention_mask = b_masks, labels=b_labels)
        return outputs

    def train(self, epochs: int) -> None:
        """Fine tune the GPT2 model.
        Parameters:
            epochs(int): The number of epochs to train for"""

        #Collect the data because this takes so much memory..
        gc.collect()
        
        #Set the model to training
        self.model.train()
        for _ in range(epochs):
            t0 = time.time()
            total_train_loss = 0
            #Normal standard basic training loop stuff
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

    
    def save_sequences(self, num_to_gen: int, path: str = "samples.txt") -> None:
        """Generates a number of new sequences for each label equal to num_to_gen.
        Parameters:
            num_to_gen(int): Number of sequences to create per label"""
        gc.collect()
        self.model.eval()
        for label in self.labels:
            #We feed it label + start of sequence
            input_seq = str(label) + " " + self.bos
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
            #Save the generated sequences
            with open(path, "a") as f:
                for sample_output in sample_outputs:
                    #Do this weird hack so that we are guaranteed label + space at the start of the sequence, so we can get the label easily
                    #later
                    seq = str(label) + " " + self.tokenizer.decode(sample_output, skip_special_tokens=True).replace("\n", "")[len(str(label)):]
                    #Only save files that have atleast 10 characters. Otherwise a lot of empty sentences will be saved
                    #in some cases
                    if len(seq[len(str(label))+1:]) > 10: 
                        f.write(seq+"\n")


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
    parser.add_argument('--repeat_num', default='1', help='Number to repeat sequence generation. Used to get around out of memory errors')

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
    print("Generating sequences")
    for i in range(int(args.repeat_num)):
        tuner.save_sequences(int(args.samples_per_class), path=args.output_dir + "/" + args.output_name)
    print("Finished generating sequences.")
